# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# REQUIRES: peano, hrxxclbinutil

"""Tiled copy with DispatchTime tile count and start tile, and a repeated read.

The taps are computed by taplib *inside* the runtime sequence body (a staged
grid index into a TensorAccessPattern partition). The design compiles once;
`instructions()` then drives its host-side C++ transaction builder at several
(start, n) pairs, and the DMA events it produces are compared with a fully
static specialization of the same generator. A dispatch that steps outside
the buffer is refused by the `cf.assert` guards taplib emitted. No NPU is
needed.
"""

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    CompileTime,
    DispatchTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1
from aie.utils.hostruntime.hostruntime import HostRuntimeError
from aie.utils.txn_trace import compare, trace

iron.set_current_device(NPU1Col1())
TILE_SIZE, MAX_TILES = 256, 8


@iron.jit
def tiled_copy(
    a: In,
    b: Out,
    *,
    n_tiles: DispatchTime[np.int32] = 3,
    start_tile: DispatchTime[np.int32] = 0,
    tile_size: CompileTime[int] = TILE_SIZE,
    max_tiles: CompileTime[int] = MAX_TILES,
):
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    max_ty = np.ndarray[(max_tiles * tile_size,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name="of_in", depth=2)
    of_out = ObjectFifo(tile_ty, name="of_out", depth=2)

    def core_fn(in_cons, out_prod):
        elem_in = in_cons.acquire(1)
        elem_out = out_prod.acquire(1)
        for i in range_(tile_size):
            elem_out[i] = elem_in[i]
        in_cons.release(1)
        out_prod.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    def seq(a_h, b_h, start, n, in_prod, out_cons):
        # The buffer as max_tiles equal chunks; the chunk index is staged
        # arithmetic (start + loop iv), so the tap's offset is too.
        chunks = TensorAccessPattern.full((max_tiles * tile_size,)).partition(
            max_tiles
        )
        for tile in range_(n):  # an index counter; the tiler casts it
            tap = chunks[start + tile]
            tg = TaskGroup()
            out_cons.drain(b_h, tap=tap, wait=True, group=tg)
            in_prod.fill(a_h, tap=tap, group=tg)
            tg.finish()

    rt = Runtime(
        seq, [max_ty, max_ty, start_tile, n_tiles, of_in.prod(), of_out.cons()]
    )
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


dyn = tiled_copy.specialize()  # both scalars stay dispatch-time
print("guards emitted:", "cf.assert" in dyn.as_mlir())
# CHECK: guards emitted: True

for start, n in ((0, 3), (1, 6), (2, 2), (0, MAX_TILES), (5, 3)):
    words = dyn.instructions(n_tiles=n, start_tile=start)
    static = tiled_copy.specialize(n_tiles=n, start_tile=start).instructions()
    diffs = compare(words, static, names=("dynamic", "static"))
    events = trace(words)
    pushes = [e for e in events if e.kind == "push" and e.direction == "MM2S"]
    offsets = [e.bd.address[2] // (4 * TILE_SIZE) for e in pushes]
    print(
        f"start={start} n={n}: {len(pushes)} tiles from {offsets[0]}..{offsets[-1]}"
        f" equivalent={not diffs}"
    )
    for d in diffs:
        print(d)
# CHECK: start=0 n=3: 3 tiles from 0..2 equivalent=True
# CHECK: start=1 n=6: 6 tiles from 1..6 equivalent=True
# CHECK: start=2 n=2: 2 tiles from 2..3 equivalent=True
# CHECK: start=0 n=8: 8 tiles from 0..7 equivalent=True
# CHECK: start=5 n=3: 3 tiles from 5..7 equivalent=True

try:
    dyn.instructions(n_tiles=4, start_tile=6)
    print("out-of-range dispatch: accepted")
except HostRuntimeError as e:
    print("out-of-range dispatch: refused:", e)
# The guard taplib emitted for the index names the reason.
# CHECK: out-of-range dispatch: refused: dispatch refused for DispatchTime[T] value(s) {'n_tiles': 4, 'start_tile': 6}: index exceeds the dimension


@iron.jit
def repeated_read(
    a: In,
    b: Out,
    *,
    reps: DispatchTime[np.int64] = 2,
    tile_size: CompileTime[int] = TILE_SIZE,
):
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name="of_in", depth=2)
    of_out = ObjectFifo(tile_ty, name="of_out", depth=2)

    def core_fn(in_cons, out_prod):
        elem_in = in_cons.acquire(1)
        elem_out = out_prod.acquire(1)
        for i in range_(tile_size):
            elem_out[i] = elem_in[i]
        in_cons.release(1)
        out_prod.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    def seq(a_h, b_h, n, in_prod, out_cons):
        # The outer dimension of a repeated tap is the queue's repeat count.
        tap = TensorAccessPattern.full((1, tile_size)).repeat(n)
        tg = TaskGroup()
        out_cons.drain(b_h, tap=tap, wait=True, group=tg)
        in_prod.fill(a_h, tap=tap, group=tg)
        tg.finish()

    rt = Runtime(seq, [tile_ty, tile_ty, reps, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


dyn_reps = repeated_read.specialize()
for reps in (1, 65, 256):
    words = dyn_reps.instructions(reps=reps)
    static = repeated_read.specialize(reps=reps).instructions()
    repeats = {e.repeat for e in trace(words) if e.kind == "push"}
    print(f"reps={reps}: repeat {repeats} equivalent={not compare(words, static)}")
# CHECK: reps=1: repeat {0} equivalent=True
# CHECK: reps=65: repeat {64} equivalent=True
# CHECK: reps=256: repeat {255} equivalent=True

# A repeat count is never truncated into its field. Below 1, or past the
# queue's 8-bit repeat count however wide, the dispatch is refused.
for reps in (257, 0, 2**31 + 1, 2**32 + 2):
    try:
        dyn_reps.instructions(reps=reps)
        print(f"reps={reps}: accepted")
    except HostRuntimeError as e:
        print(f"reps={reps}: refused:", str(e).split("}: ", 1)[1])
# CHECK: reps=257: refused: a runtime DMA repeat count must be in [1:256]
# CHECK: reps=0: refused: repeat count must be >= 1, got <runtime>
# CHECK: reps=2147483649: refused: a runtime DMA repeat count must be in [1:256]
# CHECK: reps=4294967298: refused: a runtime DMA repeat count must be in [1:256]
