# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# REQUIRES: peano, hrxxclbinutil

"""Tiled copy with DispatchTime tile count and start tile.

The taps are computed by taplib *inside* the runtime sequence body (a staged
grid index into a Layout partition). The design compiles once; its host-side
C++ transaction builder is then driven at several (start, n) pairs and the DMA
events it produces are compared with a fully static specialization of the
same generator. A dispatch that steps outside the buffer is refused by the
`npu.require` guards taplib emitted. No NPU is needed.
"""

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import Layout
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
from aie.utils.compile.jit._dispatch_bridge import DispatchBridge
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
        chunks = Layout.full((1, max_tiles * tile_size)).partition(max_tiles)
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
print("guards emitted:", "aiex.npu.require" in dyn.as_mlir())
# CHECK: guards emitted: True
dyn.compile()
bridge = DispatchBridge(dyn.get_dispatch_lib_path(), dyn.compilable.dispatch_params)

for start, n in ((0, 3), (1, 6), (2, 2), (0, MAX_TILES), (5, 3)):
    words = bridge.generate({"n_tiles": n, "start_tile": start})
    _, static_insts = tiled_copy.specialize(n_tiles=n, start_tile=start).compile()
    static = np.fromfile(static_insts, dtype=np.uint32)
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
    bridge.generate({"n_tiles": 4, "start_tile": 6})
    print("out-of-range dispatch: accepted")
except HostRuntimeError as e:
    print("out-of-range dispatch: refused:", e)
# The guard taplib emitted for the grid index names the reason.
# CHECK: out-of-range dispatch: refused: dispatch refused for DispatchTime[T] value(s) {'n_tiles': 4, 'start_tile': 6}: grid index {{[0-9]}} exceeds the grid
