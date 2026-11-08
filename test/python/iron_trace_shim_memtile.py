# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s 2>&1 | FileCheck %s

# Program.enable_trace(shimtile_events=..., memtile_events=...) traces every
# shim and mem tile the design uses, not only its workers. Before, both lists
# were accepted and silently dropped, as configure_trace was handed compute
# tiles only.

import logging
import sys

import numpy as np

from aie import ir
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1, NPU1Col2, Tile
from aie.passmanager import PassManager
from aie.utils.trace.events import MemTileEvent, ShimTileEvent

logging.basicConfig(stream=sys.stdout, format="LOG: %(message)s")

ty = np.ndarray[(16,), np.dtype[np.int32]]
SHIM = [ShimTileEvent.DMA_S2MM_0_START_TASK]
MEM = [MemTileEvent.DMA_S2MM_SEL0_START_TASK]

PLACE_AND_TRACE = "builtin.module(aie.device(aie-place-tiles,aie-insert-trace-flows))"


def copy_worker(of_in, of_out):
    def core_fn(src, dst):
        dst.acquire(1)
        src.acquire(1)
        src.release(1)
        dst.release(1)

    return Worker(core_fn, [of_in.cons(), of_out.prod()], while_true=False)


def build(through_memtile, **events):
    of_in = ObjectFifo(ty, name="in")
    of_out = ObjectFifo(ty, name="out")
    if through_memtile:
        of_core_in = of_in.cons().forward(name="in_core")
    else:
        of_core_in = of_in
    worker = copy_worker(of_core_in, of_out)

    def sequence(a, c, in_h, out_h):
        in_h.fill(a)
        out_h.drain(c, wait=True)

    rt = Runtime(sequence, [ty, ty, of_in.prod(), of_out.cons()])
    prog = Program(NPU1Col1(), rt, workers=[worker])
    prog.enable_trace(trace_size=8192, workers=[worker], **events)
    return prog.resolve_program()


def build_two_columns(**events):
    """One copy per column, each fed and drained through its own shim."""
    workers, ends = [], []
    for col in range(2):
        of_in = ObjectFifo(ty, name=f"in{col}")
        of_out = ObjectFifo(ty, name=f"out{col}")
        workers.append(copy_worker(of_in, of_out))
        shim = Tile(col, 0)
        ends += [of_in.prod(tile=shim), of_out.cons(tile=shim)]

    def sequence(a0, c0, a1, c1, in0, out0, in1, out1):
        in0.fill(a0)
        in1.fill(a1)
        out0.drain(c0, wait=True)
        out1.drain(c1, wait=True)

    rt = Runtime(sequence, [ty] * 4 + ends)
    prog = Program(NPU1Col2(), rt, workers=workers)
    prog.enable_trace(trace_size=8192, workers=workers, **events)
    return prog.resolve_program()


def place_and_trace(module):
    with ir.Context() as ctx, ir.Location.unknown():
        placed = ir.Module.parse(str(module))
        PassManager.parse(PLACE_AND_TRACE, ctx).run(placed.operation)
        return placed


# Each shim endpoint is its own logical tile, so each gets a trace. Placement
# puts both on shim (0, 0), and -aie-insert-trace-flows merges their identical
# traces into one, with one packet flow per traced unit.
# CHECK-LABEL: TEST: shim_and_mem_events
# CHECK: aie.trace @trace_core_1
# CHECK: aie.trace @trace_shim_2
# CHECK: aie.trace @trace_shim_3
# CHECK: aie.trace @trace_memtile_4
# CHECK-NOT: aie.trace @
print("\nTEST: shim_and_mem_events")
module = build(True, shimtile_events=SHIM, memtile_events=MEM)
print(module)

# CHECK: PLACED
# CHECK-DAG: aie.trace @trace_core_1(%{{.*}}tile_0_2)
# CHECK-DAG: aie.trace @trace_shim_2(%{{.*}}tile_0_0)
# CHECK-DAG: aie.trace @trace_memtile_4(%{{.*}}tile_0_1)
# CHECK-NOT: @trace_shim_3
# CHECK-COUNT-3: aie.packet_flow(
# CHECK-NOT: aie.packet_flow(
print("PLACED")
print(place_and_trace(module))

# Without the event lists the design traces only its worker, as before.
# CHECK-LABEL: TEST: no_event_lists
# CHECK: aie.trace @trace_core_1
# CHECK-NOT: aie.trace @
print("\nTEST: no_event_lists")
print(build(True))

# An event list for a tile the design does not use says so, and the rest of the
# trace is unaffected.
# CHECK-LABEL: TEST: no_mem_tile
# CHECK: LOG: memtile_events was set but the design uses no mem tile
# CHECK: aie.trace @trace_core_1
# CHECK: aie.trace @trace_shim_2
# CHECK: aie.trace @trace_shim_3
# CHECK-NOT: aie.trace @
print("\nTEST: no_mem_tile")
print(build(False, shimtile_events=SHIM, memtile_events=MEM))

# A design that uses two shim columns traces both of them.
# CHECK-LABEL: TEST: two_shim_columns
# CHECK: PLACED
# CHECK-DAG: aie.trace @{{trace_shim_[0-9]+}}(%{{.*}}tile_0_0)
# CHECK-DAG: aie.trace @{{trace_shim_[0-9]+}}(%{{.*}}tile_1_0)
# CHECK-COUNT-4: aie.packet_flow(
# CHECK-NOT: aie.packet_flow(
print("\nTEST: two_shim_columns")
module = build_two_columns(shimtile_events=SHIM)
print("PLACED")
print(place_and_trace(module))
