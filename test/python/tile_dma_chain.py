# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# RUN: %python %s | aie-opt --aie-place-tiles --aie-split-long-repeats \
# RUN:   | FileCheck %s --check-prefix=SPLIT

"""DmaEndpoint.task configures a chain of mem tile descriptors from inside the
runtime sequence as one task, and Task.start pushes that chain as often as
needed, with a different pass count if asked. Together they let a sequence
program a ring buffer on a mem tile -- one descriptor per slot, handed to and
from a compute tile through locks -- without reconfiguring it for every batch
of passes."""

import numpy as np

from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Acquire,
    Bd,
    BdIteration,
    Buffer,
    DmaEndpoint,
    Lock,
    Program,
    Release,
    Runtime,
)
from aie.iron.device import NPU2Col1, Tile

SLOTS = 3
SLOT = 512


def emit_ring(bad=None):
    buf_ty = np.ndarray[(SLOTS * SLOT,), np.dtype[np.int32]]
    host_ty = np.ndarray[(SLOT,), np.dtype[np.int32]]

    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    other_tile = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    ring = Buffer(tile=mem_tile, type=buf_ty, name="ring")
    stray = Buffer(tile=other_tile, type=host_ty, name="stray")
    prod = [Lock(mem_tile, init=2, name=f"prod{i}") for i in range(SLOTS)]
    cons = [Lock(mem_tile, init=0, name=f"cons{i}") for i in range(SLOTS)]

    def slot_bds():
        bds = [
            Bd(
                ring,
                tap=slot,
                acquires=[Acquire(prod[i], value=2)],
                releases=[Release(cons[i], value=2)],
            )
            for i, slot in enumerate(
                TensorAccessPattern.full((SLOTS * SLOT,)).partition(SLOTS)
            )
        ]
        if bad == "next":
            bds[0].next = "self"
        elif bad == "iteration":
            bds[1].tap = TensorAccessPattern(
                (SLOTS * SLOT,), 0, [2, 2, 2, 8], [1, 16, 64, 128]
            )
            bds[1].iteration = BdIteration(size=2, stride=SLOT)
        elif bad == "tile":
            bds[2] = Bd(stray)
        elif bad == "empty":
            bds = []
        return bds

    def sequence(_host):
        # 600 passes over the ring: more than one queue push carries, so the
        # compiler issues the start as several pushes of the same task.
        ring_in = DmaEndpoint(mem_tile, DMAChannelDir.S2MM, 0)
        task = ring_in.task(*slot_bds(), runs=600).start()
        # Four more passes: the chain is already written, so this is one push.
        task.start(repeat_count=3)
        task.free()

    rt = Runtime(sequence, [host_ty])
    module = Program(NPU2Col1(), rt).resolve_program()
    module.operation.verify()
    return module


# One task, one descriptor per slot, each taking and handing back its slot's
# locks. The chain is linear and ends after the last slot.
# CHECK: %[[T:.*]] = aiex.dma_configure_task(%{{.*}}, S2MM, 0) {
# CHECK:   aie.use_lock(%prod0, AcquireGreaterEqual, %{{.*}})
# CHECK:   aie.dma_bd(%{{.*}} : memref<1536xi32> len = 512)
# CHECK:   aie.use_lock(%cons0, Release, %{{.*}})
# CHECK:   aie.next_bd ^bb1

# CHECK: ^bb1:
# CHECK:   aie.dma_bd(%{{.*}} : memref<1536xi32> offset = 512 len = 512)
# CHECK:   aie.next_bd ^bb2
# CHECK: ^bb2:
# CHECK:   aie.dma_bd(%{{.*}} : memref<1536xi32> offset = 1024 len = 512)
# CHECK:   aie.end
# CHECK: } {repeat_count = 599 : i32}
# CHECK: aiex.dma_start_task(%[[T]])
# CHECK-NEXT: aiex.dma_start_task(%[[T]]) {repeat_count = 3 : i32}
# CHECK-NEXT: aiex.dma_free_task(%[[T]])

# 600 passes = 256 + 256 + 88, then the restart's 4.
# SPLIT: aiex.dma_start_task(%[[T:.*]]) {no_token, repeat_count = 255 : i32}
# SPLIT-NEXT: aiex.dma_start_task(%[[T]]) {no_token, repeat_count = 255 : i32}
# SPLIT-NEXT: aiex.dma_start_task(%[[T]]) {repeat_count = 87 : i32}
# SPLIT-NEXT: aiex.dma_start_task(%[[T]]) {repeat_count = 3 : i32}
# SPLIT-NOT: aiex.dma_start_task
print(emit_ring())

# Printed as MLIR comments so the SPLIT run still parses the module above.
for bad in ("next", "iteration", "tile", "empty"):
    try:
        emit_ring(bad)
    except Exception as e:
        print(f"// {bad}: {type(e).__name__}: {e}".replace("\n", " "))
# CHECK: // next: ValueError: Bd 0 of a task on S2MM channel 0 on {{.*}} sets next='self'; a task runs its Bds in order and ends after the last.
# CHECK: // iteration: {{.*}}Cannot give more than 3 dimensions for step sizes and wraps alongside the iteration attribute on this tile (got 4 dimensions).
# CHECK: // tile: ValueError: A task on S2MM channel 0 on {{.*}} was given a buffer on {{.*}}; a tile's DMA can only address buffers on that tile.
# CHECK: // empty: ValueError: A task on S2MM channel 0 on {{.*}} needs at least one Bd.
