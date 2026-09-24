# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# RUN: %python %s | aie-opt --aie-place-tiles --aie-assign-runtime-sequence-bd-ids \
# RUN:   | FileCheck %s --check-prefix=SPLIT

"""tile_dma_chain configures a chain of mem tile descriptors from inside the
runtime sequence as one task, and Task.start pushes that chain again with a
different pass count. Together they let a sequence program a ring buffer on a
mem tile -- one descriptor per slot, handed to and from a compute tile through
locks -- without reconfiguring it for every batch of passes."""

import numpy as np

from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.iron import (
    Acquire,
    Bd,
    BdIteration,
    Buffer,
    Lock,
    Program,
    Release,
    Runtime,
    tile_dma_chain,
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
                offset=i * SLOT,
                length=SLOT,
                acquires=[Acquire(prod[i], value=2)],
                releases=[Release(cons[i], value=2)],
            )
            for i in range(SLOTS)
        ]
        if bad == "next":
            bds[0].next = "self"
        elif bad == "iteration":
            bds[1].iteration = BdIteration(size=2, stride=SLOT)
        elif bad == "tile":
            bds[2] = Bd(stray, length=SLOT)
        elif bad == "empty":
            bds = []
        return bds

    def sequence(_host):
        # 600 passes over the ring: more than one queue push carries, so the
        # compiler issues the start as several pushes of the same task.
        task = tile_dma_chain(
            mem_tile, DMAChannelDir.S2MM, 0, slot_bds(), repeat_count=599
        )
        # Four more passes: the chain is already written, so this is one push.
        task.start(repeat_count=3)
        task.free()

    rt = Runtime(sequence, [host_ty])
    rt.add_buffer(ring)
    if bad == "tile":
        rt.add_buffer(stray)
    for lk in prod + cons:
        rt.add_lock(lk)
    return Program(NPU2Col1(), rt).resolve_program()


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
    except ValueError as e:
        print(f"// {bad}: {e}")
# CHECK: // next: tile_dma_chain Bd 0 sets next='self'; a runtime chain runs its Bds in list order and ends after the last.
# CHECK: // iteration: tile_dma_chain Bd 1 sets iteration; use the chain's repeat_count or one Bd per sub-buffer instead.
# CHECK: // tile: tile_dma_chain on {{.*}} was given a buffer on {{.*}} (Bd 2); a tile's DMA can only address buffers on that tile (only a shim BD reaches host memory).
# CHECK: // empty: tile_dma_chain needs at least one Bd
