# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test that a Buffer pinned to an explicit tile may be referenced by a
second Worker (valid cross-core neighbor-L1 read): the first referencing
Worker owns/places it, later ones are non-owning readers. Auto-placed
(no-tile) buffers are still forbidden from being shared across Workers."""

import numpy as np

from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2Col1, Tile
from aie.dialects._aie_enum_gen import AIETileType
from aie.passmanager import PassManager

buf_ty = np.ndarray[(64,), np.dtype[np.int32]]


def test_explicit_tile_buffer_shared_across_workers():
    """A Buffer pinned to an explicit tile can be passed to two Workers."""
    tile = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    shared = Buffer(tile=tile, type=buf_ty, name="shared_buf")

    producer = Worker(None, [shared], tile=tile)
    consumer_tile = Tile(col=0, row=3, tile_type=AIETileType.CoreTile)
    consumer = Worker(None, [shared], tile=consumer_tile)

    # First worker owns/places it; the second is a non-owning reader.
    assert shared._owner_worker is producer, "First worker should own the buffer"
    assert shared in producer._buffers
    assert shared in consumer._buffers
    assert shared._tile is tile, "Explicit tile placement must be honored"


def test_auto_placed_buffer_sharing_still_raises():
    """A Buffer with no explicit tile shared across Workers is ambiguous."""
    shared = Buffer(type=buf_ty, name="auto_buf")

    w1 = Worker(None, [shared])
    raised = False
    try:
        Worker(None, [shared])
    except ValueError as e:
        raised = True
        assert "no explicit tile" in str(e)
    assert raised, "Sharing an auto-placed Buffer across Workers must raise"


def test_shared_buffer_resolves_to_cross_core_access():
    """End-to-end: a Buffer pinned to a producer tile and read by a consumer
    Worker on a neighbor tile must resolve to a single aie.buffer declared on
    the producer tile, written by the producer core and read by the consumer
    core (the cross-core neighbor-L1 access this change enables)."""
    prod_tile = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    cons_tile = Tile(col=0, row=3, tile_type=AIETileType.CoreTile)

    # Pinned to the producer tile; the consumer reads it from the neighbor tile.
    shared = Buffer(buf_ty, name="shared_l1", tile=prod_tile)
    of_out = ObjectFifo(buf_ty, name="out")

    def prod_fn(buf):
        buf[0] = 1

    def cons_fn(buf, of_out):
        elem = of_out.acquire(1)
        x = buf[0]
        of_out.release(1)

    producer = Worker(prod_fn, [shared], tile=prod_tile)
    consumer = Worker(cons_fn, [shared, of_out.prod()], tile=cons_tile)

    rt = Runtime()
    with rt.sequence(buf_ty) as out:
        rt.start(producer, consumer)
        rt.drain(of_out.cons(), out, wait=True)

    print(Program(NPU2Col1(), rt).resolve_program())


def test_unpinned_consumer_is_steered_to_owner_neighbor():
    """End-to-end through --aie-place-tiles: when the owner is pinned but the
    consumer Worker is left unpinned, the SequentialPlacer's buffer-adjacency
    constraint (buildBufferAdjacency + isLegalMemAffinity) must steer the
    consumer onto a tile whose L1 is shared with the owner. Owner pinned at
    (0, 2); the consumer must land on its N neighbor (0, 3)."""
    prod_tile = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)

    shared = Buffer(buf_ty, name="shared_l1", tile=prod_tile)
    of_out = ObjectFifo(buf_ty, name="out")

    def prod_fn(buf):
        buf[0] = 1

    def cons_fn(buf, of_out):
        elem = of_out.acquire(1)
        x = buf[0]
        of_out.release(1)

    producer = Worker(prod_fn, [shared], tile=prod_tile)  # owner pinned (0, 2)
    consumer = Worker(cons_fn, [shared, of_out.prod()])  # unpinned

    rt = Runtime()
    with rt.sequence(buf_ty) as out:
        rt.start(producer, consumer)
        rt.drain(of_out.cons(), out, wait=True)

    module = Program(NPU2Col1(), rt).resolve_program()
    pm = PassManager.parse(
        "builtin.module(aie.device(aie-place-tiles))", context=module.context
    )
    pm.run(module.operation)
    print(module)


test_explicit_tile_buffer_shared_across_workers()
test_auto_placed_buffer_sharing_still_raises()

# A single buffer, declared once on the producer tile (0, 2)...
# CHECK: %[[PTILE:.+]] = aie.logical_tile<CoreTile>(0, 2)
# CHECK: %[[CTILE:.+]] = aie.logical_tile<CoreTile>(0, 3)
# CHECK: %[[BUF:.+]] = aie.buffer(%[[PTILE]]) {sym_name = "shared_l1"}
# ...written by the producer core...
# CHECK: aie.core(%[[PTILE]])
# CHECK: memref.store {{.*}}, %[[BUF]]
# ...and read by the consumer core on the neighbor tile.
# CHECK: aie.core(%[[CTILE]])
# CHECK: memref.load %[[BUF]]
test_shared_buffer_resolves_to_cross_core_access()

# After --aie-place-tiles, the pinned owner is at (0, 2) and the unpinned
# consumer is steered onto its memory-affinity neighbor (0, 3); the shared
# buffer and both cores reference the placed physical tiles. (Tiles are
# emitted in placement order, which lands (0, 3) before (0, 2).)
# CHECK: %[[CT:.+]] = aie.tile(0, 3)
# CHECK: %[[PT:.+]] = aie.tile(0, 2)
# CHECK: %[[PBUF:.+]] = aie.buffer(%[[PT]]) {sym_name = "shared_l1"}
# CHECK: aie.core(%[[PT]])
# CHECK: aie.core(%[[CT]])
# CHECK: memref.load %[[PBUF]]
# CHECK-NOT: aie.logical_tile
test_unpinned_consumer_is_steered_to_owner_neighbor()
