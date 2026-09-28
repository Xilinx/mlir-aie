# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s

import numpy as np

from aie.dialects._aie_enum_gen import AIETileType
from aie.iron import ObjectFifo, Worker
from aie.iron.device import AnyComputeTile, AnyMemTile, AnyShimTile, Tile

VECTOR_TYPE = np.ndarray[(16,), np.dtype[np.int32]]


def make_link(kind, **kwargs):
    fifo = ObjectFifo(VECTOR_TYPE)
    if kind == "join":
        handle = fifo.prod()
        handle.join([0, 8], **kwargs)
    else:
        handle = fifo.cons()
        if kind == "split":
            handle.split([0, 8], **kwargs)
        else:
            handle.forward(**kwargs)
    return handle.endpoint


def test_shared_worker_tile(kind):
    for coords in ({}, {"col": 0}, {"col": 0, "row": 2}):
        tile = Tile(**coords, tile_type=AIETileType.CoreTile)
        worker = Worker(None, tile=tile)
        first = make_link(kind, tile=tile)
        second = make_link(kind, tile=tile)
        assert first.tile is second.tile is worker.tile is tile
        assert tile.tile_type == AIETileType.CoreTile


def test_explicit_tile_identity(kind):
    for coords in ({}, {"col": 0}, {"col": 0, "row": 1}, {"col": 0, "row": 2}):
        tile = Tile(**coords, packet_type=2, packet_id=7)
        link = make_link(kind, tile=tile)
        assert link.tile is tile
        assert tile.packet_type == 2 and tile.packet_id == 7
        expected = None if "row" in coords else AIETileType.MemTile
        assert tile.tile_type == expected


def test_default_isolation(kind):
    for kwargs in ({}, {"tile": None}):
        first = make_link(kind, **kwargs)
        second = make_link(kind, **kwargs)
        assert first.tile is not second.tile
        assert first.tile is not AnyMemTile
        assert first.tile.tile_type == AIETileType.MemTile

    for default in (AnyMemTile, AnyComputeTile, AnyShimTile):
        first = make_link(kind, tile=default)
        second = make_link(kind, tile=default)
        assert first.tile is not second.tile
        assert first.tile is not default
        assert first.tile.tile_type == default.tile_type
        first.tile.col = 3
        first.tile.row = 2
        assert second.tile.col is None and second.tile.row is None
        assert default.col is None and default.row is None


for kind in ("split", "join", "forward"):
    test_shared_worker_tile(kind)
    test_explicit_tile_identity(kind)
    test_default_isolation(kind)
print("All ObjectFifo tile sharing tests passed")
