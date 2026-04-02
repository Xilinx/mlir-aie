# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
from aie.iron.device import NPU1Col1, NPU1Col2, NPU1, NPU2
from aie.iron.device import Tile


@pytest.fixture(params=[NPU1Col1, NPU1Col2, NPU1, NPU2])
def device(request):
    return request.param()


def test_rows_cols(device):
    assert device.rows == device._tm.rows()
    assert device.cols == device._tm.columns()


def test_get_tiles(device):
    shim_tiles = device.get_shim_tiles()
    mem_tiles = device.get_mem_tiles()
    compute_tiles = device.get_compute_tiles()
    assert all(t.row == 0 for t in shim_tiles)
    assert all(device._tm.is_shim_noc_or_pl_tile(t.col, t.row) for t in shim_tiles)
    assert all(device._tm.is_mem_tile(t.col, t.row) for t in mem_tiles)
    assert all(device._tm.is_core_tile(t.col, t.row) for t in compute_tiles)
    assert (
        len(shim_tiles) + len(mem_tiles) + len(compute_tiles)
        == device.rows * device.cols
    )


def test_legal_mem_affinity(device):
    # Test single tile
    assert device.is_mem_accessible(Tile(1, 2), [Tile(1, 2)])

    # Test adjacent compute tiles
    assert device.is_mem_accessible(Tile(1, 2), [Tile(1, 3)])
    assert device.is_mem_accessible(Tile(1, 2), [Tile(1, 3)])

    # Test adjacent memory tiles
    assert device.is_mem_accessible(Tile(1, 1), [Tile(2, 1)])
    assert device.is_mem_accessible(Tile(2, 1), [Tile(1, 1)])

    # Test non-adjacent tiles
    assert not device.is_mem_accessible(Tile(1, 2), [Tile(3, 4)])

    # Test diagonal compute tiles
    assert not device.is_mem_accessible(Tile(1, 2), [Tile(2, 3)])

    # Test adjacent shim and mem tiles
    assert not device.is_mem_accessible(Tile(0, 0), [Tile(0, 1)])
    assert not device.is_mem_accessible(Tile(0, 1), [Tile(0, 0)])

    # Test adjacent mem and compute tiles
    assert not device.is_mem_accessible(Tile(0, 1), [Tile(0, 2)])
    assert not device.is_mem_accessible(Tile(0, 2), [Tile(0, 1)])

    # Test multiple adjacent compute tiles
    assert not device.is_mem_accessible(Tile(1, 2), [Tile(1, 3), Tile(1, 4)])
    assert device.is_mem_accessible(Tile(1, 3), [Tile(1, 2), Tile(1, 4)])
    assert not device.is_mem_accessible(Tile(1, 4), [Tile(1, 2), Tile(1, 3)])

    # Test multiple non-adjacent compute tiles
    assert not device.is_mem_accessible(Tile(1, 2), [Tile(1, 3), Tile(1, 5)])

    # Test multiple tiles of different types
    assert not device.is_mem_accessible(Tile(0, 0), [Tile(1, 2), Tile(1, 1)])

    # Test ValueError for unplaced tiles
    with pytest.raises(ValueError):
        device.is_mem_accessible(Tile(), [Tile(1, 2)])
    with pytest.raises(ValueError):
        device.is_mem_accessible(Tile(1, 2), [Tile()])


def test_unplaced_tile_queries(device):
    """Device query methods must raise ValueError for unplaced tiles."""
    unplaced = Tile()
    with pytest.raises(ValueError):
        device.get_num_source_switchbox_connections(unplaced)
    with pytest.raises(ValueError):
        device.get_num_dest_switchbox_connections(unplaced)
    with pytest.raises(ValueError):
        device.get_num_source_shim_mux_connections(unplaced)
    with pytest.raises(ValueError):
        device.get_num_dest_shim_mux_connections(unplaced)
    with pytest.raises(ValueError):
        device.get_num_connections(unplaced, output=True)
    with pytest.raises(ValueError):
        device.get_num_connections(unplaced, output=False)


def test_tile_type_coordinate_mismatch(device):
    """resolve_tile must reject tiles where tile_type contradicts coordinates."""
    from aie.dialects._aie_enum_gen import AIETileType

    # Shim tile coordinates (row 0) with CoreTile type
    bad_tile = Tile(0, 0, tile_type=AIETileType.CoreTile)
    with pytest.raises(ValueError, match="coordinates indicate"):
        device.resolve_tile(bad_tile)

    # Compute tile coordinates with ShimNOCTile type
    bad_tile2 = Tile(0, 2, tile_type=AIETileType.ShimNOCTile)
    with pytest.raises(ValueError, match="coordinates indicate"):
        device.resolve_tile(bad_tile2)


def test_tile_type_inferred_from_coordinates(device):
    """_infer_tile_type must return the correct tile type for known coordinates."""
    from aie.dialects._aie_enum_gen import AIETileType

    # Shim tile at row 0
    assert device._infer_tile_type(0, 0) in (
        AIETileType.ShimNOCTile,
        AIETileType.ShimPLTile,
    )

    # Compute tile at row 2
    assert device._infer_tile_type(0, 2) == AIETileType.CoreTile
