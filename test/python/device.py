# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from aie.iron.device import NPU1Col1, NPU1Col2, NPU1, NPU2
from aie.iron.device import Tile


@pytest.fixture(params=[NPU1Col1, NPU1Col2, NPU1, NPU2])
def device(request):
    return request.param()


def test_legal_mem_affinity(device):
    # Test single tile
    assert device.is_legal_mem_affinity(Tile(1, 2))
    # Test adjacent compute tiles
    assert device.is_legal_mem_affinity(Tile(1, 2), Tile(1, 3))
    # Test adjacent memory tiles
    assert device.is_legal_mem_affinity(Tile(1, 1), Tile(2, 1))
    # Test non-adjacent tiles
    assert not device.is_legal_mem_affinity(Tile(1, 2), Tile(3, 4))
    # Test same tile
    assert device.is_legal_mem_affinity(Tile(1, 2), Tile(1, 2))
    # Test diagonal compute tiles
    assert not device.is_legal_mem_affinity(Tile(1, 2), Tile(2, 3))
    # Test adjacent shim and mem tiles
    assert not device.is_legal_mem_affinity(Tile(0, 0), Tile(0, 1))
    # Test adjacent mem and compute tiles
    assert not device.is_legal_mem_affinity(Tile(0, 1), Tile(0, 2))
    # Test multiple adjacent compute tiles
    assert device.is_legal_mem_affinity(Tile(1, 2), Tile(1, 3), Tile(1, 4))
    # Test multiple non-adjacent compute tiles
    assert not device.is_legal_mem_affinity(Tile(1, 2), Tile(1, 3), Tile(1, 5))
    # Test multiple tiles of different types
    assert not device.is_legal_mem_affinity(Tile(0, 0), Tile(1, 2))


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


def test_rows_cols(device):
    assert device.rows == device._tm.rows()
    assert device.cols == device._tm.columns()
