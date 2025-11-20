# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from aie.iron.device.device import DeviceView
from aie.iron.device import NPU2, NPU1, NPU1Col1

# RUN: %python -m pytest %s

devices = [NPU1, NPU1Col1, NPU2]


@pytest.mark.parametrize("device_class", devices)
def test_initial_state(device_class):
    device = device_class()
    total_tiles = device.cols * device.rows
    assert len(list(device.tile_iterator())) == total_tiles


@pytest.mark.parametrize("device_class", devices)
def test_destructive_slice(device_class):
    device = device_class()
    if device.cols < 3 or device.rows < 3:
        pytest.skip("Device too small for this test")

    initial_tile_count = len(list(device.tile_iterator()))

    partition = device[1:3, 1:3]
    assert isinstance(partition, DeviceView)

    partition_tiles = list(partition.tile_iterator())
    assert len(partition_tiles) == 4
    assert partition_tiles[0].col == 1
    assert partition_tiles[0].row == 1

    remaining_tiles_count = len(list(device.tile_iterator()))
    assert remaining_tiles_count == initial_tile_count - 4


@pytest.mark.parametrize("device_class", devices)
def test_double_claim_error_overlap(device_class):
    device = device_class()
    if device.cols < 4 or device.rows < 4:
        pytest.skip("Device too small for this test")

    _ = device[1:3, 1:3]
    with pytest.raises(ValueError, match="Tile .* has already been claimed."):
        _ = device[2:4, 2:4]  # Overlapping slice


@pytest.mark.parametrize("device_class", devices)
def test_double_claim_error_single_col(device_class):
    device = device_class()
    if device.rows < 4:
        pytest.skip("Device too small for this test")
    _ = device[0:1, 0:3]
    with pytest.raises(ValueError, match="Tile .* has already been claimed."):
        _ = device[0:1, 2:4]  # Overlapping slice


@pytest.mark.parametrize("device_class", devices)
def test_non_destructive_single_tile_access(device_class):
    device = device_class()
    if device.cols < 2 or device.rows < 3:
        pytest.skip("Device too small for this test")

    initial_tile_count = len(list(device.tile_iterator()))
    tile = device[1, 2]
    assert tile.col == 1
    assert tile.row == 2

    # Check that the tile was not claimed
    remaining_tiles_count = len(list(device.tile_iterator()))
    assert remaining_tiles_count == initial_tile_count


@pytest.mark.parametrize("device_class", devices)
def test_disjoint_partitions(device_class):
    device = device_class()
    if device.cols < 4 or device.rows < 4:
        pytest.skip("Device too small for this test")

    initial_tile_count = len(list(device.tile_iterator()))

    part1 = device[0:2, 0:2]
    part2 = device[2:4, 2:4]

    assert len(list(part1.tile_iterator())) == 4
    assert len(list(part2.tile_iterator())) == 4

    remaining_tiles_count = len(list(device.tile_iterator()))
    assert remaining_tiles_count == initial_tile_count - 8


@pytest.mark.parametrize("device_class", devices)
def test_empty_slice(device_class):
    device = device_class()
    partition = device[0:0, :]
    assert isinstance(partition, DeviceView)
    assert len(list(partition.tile_iterator())) == 0
