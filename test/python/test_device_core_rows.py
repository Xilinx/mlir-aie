# test_device_core_rows.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Device.core_rows: the rows of compute tiles, from the target model."""

import pytest

from aie.dialects._aie_enum_gen import AIETileType
from aie.iron.device import NPU1, NPU1Col1, NPU2, NPU2Col1


@pytest.mark.parametrize("device", [NPU1, NPU1Col1, NPU2, NPU2Col1])
def test_core_rows_are_the_compute_tiles_bottom_to_top(device):
    dev = device()
    rows = dev.core_rows
    assert rows == sorted(rows) and rows
    for r in range(dev.rows):
        is_core = dev.get_tile_type(0, r) is AIETileType.CoreTile
        assert (r in rows) == is_core


def test_npus_have_four_rows_of_cores_above_the_shim_and_memtile():
    assert NPU1().core_rows == [2, 3, 4, 5]
    assert NPU2().core_rows == [2, 3, 4, 5]
