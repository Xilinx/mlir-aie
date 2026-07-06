# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
from aie.iron.device import NPU1Col1, NPU1Col2, NPU1, NPU2


@pytest.fixture(params=[NPU1Col1, NPU1Col2, NPU1, NPU2])
def device(request):
    return request.param()


def test_rows_cols(device):
    assert device.rows == device._tm.rows()
    assert device.cols == device._tm.columns()


def test_tile_type_inferred_from_coordinates(device):
    """get_tile_type must return the correct tile type for known coordinates."""
    from aie.dialects._aie_enum_gen import AIETileType

    # Shim tile at row 0
    assert device.get_tile_type(0, 0) in (
        AIETileType.ShimNOCTile,
        AIETileType.ShimPLTile,
    )

    # Compute tile at row 2
    assert device.get_tile_type(0, 2) == AIETileType.CoreTile


def test_out_of_range_coordinates_error(device):
    """get_tile_type must reject out-of-range coordinates."""
    with pytest.raises(ValueError, match="out of range"):
        device.get_tile_type(99, 99)
    with pytest.raises(ValueError, match="out of range"):
        device.get_tile_type(-1, 0)
    with pytest.raises(ValueError, match="out of range"):
        device.get_tile_type(0, -1)
