# test_transform_channel_budget.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Transform validates worker counts; the compiler owns DMA resource checks."""

import numpy as np
import pytest

from aie.iron.algorithms._transform import (
    transform_parallel,
    transform_parallel_binary,
)


@pytest.mark.parametrize("transform", [transform_parallel, transform_parallel_binary])
@pytest.mark.parametrize("bad", [0, -1])
def test_worker_count_must_be_positive(transform, bad):
    tensor_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    with pytest.raises(ValueError, match="num_channels must be positive"):
        transform(lambda *args: args[0], tensor_ty, num_channels=bad)


@pytest.mark.parametrize("num_channels", [1, 2, 3])
def test_dma_budget_is_left_to_compiler(npu2_device, num_channels):
    tensor_ty = np.ndarray[(96,), np.dtype[np.int32]]
    module = transform_parallel_binary(
        lambda a, b: a + b, tensor_ty, tile_size=16, num_channels=num_channels
    )
    assert module.operation.verify()
