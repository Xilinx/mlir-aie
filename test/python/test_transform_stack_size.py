# test_transform_stack_size.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Library designs leave the core stack to aiecc unless the caller fixes it."""

import re

import numpy as np
import pytest
from ml_dtypes import bfloat16

from aie.iron import kernels
from aie.iron.algorithms import kernel_design as kd
from aie.iron.algorithms._transform import transform, transform_parallel
from aie.iron.device import NPU1Col1
from aie.utils import get_current_device
from aie.utils.hostruntime import set_current_device


@pytest.fixture
def npu1_device():
    previous = get_current_device(probe_runtime=False)
    set_current_device(NPU1Col1())
    try:
        yield
    finally:
        set_current_device(previous)


@pytest.mark.parametrize("num_channels", [1, 2])
def test_transform_parallel_leaves_the_stack_to_aiecc(npu1_device, num_channels):
    module = transform_parallel(
        kernels.gelu(tile_size=1024),
        np.ndarray[(4096,), np.dtype[bfloat16]],
        tile_size=1024,
        num_channels=num_channels,
        pass_size_to_kernel=False,
    )
    assert "stack_size" not in str(module)


def test_transform_leaves_the_stack_to_aiecc(npu1_device):
    module = transform(
        kernels.gelu_sized(tile_size=1024),
        np.ndarray[(4096,), np.dtype[bfloat16]],
        tile_size=1024,
    )
    assert "stack_size" not in str(module)


def test_kernel_design_leaves_the_stack_to_aiecc(npu1_device):
    mlir = str(kd.design(kernels.gelu, tile_size=1024).as_mlir())
    assert "stack_size" not in mlir


def test_kernel_design_keeps_an_explicit_stack(npu1_device):
    mlir = str(kd.design(kernels.gelu, tile_size=1024, stack_bytes=4096).as_mlir())
    assert re.findall(r"stack_size = (\d+) : i32", mlir) == ["4096"]
