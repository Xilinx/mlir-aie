# test_transform_stack_size.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Transform Workers take the core stack a library kernel's contract declares."""

import re

import numpy as np
import pytest
from ml_dtypes import bfloat16

from aie.iron import kernels
from aie.iron.algorithms._transform import transform_parallel
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
def test_contract_stack_reaches_every_core(npu1_device, num_channels):
    gelu = kernels.gelu(tile_size=1024)
    assert gelu.contract.stack_bytes
    module = transform_parallel(
        gelu,
        np.ndarray[(4096,), np.dtype[bfloat16]],
        tile_size=1024,
        num_channels=num_channels,
        pass_size_to_kernel=False,
    )
    sizes = re.findall(r"stack_size = (\d+)", str(module))
    assert sizes == [str(gelu.contract.stack_bytes)] * num_channels


def test_a_kernel_without_a_contract_keeps_the_default(npu1_device):
    module = transform_parallel(
        lambda x: x, np.ndarray[(64,), np.dtype[np.int32]], tile_size=16
    )
    assert "stack_size" not in str(module)
