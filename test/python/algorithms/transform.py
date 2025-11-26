# transform.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import aie.iron as iron
from aie.iron.algorithms import transform
import pytest
import numpy as np


@pytest.mark.parametrize(
    "dtype,num_elements",
    [(np.float32, 1024)],
)
def test_transform_add(dtype, num_elements):
    if np.issubdtype(dtype, np.floating):
        input = iron.rand(num_elements, dtype=dtype, device="npu")
    else:
        input = iron.randint(0, 100, (num_elements,), dtype=dtype, device="npu")
    output = iron.zeros_like(input)
    transform(input, output, lambda a: a + 1)
    assert np.allclose(input.numpy() + 1, output.numpy())


# TODO(erika): Need to regenerate the kernel AND looks like there is some state in the placement algorithm that breaks if I regenerate the kernel
# @pytest.mark.parametrize(
#    "dtype,num_elements",
#    [(np.float32, 1024)],
# )
# def test_transform_subtract(dtype, num_elements):
#    if np.issubdtype(dtype, np.floating):
#        input = iron.rand(num_elements, dtype=dtype, device="npu")
#    else:
#        input = iron.randint(0, 100, (num_elements,), dtype=dtype, device="npu")
#    output = iron.zeros_like(input)
#    transform(input, output, lambda a: a - 1)
#    assert np.allclose(input.numpy() - 1, output.numpy())
