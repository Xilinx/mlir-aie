# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import pytest
import aie.iron as iron
import numpy as np
import os

from .utils import _vector_vector_add_impl


@pytest.mark.parametrize("num_elements", [16, 64])
@pytest.mark.parametrize("dtype", [np.int32])
def test_precompiled(num_elements, dtype):
    # Construct two input random tensors and an output zeroed tensor
    input0 = iron.randint(1, 100, (num_elements,), dtype=dtype, device="npu")
    input1 = iron.randint(1, 100, (num_elements,), dtype=dtype, device="npu")
    output = iron.zeros_like(input0)

    jitted_vector_add = iron.jit(_vector_vector_add_impl)
    jitted_vector_add(input0, input1, output)
    xclbin_path, insts_path = jitted_vector_add.compilable.get_artifacts()

    precompiled = iron.PreCompiled(xclbin_path, insts_path)

    @iron.jit
    def jitted_vector_add_precompiled(input0, input1, output):
        return precompiled(input0, input1, output)

    jitted_vector_add_precompiled(input0, input1, output)

    assert np.array_equal(input0.numpy() + input1.numpy(), output.numpy())
