# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import pytest
import numpy as np
import aie.iron as iron
import os

from util import _vector_vector_add_impl


@iron.jit()
def vector_vector_add(input0, input1, output):
    return _vector_vector_add_impl(input0, input1, output)


@pytest.mark.parametrize("num_elements", [16, 64])
@pytest.mark.parametrize("dtype", [np.int32])
def test_jit_return_type(num_elements, dtype):
    assert isinstance(vector_vector_add, iron.CallableDesign)


@iron.compileconfig()
def vector_vector_add_config(input0, input1, output):
    return _vector_vector_add_impl(input0, input1, output)


@pytest.mark.parametrize("num_elements", [16, 64])
@pytest.mark.parametrize("dtype", [np.int32])
def test_multiple_jit_compilations(num_elements, dtype):
    # Construct two input random tensors and an output zeroed tensor
    input0 = iron.randint(1, 100, (num_elements,), dtype=dtype, device="npu")
    input1 = iron.randint(1, 100, (num_elements,), dtype=dtype, device="npu")
    output = iron.zeros_like(input0)

    # JIT-compile the kernel then launch the kernel with the given arguments
    vector_vector_add(input0, input1, output)
    xclbin_path, insts_path = vector_vector_add.compilable.get_artifacts()
    assert os.path.exists(xclbin_path)
    assert os.path.exists(insts_path)
    assert np.array_equal(input0.numpy() + input1.numpy(), output.numpy())


@pytest.mark.parametrize("num_elements", [16, 64])
@pytest.mark.parametrize("dtype", [np.int32])
def test_jit_with_config(num_elements, dtype):
    # Construct two input random tensors and an output zeroed tensor
    input0 = iron.randint(1, 100, (num_elements,), dtype=dtype, device="npu")
    input1 = iron.randint(1, 100, (num_elements,), dtype=dtype, device="npu")
    output = iron.zeros_like(input0)

    # JIT-compile the kernel then launch the kernel with the given arguments
    jitted_vector_add = iron.jit(vector_vector_add_config)
    jitted_vector_add(input0, input1, output)
    xclbin_path, insts_path = jitted_vector_add.compilable.get_artifacts()
    assert os.path.exists(xclbin_path)
    assert os.path.exists(insts_path)
    assert np.array_equal(input0.numpy() + input1.numpy(), output.numpy())


@pytest.mark.parametrize("num_elements", [16, 64])
@pytest.mark.parametrize("dtype", [np.int32])
def test_jit_with_source_files(num_elements, dtype):
    # Construct two input random tensors and an output zeroed tensor
    input0 = iron.randint(1, 100, (num_elements,), dtype=dtype, device="npu")
    input1 = iron.randint(1, 100, (num_elements,), dtype=dtype, device="npu")
    output = iron.zeros_like(input0)

    @iron.jit(
        source_files=["test/python/test_file_1.cpp", "test/python/test_file_2.cpp"],
    )
    def jitted_vector_add(input0, input1, output):
        return _vector_vector_add_impl(input0, input1, output)

    jitted_vector_add(input0, input1, output)
    xclbin_path, insts_path = jitted_vector_add.compilable.get_artifacts()
    assert os.path.exists(xclbin_path)
    assert os.path.exists(insts_path)
    assert np.array_equal(input0.numpy() + input1.numpy(), output.numpy())
