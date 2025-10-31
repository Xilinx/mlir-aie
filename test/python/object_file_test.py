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

from .utils import _vector_vector_add_impl


@pytest.mark.parametrize("num_elements", [16, 64])
@pytest.mark.parametrize("dtype", [np.int32])
def test_jit_with_object_files(num_elements, dtype):
    # Construct two input random tensors and an output zeroed tensor
    input0 = iron.randint(1, 100, (num_elements,), dtype=dtype, device="npu")
    input1 = iron.randint(1, 100, (num_elements,), dtype=dtype, device="npu")
    output = iron.zeros_like(input0)

    @iron.jit(
        object_files=["test/python/test_obj.o"],
    )
    def jitted_vector_add(input0, input1, output):
        return _vector_vector_add_impl(input0, input1, output)

    jitted_vector_add(input0, input1, output)
    assert np.array_equal(input0.numpy() + input1.numpy(), output.numpy())
