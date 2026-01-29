# for_each.py -*- Python -*-
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
from aie.iron import ExternalFunction
from aie.iron.algorithms import for_each
import pytest
import numpy as np

TILE_SIZE = 16

# =============================================================================
# for_each tests
# =============================================================================


def test_for_each_add():
    """Test for_each algorithm with add_one lambda (in-place modification)."""
    data = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    original = data.numpy().copy()
    for_each(data, lambda a: a + 1)
    assert np.allclose(original + 1, data.numpy())


def test_for_each_sub():
    """Test for_each algorithm with subtract_one lambda (in-place modification)."""
    data = iron.randint(1, 100, (1024,), dtype=np.int32, device="npu")
    original = data.numpy().copy()
    for_each(data, lambda a: a - 1)
    assert np.allclose(original - 1, data.numpy(), atol=1e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_for_each_different_datatypes(dtype):
    """Test for_each algorithm with add_one operation on different datatypes."""
    if np.issubdtype(dtype, np.floating):
        data = iron.rand(1024, dtype=dtype, device="npu")
    else:
        data = iron.randint(0, 100, (1024,), dtype=dtype, device="npu")
    original = data.numpy().copy()
    for_each(data, lambda a: a + 1)
    assert np.allclose(original + 1, data.numpy())


@pytest.mark.parametrize(
    "dtype,c_type",
    [
        (np.int32, "int"),
        (np.float32, "float"),
    ],
)
def test_for_each_different_datatypes_extern(dtype, c_type):
    """Test for_each algorithm with an add_one ExternalFunction on different datatypes."""
    add_one = ExternalFunction(
        "add_one",
        source_string=f"""extern "C" {{
            void add_one({c_type}* input, {c_type}* output, int tile_size) {{
                for (int i = 0; i < tile_size; i++) {{
                    output[i] = input[i] + 1.0f;
                }}
            }}
        }}""",
        arg_types=[
            np.ndarray[(TILE_SIZE,), np.dtype[dtype]],
            np.ndarray[(TILE_SIZE,), np.dtype[dtype]],
            np.int32,
        ],
    )
    if np.issubdtype(dtype, np.floating):
        data = iron.rand(1024, dtype=dtype, device="npu")
    else:
        data = iron.randint(0, 100, (1024,), dtype=dtype, device="npu")
    original = data.numpy().copy()
    for_each(data, add_one)
    assert np.allclose(original + 1, data.numpy())


@pytest.mark.parametrize("num_elements", [512, 1024, 2048])
def test_for_each_different_num_elements(num_elements):
    """Test for_each algorithm with different input sizes."""
    data = iron.randint(0, 100, (num_elements,), dtype=np.int32, device="npu")
    original = data.numpy().copy()
    for_each(data, lambda a: a + 1)
    assert np.allclose(original + 1, data.numpy())
