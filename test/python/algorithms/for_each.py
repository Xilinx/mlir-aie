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


def test_for_each_add():
    """Test for_each algorithm with simple add_one operation."""
    data = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    original = data.numpy().copy()
    iron.jit(is_placed=False)(for_each)(data, lambda a: a + 1)
    assert np.allclose(original + 1, data.numpy())


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_for_each_different_datatypes(dtype):
    """Test for_each algorithm on different datatypes."""
    if np.issubdtype(dtype, np.floating):
        data = iron.rand(1024, dtype=dtype, device="npu")
    else:
        data = iron.randint(0, 100, (1024,), dtype=dtype, device="npu")
    original = data.numpy().copy()
    iron.jit(is_placed=False)(for_each)(data, lambda a: a + 1)
    assert np.allclose(original + 1, data.numpy())


@pytest.mark.parametrize(
    "dtype,c_type",
    [
        (np.int32, "int"),
        (np.float32, "float"),
    ],
)
def test_for_each_different_datatypes_extern(dtype, c_type):
    """Test for_each algorithm with ExternalFunction on different datatypes."""
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
    iron.jit(is_placed=False)(for_each)(data, add_one)
    assert np.allclose(original + 1, data.numpy())


@pytest.mark.parametrize("num_elements", [512, 1024, 2048])
def test_for_each_different_num_elements(num_elements):
    """Test for_each algorithm with different input sizes."""
    data = iron.randint(0, 100, (num_elements,), dtype=np.int32, device="npu")
    original = data.numpy().copy()
    iron.jit(is_placed=False)(for_each)(data, lambda a: a + 1)
    assert np.allclose(original + 1, data.numpy())


def test_for_each_extern_missing_tile_size():
    """Test error handling for kernels without tile_size"""
    bad_func = ExternalFunction(
        "bad_func",
        source_string="""extern "C" {
            void bad_func(int* input, int* output, float scale) {
                // This function doesn't have tile_size as last argument
            }
        }""",
        arg_types=[
            np.ndarray[(TILE_SIZE,), np.dtype[np.int32]],
            np.ndarray[(TILE_SIZE,), np.dtype[np.int32]],
            np.float32,
        ],
    )
    data = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")

    with pytest.raises(ValueError, match="last argument must be np.int32"):
        iron.jit(is_placed=False)(for_each)(data, bad_func)


def test_for_each_extern_wrong_args_count():
    """Test error handling for mismatched extra_func_args"""
    scale_func = ExternalFunction(
        "scale_func",
        source_string="""extern "C" {
            void scale_func(int* input, int* output, int s, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] * s;
                }
            }
        }""",
        arg_types=[
            np.ndarray[(TILE_SIZE,), np.dtype[np.int32]],
            np.ndarray[(TILE_SIZE,), np.dtype[np.int32]],
            np.int32,
            np.int32,
        ],
    )
    data = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")

    with pytest.raises(ValueError, match="expects 1 extra argument"):
        iron.jit(is_placed=False)(for_each)(data, scale_func)
