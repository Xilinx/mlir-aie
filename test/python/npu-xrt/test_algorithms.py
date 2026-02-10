# test_algorithms.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
import numpy as np

import aie.iron as iron
from aie.iron import ExternalFunction
from aie.iron.algorithms import (
    transform,
    transform_parallel,
    transform_binary,
    transform_parallel_binary,
    for_each,
)

TILE_SIZE = 16


# =============================================================================
# transform tests
# =============================================================================


def test_transform_add():
    """Test transform algorithm with simple add_one operation"""
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    original = input.numpy().copy()
    iron.jit(is_placed=False)(transform)(lambda a: a + 1, input, output)

    assert np.allclose(original + 1, output.numpy())


@pytest.mark.parametrize(
    "dtype,c_type",
    [
        (np.int32, "int"),
        (np.float32, "float"),
    ],
)
def test_transform_different_datatypes_extern(dtype, c_type):
    """Test transform algorithm with ExternalFunction on different datatypes."""
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
        input = iron.rand(1024, dtype=dtype, device="npu")
    else:
        input = iron.randint(0, 100, (1024,), dtype=dtype, device="npu")
    output = iron.zeros_like(input)
    iron.jit(is_placed=False)(transform)(add_one, input, output, TILE_SIZE)
    assert np.allclose(input.numpy() + 1, output.numpy())


@pytest.mark.parametrize("num_elements", [512, 1024, 2048])
def test_transform_different_num_elements(num_elements):
    """Test transform algorithm with different input size."""
    input = iron.randint(0, 100, (num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    iron.jit(is_placed=False)(transform)(lambda a: a + 1, input, output)
    assert np.allclose(input.numpy() + 1, output.numpy())


def test_transform_shape_mismatch():
    """Test transform when input and output shapes don't match."""
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros((512,), dtype=np.int32, device="npu")
    with pytest.raises(ValueError, match="shape.*doesn't match"):
        iron.jit(is_placed=False)(transform)(lambda a: a + 1, input, output)


def test_transform_dtype_mismatch():
    """Test transform when input and output dtypes don't match."""
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros((1024,), dtype=np.float32, device="npu")
    with pytest.raises(ValueError, match="dtype.*doesn't match"):
        iron.jit(is_placed=False)(transform)(lambda a: a + 1, input, output)


def test_transform_tile_size_mismatch():
    """Test transform when number of elements is not a multiple of tile size."""
    # 1000 is not divisible by default tile size of 16
    input = iron.randint(0, 100, (1000,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    with pytest.raises(ValueError, match="must be a multiple of tile size"):
        iron.jit(is_placed=False)(transform)(lambda a: a + 1, input, output)


# =============================================================================
# transform_binary tests
# =============================================================================


def test_transform_binary_add():
    """Test transform_binary algorithm with simple add_one operation."""
    first = iron.randint(0, 50, (1024,), dtype=np.int32, device="npu")
    second = iron.randint(0, 50, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(first)
    iron.jit(is_placed=False)(transform_binary)(
        lambda a, b: a + b, first, second, output
    )
    assert np.allclose(first.numpy() + second.numpy(), output.numpy())


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_transform_binary_different_datatypes(dtype):
    """Test transform_binary algorithm with different datatypes."""
    if np.issubdtype(dtype, np.floating):
        first = iron.rand(1024, dtype=dtype, device="npu")
        second = iron.rand(1024, dtype=dtype, device="npu")
    else:
        first = iron.randint(0, 50, (1024,), dtype=dtype, device="npu")
        second = iron.randint(0, 50, (1024,), dtype=dtype, device="npu")
    output = iron.zeros_like(first)
    iron.jit(is_placed=False)(transform_binary)(
        lambda a, b: a + b, first, second, output
    )
    assert np.allclose(first.numpy() + second.numpy(), output.numpy())


@pytest.mark.parametrize("num_elements", [512, 1024, 2048])
def test_transform_binary_different_num_elements(num_elements):
    """Test transform_binary algorithm with different input size."""
    first = iron.randint(0, 50, (num_elements,), dtype=np.int32, device="npu")
    second = iron.randint(0, 50, (num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros_like(first)
    iron.jit(is_placed=False)(transform_binary)(
        lambda a, b: a + b, first, second, output
    )
    assert np.allclose(first.numpy() + second.numpy(), output.numpy())


# =============================================================================
# transform_parallel tests
# =============================================================================


def test_transform_parallel_add():
    """Test transform_parallel algorithm with simple add_one operation"""
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    iron.jit(is_placed=False)(transform_parallel)(lambda a: a + 1, input, output)
    assert np.allclose(input.numpy() + 1, output.numpy())


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_transform_parallel_different_datatypes(dtype):
    """Test transform_parallel algorithm with add operation on different datatypes."""
    if np.issubdtype(dtype, np.floating):
        input = iron.rand(1024, dtype=dtype, device="npu")
    else:
        input = iron.randint(0, 50, (1024,), dtype=dtype, device="npu")
    output = iron.zeros_like(input)
    iron.jit(is_placed=False)(transform_parallel)(lambda a: a + 1, input, output)
    assert np.allclose(input.numpy() + 1, output.numpy())


@pytest.mark.parametrize("num_elements", [512, 1024, 2048])
def test_transform_parallel_different_num_elements(num_elements):
    """Test transform_parallel algorithm with different input size."""
    input = iron.randint(0, 100, (num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    iron.jit(is_placed=False)(transform_parallel)(lambda a: a + 1, input, output)
    assert np.allclose(input.numpy() + 1, output.numpy())


def test_transform_parallel_shape_mismatch():
    """Test transform parallel when input and output shapes don't match."""
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros((512,), dtype=np.int32, device="npu")
    with pytest.raises(ValueError, match="shape.*doesn't match"):
        iron.jit(is_placed=False)(transform_parallel)(lambda a: a + 1, input, output)


def test_transform_parallel_dtype_mismatch():
    """Test transform parallel when input and output dtypes don't match."""
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros((1024,), dtype=np.float32, device="npu")
    with pytest.raises(ValueError, match="dtype.*doesn't match"):
        iron.jit(is_placed=False)(transform_parallel)(lambda a: a + 1, input, output)


def test_transform_parallel_tile_size_mismatch():
    """Test transform parallel when number of elements is not a multiple of tile size."""
    # 1000 is not divisible by default tile size of 16
    input = iron.randint(0, 100, (1000,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    with pytest.raises(ValueError, match="must be a multiple of tile size"):
        iron.jit(is_placed=False)(transform_parallel)(lambda a: a + 1, input, output)


# =============================================================================
# transform_parallel_binary tests
# =============================================================================


def test_transform_parallel_binary_add():
    """Test transform_parallel_binary algorithm with add operation."""
    first = iron.randint(0, 50, (1024,), dtype=np.int32, device="npu")
    second = iron.randint(0, 50, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(first)
    iron.jit(is_placed=False)(transform_parallel_binary)(
        lambda a, b: a + b, first, second, output
    )
    assert np.allclose(first.numpy() + second.numpy(), output.numpy())


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_transform_parallel_binary_different_datatypes(dtype):
    """Test transform_parallel_binary algorithm with add operation on different datatypes."""
    if np.issubdtype(dtype, np.floating):
        first = iron.rand(1024, dtype=dtype, device="npu")
        second = iron.rand(1024, dtype=dtype, device="npu")
    else:
        first = iron.randint(0, 50, (1024,), dtype=dtype, device="npu")
        second = iron.randint(0, 50, (1024,), dtype=dtype, device="npu")
    output = iron.zeros_like(first)
    iron.jit(is_placed=False)(transform_parallel_binary)(
        lambda a, b: a + b, first, second, output
    )
    assert np.allclose(first.numpy() + second.numpy(), output.numpy())


@pytest.mark.parametrize("num_elements", [512, 1024, 2048])
def test_transform_parallel_binary_different_num_elements(num_elements):
    """Test transform_parallel_binary algorithm with different input size."""
    first = iron.randint(0, 50, (num_elements,), dtype=np.int32, device="npu")
    second = iron.randint(0, 50, (num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros_like(first)
    iron.jit(is_placed=False)(transform_parallel_binary)(
        lambda a, b: a + b, first, second, output
    )
    assert np.allclose(first.numpy() + second.numpy(), output.numpy())


# =============================================================================
# for_each tests
# =============================================================================


def test_for_each_add():
    """Test for_each algorithm with simple add_one operation."""
    data = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    original = data.numpy().copy()
    iron.jit(is_placed=False)(for_each)(lambda a: a + 1, data)
    assert np.allclose(original + 1, data.numpy())


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_for_each_different_datatypes(dtype):
    """Test for_each algorithm on different datatypes."""
    if np.issubdtype(dtype, np.floating):
        data = iron.rand(1024, dtype=dtype, device="npu")
    else:
        data = iron.randint(0, 100, (1024,), dtype=dtype, device="npu")
    original = data.numpy().copy()
    iron.jit(is_placed=False)(for_each)(lambda a: a + 1, data)
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
    iron.jit(is_placed=False)(for_each)(add_one, data, TILE_SIZE)
    assert np.allclose(original + 1, data.numpy())


@pytest.mark.parametrize("num_elements", [512, 1024, 2048])
def test_for_each_different_num_elements(num_elements):
    """Test for_each algorithm with different input sizes."""
    data = iron.randint(0, 100, (num_elements,), dtype=np.int32, device="npu")
    original = data.numpy().copy()
    iron.jit(is_placed=False)(for_each)(lambda a: a + 1, data)
    assert np.allclose(original + 1, data.numpy())
