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
from aie.iron import ExternalFunction
from aie.iron.algorithms import (
    transform,
    transform_parallel,
    transform_binary,
    transform_parallel_binary,
)
import pytest
import numpy as np

TILE_SIZE = 16

# =============================================================================
# transform tests
# =============================================================================


def test_transform_add():
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    transform(input, output, lambda a: a + 1)
    assert np.allclose(input.numpy() + 1, output.numpy())


def test_transform_sub():
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    transform(input, output, lambda a: a - 1)
    assert np.allclose(input.numpy() - 1, output.numpy(), atol=1e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_transform_different_datatypes(dtype):
    """Test transform algorithm with add_one operation on different datatypes."""
    if np.issubdtype(dtype, np.floating):
        input = iron.rand(1024, dtype=dtype, device="npu")
    else:
        input = iron.randint(0, 100, (1024,), dtype=dtype, device="npu")
    output = iron.zeros_like(input)
    transform(input, output, lambda a: a + 1)
    assert np.allclose(input.numpy() + 1, output.numpy())


@pytest.mark.parametrize(
    "dtype,c_type",
    [
        (np.int32, "int"),
        (np.float32, "float"),
    ],
)
def test_transform_different_datatypes_extern(dtype, c_type):
    """Test transform algorithm with an add_one ExternalFunction on different datatypes."""
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
    transform(input, output, add_one)
    assert np.allclose(input.numpy() + 1, output.numpy())


@pytest.mark.parametrize("num_elements", [512, 1024, 2048])
def test_transform_different_num_elements(num_elements):
    """Test transform algorithm with different input size."""
    input = iron.randint(0, 100, (num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    transform(input, output, lambda a: a + 1)
    assert np.allclose(input.numpy() + 1, output.numpy())


# =============================================================================
# transform_parallel tests
# =============================================================================


def test_transform_parallel_add():
    """Test transform_parallel algorithm with add_one lambda."""
    input = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    transform_parallel(input, output, lambda a: a + 1)
    assert np.allclose(input.numpy() + 1, output.numpy())


def test_transform_parallel_sub():
    """Test transform_parallel algorithm with subtract_one lambda."""
    input = iron.randint(1, 100, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input)
    transform_parallel(input, output, lambda a: a - 1)
    assert np.allclose(input.numpy() - 1, output.numpy(), atol=1e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_transform_parallel_different_datatypes(dtype):
    """Test transform_parallel algorithm with add_one operation on different datatypes."""
    if np.issubdtype(dtype, np.floating):
        input = iron.rand(1024, dtype=dtype, device="npu")
    else:
        input = iron.randint(0, 100, (1024,), dtype=dtype, device="npu")
    output = iron.zeros_like(input)
    transform_parallel(input, output, lambda a: a + 1)
    assert np.allclose(input.numpy() + 1, output.numpy())


@pytest.mark.parametrize(
    "dtype,c_type",
    [
        (np.int32, "int"),
        (np.float32, "float"),
    ],
)
def test_transform_parallel_different_datatypes_extern(dtype, c_type):
    """Test transform_parallel algorithm with an add_one ExternalFunction on different datatypes."""
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
    transform_parallel(input, output, add_one)
    assert np.allclose(input.numpy() + 1, output.numpy())


# =============================================================================
# transform_binary tests
# =============================================================================


def test_transform_binary_add():
    """Test transform_binary algorithm with add operation."""
    first = iron.randint(0, 50, (1024,), dtype=np.int32, device="npu")
    second = iron.randint(0, 50, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(first)
    transform_binary(first, second, output, lambda a, b: a + b)
    assert np.allclose(first.numpy() + second.numpy(), output.numpy())


def test_transform_binary_sub():
    """Test transform_binary algorithm with subtract operation."""
    first = iron.randint(50, 100, (1024,), dtype=np.int32, device="npu")
    second = iron.randint(0, 50, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(first)
    transform_binary(first, second, output, lambda a, b: a - b)
    assert np.allclose(first.numpy() - second.numpy(), output.numpy())


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_transform_binary_different_datatypes(dtype):
    """Test transform_binary algorithm with add operation on different datatypes."""
    if np.issubdtype(dtype, np.floating):
        first = iron.rand(1024, dtype=dtype, device="npu")
        second = iron.rand(1024, dtype=dtype, device="npu")
    else:
        first = iron.randint(0, 50, (1024,), dtype=dtype, device="npu")
        second = iron.randint(0, 50, (1024,), dtype=dtype, device="npu")
    output = iron.zeros_like(first)
    transform_binary(first, second, output, lambda a, b: a + b)
    assert np.allclose(first.numpy() + second.numpy(), output.numpy())


# =============================================================================
# transform_parallel_binary tests
# =============================================================================


def test_transform_parallel_binary_add():
    """Test transform_parallel_binary algorithm with add operation."""
    first = iron.randint(0, 50, (1024,), dtype=np.int32, device="npu")
    second = iron.randint(0, 50, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(first)
    transform_parallel_binary(first, second, output, lambda a, b: a + b)
    assert np.allclose(first.numpy() + second.numpy(), output.numpy())


def test_transform_parallel_binary_sub():
    """Test transform_parallel_binary algorithm with subtract operation."""
    first = iron.randint(50, 100, (1024,), dtype=np.int32, device="npu")
    second = iron.randint(0, 50, (1024,), dtype=np.int32, device="npu")
    output = iron.zeros_like(first)
    transform_parallel_binary(first, second, output, lambda a, b: a - b)
    assert np.allclose(first.numpy() - second.numpy(), output.numpy())


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
    transform_parallel_binary(first, second, output, lambda a, b: a + b)
    assert np.allclose(first.numpy() + second.numpy(), output.numpy())
