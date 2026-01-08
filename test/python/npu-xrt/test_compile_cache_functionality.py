# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import numpy as np
import tempfile
import os


import aie.iron as iron
from aie.iron import ExternalFunction
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_


@iron.jit(is_placed=False)
def transform(input, output, func):
    """Transform kernel that applies a function to input tensor and stores result in output tensor."""
    if input.shape != output.shape:
        raise ValueError(
            f"Input shapes are not the equal ({input.shape} != {output.shape})."
        )
    num_elements = np.size(input)

    if isinstance(func, iron.ExternalFunction):
        tile_size = func.tile_size(0)
    else:
        tile_size = 16 if num_elements >= 16 else 1

    if num_elements % tile_size != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {tile_size}."
        )
    num_tiles = num_elements // tile_size

    if input.dtype != output.dtype:
        raise ValueError(
            f"Input data types are not the same ({input.dtype} != {output.dtype})."
        )

    dtype = input.dtype

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task that will run on a compute tile
    def core_body(of_in, of_out, func_to_apply):
        for _ in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            if isinstance(func_to_apply, iron.ExternalFunction):
                func_to_apply(elem_in, elem_out, tile_size)
            else:
                for j in range_(tile_size):
                    elem_out[j] = func_to_apply(elem_in[j])
            of_in.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), func])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, B):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def test_cache_lambda_functions():
    """Test that caching works correctly with different lambda functions."""
    # Create input tensor
    input_tensor = iron.arange(32, dtype=np.int32)

    # Test 1: First execution with lambda function
    transform(input_tensor, input_tensor, lambda x: x + 1)
    result1 = input_tensor.numpy().copy()

    # Reset tensor
    input_tensor[:] = np.arange(32, dtype=np.int32)

    # Test 2: Second execution with same lambda function (should use cache)
    transform(input_tensor, input_tensor, lambda x: x + 1)
    result2 = input_tensor.numpy()

    # Results should be identical
    np.testing.assert_array_equal(result1, result2)

    # Test 3: Different lambda function (should generate new cache entry)
    input_tensor[:] = np.arange(1, 33, dtype=np.int32)
    transform(input_tensor, input_tensor, lambda x: x * 2)
    result3 = input_tensor.numpy()

    # Results should be different
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, result1, result3
    )


def test_cache_external_functions():
    """Test that ExternalFunction caching works correctly during execution."""
    # Create input tensor
    input_tensor = iron.arange(32, dtype=np.int32)

    # Test 1: First execution
    add_one_1 = ExternalFunction(
        "add_one",
        source_string="""extern "C" {
            void add_one(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 1;
                }
            }
        }""",
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
    )
    transform(input_tensor, input_tensor, add_one_1)
    result1 = input_tensor.numpy().copy()

    # Reset tensor
    input_tensor[:] = np.arange(32, dtype=np.int32)

    # Test 2: Second execution
    add_one_2 = ExternalFunction(
        "add_one",
        source_string="""extern "C" {
            void add_one(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 1;
                }
            }
        }""",
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
    )
    transform(input_tensor, input_tensor, add_one_2)
    result2 = input_tensor.numpy()

    # Results should be identical
    np.testing.assert_array_equal(result1, result2)

    # Test 3: Different ExternalFunction (should generate new cache entry)
    multiply_two = ExternalFunction(
        "multiply_two",
        source_string="""extern "C" {
            void multiply_two(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] * 2;
                }
            }
        }""",
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    input_tensor[:] = np.arange(32, dtype=np.int32)
    transform(input_tensor, input_tensor, multiply_two)
    result3 = input_tensor.numpy()

    # Results should be different
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, result1, result3
    )


def test_cache_compile_flags():
    """Test that ExternalFunctions with different compile flags produce different results."""
    # Create input tensor
    input_tensor = iron.arange(32, dtype=np.int32)

    # Create ExternalFunctions with different compile flags
    add_5 = ExternalFunction(
        "add_value",
        source_string="""extern "C" {
            void add_value(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + ADD_VALUE;
                }
            }
        }""",
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
        compile_flags=["-DADD_VALUE=5"],
    )

    add_10 = ExternalFunction(
        "add_value",
        source_string="""extern "C" {
            void add_value(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + ADD_VALUE;
                }
            }
        }""",
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
        compile_flags=["-DADD_VALUE=10"],
    )

    # Test with ADD_VALUE=5
    transform(input_tensor, input_tensor, add_5)
    result_5 = input_tensor.numpy().copy()

    # Reset and test with ADD_VALUE=10
    input_tensor[:] = np.arange(32, dtype=np.int32)
    transform(input_tensor, input_tensor, add_10)
    result_10 = input_tensor.numpy()

    # Results should be different
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, result_5, result_10
    )

    # Verify expected results
    expected_5 = np.arange(32, dtype=np.int32) + 5
    expected_10 = np.arange(32, dtype=np.int32) + 10

    np.testing.assert_array_equal(result_5, expected_5)
    np.testing.assert_array_equal(result_10, expected_10)


def test_cache_source_changes():
    """Test that ExternalFunctions with different source content produce different results."""
    # Create input tensor
    input_tensor = iron.tensor((32,), dtype=np.int32)
    input_tensor[:] = np.arange(1, 33, dtype=np.int32)  # [1, 2, 3, ..., 32]

    # Create ExternalFunctions with different source content
    add_1 = ExternalFunction(
        "add_one",
        source_string="""extern "C" {
            void add_one(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 1;
                }
            }
        }""",
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    add_2 = ExternalFunction(
        "add_one",
        source_string="""extern "C" {
            void add_one(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 2;  // Different operation
                }
            }
        }""",
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    # Test with add_1
    transform(input_tensor, input_tensor, add_1)
    result_1 = input_tensor.numpy().copy()

    # Reset and test with add_2
    input_tensor[:] = np.arange(1, 33, dtype=np.int32)
    transform(input_tensor, input_tensor, add_2)
    result_2 = input_tensor.numpy()

    # Results should be different
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, result_1, result_2
    )

    # Verify expected results
    expected_1 = np.arange(1, 33, dtype=np.int32) + 1
    expected_2 = np.arange(1, 33, dtype=np.int32) + 2

    np.testing.assert_array_equal(result_1, expected_1)
    np.testing.assert_array_equal(result_2, expected_2)


def test_cache_file_source():
    """Test that ExternalFunctions with file sources work correctly."""
    # Create input tensor
    input_tensor = iron.tensor((32,), dtype=np.int32)
    input_tensor[:] = np.arange(1, 33, dtype=np.int32)  # [1, 2, 3, ..., 32]

    # Create temporary source file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cc", delete=False) as f:
        source_content = """extern "C" {
            void add_one_from_file(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 1;
                }
            }
        }"""
        f.write(source_content)
        source_file_path = f.name

    try:
        # Create ExternalFunction using source_file
        add_one_from_file = ExternalFunction(
            "add_one_from_file",
            source_file=source_file_path,
            arg_types=[
                np.ndarray[(16,), np.dtype[np.int32]],
                np.ndarray[(16,), np.dtype[np.int32]],
                np.int32,
            ],
        )

        # Test execution
        transform(input_tensor, input_tensor, add_one_from_file)
        result = input_tensor.numpy()

        # Verify expected results
        expected = np.arange(1, 33, dtype=np.int32) + 1
        np.testing.assert_array_equal(result, expected)

    finally:
        # Clean up the temporary file
        os.unlink(source_file_path)


def test_cache_include_directories():
    """Test that ExternalFunctions with include directories work correctly."""
    # Create input tensor
    input_tensor = iron.arange(32, dtype=np.int32)

    # Create temporary directory with header file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create header file
        header_file = os.path.join(temp_dir, "math_ops.h")
        with open(header_file, "w") as f:
            f.write("#define ADD_VALUE 42\n")

        # Create ExternalFunction that includes the header
        add_value = ExternalFunction(
            "add_value",
            source_string="""extern "C" {
                #include "math_ops.h"
                void add_value(int* input, int* output, int tile_size) {
                    for (int i = 0; i < tile_size; i++) {
                        output[i] = input[i] + ADD_VALUE;
                    }
                }
            }""",
            arg_types=[
                np.ndarray[(16,), np.dtype[np.int32]],
                np.ndarray[(16,), np.dtype[np.int32]],
                np.int32,
            ],
            include_dirs=[temp_dir],
        )

        # Test execution
        transform(input_tensor, input_tensor, add_value)
        result = input_tensor.numpy()

        # Verify expected results
        expected = np.arange(32, dtype=np.int32) + 42
        np.testing.assert_array_equal(result, expected)


def test_cache_tensor_shapes():
    """Test that different tensor shapes work correctly with caching."""
    # Test with different tensor sizes
    sizes = [16, 32, 64]
    results = []

    for size in sizes:
        input_tensor = iron.arange(size, dtype=np.int32)

        # Apply transformation
        transform(input_tensor, input_tensor, lambda x: x + 1)
        result = input_tensor.numpy()
        results.append(result)

        # Verify expected results
        expected = np.arange(size, dtype=np.int32) + 1
        np.testing.assert_array_equal(result, expected)


def test_cache_tensor_dtypes():
    """Test that different tensor dtypes work correctly with caching."""
    # Test with different dtypes
    dtypes = [np.int32, np.float32]
    results = []

    for dtype in dtypes:
        input_tensor = iron.arange(32, dtype=dtype)

        # Apply transformation
        transform(input_tensor, input_tensor, lambda x: x + 1)
        result = input_tensor.numpy()
        results.append(result)

        # Verify expected results
        expected = np.arange(32, dtype=dtype) + 1
        np.testing.assert_array_equal(result, expected)
