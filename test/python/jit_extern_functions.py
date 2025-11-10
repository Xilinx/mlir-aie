# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import numpy as np
import os
import tempfile
import pytest

import aie.iron as iron
from aie.iron import ExternalFunction, jit
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_


@jit(is_placed=False)
def transform(input, output, func):
    """Transform kernel that applies a function to input tensor and stores result in output tensor."""
    if input.shape != output.shape:
        raise ValueError(
            f"Input shapes are not the equal ({input.shape} != {output.shape})."
        )
    num_elements = np.size(input)

    # Extract tile size from ExternalFunction (using first argument)
    tile_size = func.tile_size(0)

    # Assert that input and output arrays have the same tile size
    assert func.tile_size(0) == func.tile_size(
        1
    ), f"Input and output tile sizes must match: {func.tile_size(0)} != {func.tile_size(1)}"

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
        # Extract tile size from ExternalFunction (using first argument)
        tile_size = func_to_apply.tile_size(0)

        # Number of sub-vector "tile" iterations
        for i in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            func_to_apply(elem_in, elem_out, tile_size)
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


def test_simple_add_one():
    """Test basic ExternalFunction with simple add_one operation."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Create ExternalFunction for adding one
    add_one = ExternalFunction(
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

    # Apply the transform
    transform(input_tensor, output_tensor, add_one)

    # Verify results
    expected = initial_tensor + 1
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("tile_size", [8, 16, 32, 64])
def test_different_tile_sizes(tile_size):
    """Test ExternalFunction with different tile sizes."""
    # Create input and output tensors
    num_elements = 1024
    input_tensor = iron.randint(0, 100, (num_elements,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((num_elements,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Create ExternalFunction with specific tile size
    add_one = ExternalFunction(
        "add_one",
        source_string="""extern "C" {
            void add_one(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 1;
                }
            }
        }""",
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[np.int32]],
            np.ndarray[(tile_size,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    # Apply the transform
    transform(input_tensor, output_tensor, add_one)

    # Verify results
    expected = initial_tensor + 1
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "dtype,c_type",
    [
        (np.int32, "int"),
        (np.float32, "float"),
    ],
)
def test_different_data_types(dtype, c_type):
    """Test ExternalFunction with different data types."""
    # Create input and output tensors
    input_tensor = iron.rand((1024,), dtype=dtype, device="npu")
    output_tensor = iron.zeros((1024,), dtype=dtype, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Create ExternalFunction with specific data type
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
            np.ndarray[(16,), np.dtype[dtype]],
            np.ndarray[(16,), np.dtype[dtype]],
            np.int32,
        ],
    )

    # Apply the transform
    transform(input_tensor, output_tensor, add_one)

    # Verify results
    expected = initial_tensor + 1.0
    actual = output_tensor.numpy()
    np.testing.assert_array_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("value", [5, 42])
def test_define_values(value):
    """Test ExternalFunction with different define values."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    add_value = ExternalFunction(
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
        compile_flags=[f"-DADD_VALUE={value}"],
    )

    # Apply the transform
    transform(input_tensor, output_tensor, add_value)

    # Verify results
    expected = initial_tensor + value
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)


def test_multiple_defines():
    """Test ExternalFunction with multiple defines."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Create ExternalFunction with multiple defines
    complex_op = ExternalFunction(
        "complex_op",
        source_string="""extern "C" {
            void complex_op(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    #ifdef FLAG2
                    output[i] = input[i] + ADD_VALUE + FLAG2_OFFSET;
                    #else
                    output[i] = input[i] + ADD_VALUE;
                    #endif
                }
            }
        }""",
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
        compile_flags=["-DADD_VALUE=5", "-DFLAG2", "-DFLAG2_OFFSET=10"],
    )

    # Apply the transform
    transform(input_tensor, output_tensor, complex_op)

    # Verify results (should add 15: 5 + 10 due to FLAG2 define)
    expected = initial_tensor + 15
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)


def test_include_directories():
    """Test ExternalFunction with include directories."""
    # Create a temporary directory with a header file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a header file
        header_file = os.path.join(temp_dir, "math_ops.h")
        with open(header_file, "w") as f:
            f.write(
                """
#ifndef MATH_OPS_H
#define MATH_OPS_H

#define ADD_VALUE 42

#endif
"""
            )

        # Create input and output tensors
        input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
        output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
        initial_tensor = input_tensor.numpy().copy()

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

        # Apply the transform
        transform(input_tensor, output_tensor, add_value)

        # Verify results
        expected = initial_tensor + 42
        actual = output_tensor.numpy()
        np.testing.assert_array_equal(actual, expected)


def test_multiple_include_directories():
    """Test ExternalFunction with multiple include directories."""
    # Create temporary directories with header files
    with tempfile.TemporaryDirectory() as temp_dir1, tempfile.TemporaryDirectory() as temp_dir2:
        # Create header files
        header1 = os.path.join(temp_dir1, "ops1.h")
        with open(header1, "w") as f:
            f.write("#define VALUE1 10\n")

        header2 = os.path.join(temp_dir2, "ops2.h")
        with open(header2, "w") as f:
            f.write("#define VALUE2 20\n")

        # Create input and output tensors
        input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
        output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
        initial_tensor = input_tensor.numpy().copy()

        # Create ExternalFunction that includes both headers
        add_values = ExternalFunction(
            "add_values",
            source_string="""extern "C" {
                #include "ops1.h"
                #include "ops2.h"
                void add_values(int* input, int* output, int tile_size) {
                    for (int i = 0; i < tile_size; i++) {
                        output[i] = input[i] + VALUE1 + VALUE2;
                    }
                }
            }""",
            arg_types=[
                np.ndarray[(16,), np.dtype[np.int32]],
                np.ndarray[(16,), np.dtype[np.int32]],
                np.int32,
            ],
            include_dirs=[temp_dir1, temp_dir2],
        )

        # Apply the transform
        transform(input_tensor, output_tensor, add_values)

        # Verify results
        expected = initial_tensor + 30  # 10 + 20
        actual = output_tensor.numpy()
        np.testing.assert_array_equal(actual, expected)


def test_caching_same_source():
    """Test that same source code produces same cached result."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Create two ExternalFunctions with identical source
    add_one_1 = ExternalFunction(
        "add_one_1",
        source_string="""extern "C" {
            void add_one_1(int* input, int* output, int tile_size) {
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

    add_one_2 = ExternalFunction(
        "add_one_2",
        source_string="""extern "C" {
            void add_one_2(int* input, int* output, int tile_size) {
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

    # Apply both transforms
    transform(input_tensor, output_tensor, add_one_1)
    result1 = output_tensor.numpy().copy()

    output_tensor.fill(0)
    transform(input_tensor, output_tensor, add_one_2)
    result2 = output_tensor.numpy()

    # Verify both produce same results
    np.testing.assert_array_equal(result1, result2)


def test_context_manager():
    """Test ExternalFunction with context manager syntax."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Create ExternalFunction and use it with context manager
    with ExternalFunction(
        "add_one_context",
        source_string="""extern "C" {
            void add_one_context(int* input, int* output, int tile_size) {
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
    ) as add_one:
        # Apply the transform
        transform(input_tensor, output_tensor, add_one)

    # Verify results
    expected = initial_tensor + 1
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)


def test_context_manager_with_compiler_options():
    """Test ExternalFunction with context manager and compiler options."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Create ExternalFunction with compiler options using context manager
    with ExternalFunction(
        "add_value_context",
        source_string="""extern "C" {
            void add_value_context(int* input, int* output, int tile_size) {
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
        compile_flags=["-DADD_VALUE=42"],
    ) as add_value:
        # Apply the transform
        transform(input_tensor, output_tensor, add_value)

    # Verify results
    expected = initial_tensor + 42
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)


def test_source_file():
    """Test ExternalFunction with source_file instead of source_string."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Create a temporary source file
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

        # Apply the transform
        transform(input_tensor, output_tensor, add_one_from_file)

        # Verify results
        expected = initial_tensor + 1
        actual = output_tensor.numpy()
        np.testing.assert_array_equal(actual, expected)

    finally:
        # Clean up the temporary file
        os.unlink(source_file_path)


def test_source_file_with_compiler_options():
    """Test ExternalFunction with source_file and compiler options."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Create a temporary source file with defines
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cc", delete=False) as f:
        source_content = """extern "C" {
            void add_value_from_file(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + ADD_VALUE;
                }
            }
        }"""
        f.write(source_content)
        source_file_path = f.name

    try:
        # Create ExternalFunction using source_file with compiler options
        add_value_from_file = ExternalFunction(
            "add_value_from_file",
            source_file=source_file_path,
            arg_types=[
                np.ndarray[(16,), np.dtype[np.int32]],
                np.ndarray[(16,), np.dtype[np.int32]],
                np.int32,
            ],
            compile_flags=["-DADD_VALUE=25"],
        )

        # Apply the transform
        transform(input_tensor, output_tensor, add_value_from_file)

        # Verify results
        expected = initial_tensor + 25
        actual = output_tensor.numpy()
        np.testing.assert_array_equal(actual, expected)

    finally:
        # Clean up the temporary file
        os.unlink(source_file_path)


def test_transform_with_internal_func():
    """Test transform function that creates ExternalFunction internally."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Create ExternalFunction dynamically but pass it as argument
    internal_func = ExternalFunction(
        "internal_add_one",
        source_string="""extern "C" {
            void internal_add_one(int* input, int* output, int tile_size) {
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

    # Apply the transform (ExternalFunction is passed as argument)
    transform(input_tensor, output_tensor, internal_func)

    # Verify results
    expected = initial_tensor + 1
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)


def test_caching_different_flags():
    """Test that different compile flags produce different cached results."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Create ExternalFunctions with same source but different flags
    add_value_5 = ExternalFunction(
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

    add_value_10 = ExternalFunction(
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

    # Apply transforms
    transform(input_tensor, output_tensor, add_value_5)
    result_5 = output_tensor.numpy().copy()

    output_tensor.fill(0)
    transform(input_tensor, output_tensor, add_value_10)
    result_10 = output_tensor.numpy()

    # Verify different results
    expected_5 = initial_tensor + 5
    expected_10 = initial_tensor + 10

    np.testing.assert_array_equal(result_5, expected_5)
    np.testing.assert_array_equal(result_10, expected_10)
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, result_5, result_10
    )


@pytest.mark.parametrize(
    "invalid_source",
    [
        # Missing semicolon
        """extern "C" {
        void invalid_func(int* input, int* output, int tile_size) {
            for (int i = 0; i < tile_size; i++) {
                output[i] = input[i] + 1  // Missing semicolon
            }
        }
    }""",
        # Undefined variable
        """extern "C" {
        void invalid_func(int* input, int* output, int tile_size) {
            for (int i = 0; i < tile_size; i++) {
                output[i] = input[i] + undefined_var;
            }
        }
    }""",
        # Syntax error
        """extern "C" {
        void invalid_func(int* input, int* output, int tile_size) {
            for (int i = 0; i < tile_size; i++) {
                output[i] = input[i] + 1;
            }  // Missing closing brace
    }""",
    ],
)
def test_invalid_source(invalid_source):
    """Test error handling for invalid C++ source."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")

    # Create ExternalFunction with invalid C++ source
    invalid_func = ExternalFunction(
        "invalid_func",
        source_string=invalid_source,
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    # Should raise an error during compilation
    with pytest.raises(Exception):
        transform(input_tensor, output_tensor, invalid_func)


@pytest.mark.parametrize(
    "input_tile_size,output_tile_size",
    [
        (16, 32),  # Different tile sizes
        (8, 16),  # Different tile sizes
        (64, 32),  # Different tile sizes
    ],
)
def test_mismatched_tile_sizes(input_tile_size, output_tile_size):
    """Test error handling for mismatched tile sizes."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")

    # Create ExternalFunction with mismatched tile sizes
    mismatched_func = ExternalFunction(
        "mismatched_func",
        source_string="""extern "C" {
            void mismatched_func(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 1;
                }
            }
        }""",
        arg_types=[
            np.ndarray[(input_tile_size,), np.dtype[np.int32]],
            np.ndarray[(output_tile_size,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    # Should raise an assertion error
    with pytest.raises(AssertionError, match="Input and output tile sizes must match"):
        transform(input_tensor, output_tensor, mismatched_func)


@pytest.mark.parametrize(
    "invalid_include",
    [
        "/nonexistent/directory",
        "/tmp/nonexistent_header_dir",
        "/usr/local/include/nonexistent",
    ],
)
def test_invalid_include_directory(invalid_include):
    """Test error handling for invalid include directory."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")

    # Create ExternalFunction with invalid include directory
    invalid_include_func = ExternalFunction(
        "invalid_include_func",
        source_string="""extern "C" {
            #include "nonexistent.h"
            void invalid_include_func(int* input, int* output, int tile_size) {
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
        include_dirs=[invalid_include],
    )

    # Should raise an error during compilation
    with pytest.raises(Exception):
        transform(input_tensor, output_tensor, invalid_include_func)


@pytest.mark.parametrize(
    "compile_flags,expected_value",
    [
        (["-DADD_VALUE=5"], 5),
        (["-DADD_VALUE=10", "-DMULTIPLIER=2"], 20),  # 10 * 2
        (["-DADD_VALUE=3", "-DOFFSET=7"], 10),  # 3 + 7
        (["-DADD_VALUE=1", "-DFLAG2", "-DFLAG2_OFFSET=9"], 10),  # 1 + 9 (FLAG2 enabled)
    ],
)
def test_compiler_flag_combinations(compile_flags, expected_value):
    """Test ExternalFunction with different combinations of compiler flags."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Create source that uses the defines
    source_template = """extern "C" {
        void complex_op(int* input, int* output, int tile_size) {
            for (int i = 0; i < tile_size; i++) {
                #ifdef MULTIPLIER
                output[i] = input[i] + ADD_VALUE * MULTIPLIER;
                #elif defined(FLAG2)
                output[i] = input[i] + ADD_VALUE + FLAG2_OFFSET;
                #elif defined(OFFSET)
                output[i] = input[i] + ADD_VALUE + OFFSET;
                #else
                output[i] = input[i] + ADD_VALUE;
                #endif
            }
        }
    }"""

    complex_op = ExternalFunction(
        "complex_op",
        source_string=source_template,
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
        compile_flags=compile_flags,
    )

    # Apply the transform
    transform(input_tensor, output_tensor, complex_op)

    # Verify results
    expected = initial_tensor + expected_value
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)


def test_object_file_name():
    """Test ExternalFunction with explicit object file name."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Create ExternalFunction for adding one
    add_one = ExternalFunction(
        "add_one",
        object_file_name="my_add_one.o",
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

    # Apply the transform
    transform(input_tensor, output_tensor, add_one)

    # Verify results
    expected = initial_tensor + 1
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)
