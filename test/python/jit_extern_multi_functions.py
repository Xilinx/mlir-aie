# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import numpy as np
import os
import tempfile
import pytest

import aie.iron as iron
from aie.iron import ExternalFunction, jit
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_


def _validate_tensor_compatibility(input_tensor, output_tensor):
    """Validate that input and output tensors are compatible for processing."""
    if input_tensor.shape != output_tensor.shape:
        raise ValueError(
            f"Input and output shapes must match ({input_tensor.shape} != {output_tensor.shape})."
        )

    if input_tensor.dtype != output_tensor.dtype:
        raise ValueError(
            f"Input and output data types must match ({input_tensor.dtype} != {output_tensor.dtype})."
        )


@jit(is_placed=False)
def apply_pipeline_transform(
    input_tensor, output_tensor, first_function, second_function
):
    """Transform kernel that applies two functions in sequence to input tensor and stores result in output tensor."""
    _validate_tensor_compatibility(input_tensor, output_tensor)
    num_elements = np.size(input_tensor)

    # Extract tile size from ExternalFunction
    tile_size = first_function.tile_size(0)

    if num_elements % tile_size != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {tile_size}."
        )
    num_tiles = num_elements // tile_size

    dtype = input_tensor.dtype

    # Define tensor types
    tensor_type = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_type = np.ndarray[(tile_size,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    input_fifo = ObjectFifo(tile_type, name="input")
    intermediate_fifo = ObjectFifo(tile_type, name="intermediate")
    output_fifo = ObjectFifo(tile_type, name="output")

    # Define a task that will run on a compute tile for first function
    def process_first_stage(input_fifo, intermediate_fifo, function_to_apply):
        for i in range_(num_tiles):
            input_element = input_fifo.acquire(1)
            intermediate_element = intermediate_fifo.acquire(1)
            function_to_apply(input_element, intermediate_element, tile_size)
            input_fifo.release(1)
            intermediate_fifo.release(1)

    def process_second_stage(intermediate_fifo, output_fifo, function_to_apply):
        for i in range_(num_tiles):
            intermediate_element = intermediate_fifo.acquire(1)
            output_element = output_fifo.acquire(1)
            function_to_apply(intermediate_element, output_element, tile_size)
            intermediate_fifo.release(1)
            output_fifo.release(1)

    # Create workers to run the tasks on compute tiles
    first_stage_worker = Worker(
        process_first_stage,
        fn_args=[input_fifo.cons(), intermediate_fifo.prod(), first_function],
    )
    second_stage_worker = Worker(
        process_second_stage,
        fn_args=[intermediate_fifo.cons(), output_fifo.prod(), second_function],
    )

    # Runtime operations to move data to/from the AIE-array
    runtime = Runtime()
    with runtime.sequence(tensor_type, tensor_type) as (
        input_sequence,
        output_sequence,
    ):
        runtime.start(first_stage_worker, second_stage_worker)
        runtime.fill(input_fifo.prod(), input_sequence)
        runtime.drain(output_fifo.cons(), output_sequence, wait=True)

    # Place program components and generate an MLIR module
    return Program(iron.get_current_device(), runtime).resolve_program(
        SequentialPlacer()
    )


def test_add_pipeline():
    """Test basic ExternalFunction with two add operations applied in sequence."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    source_code = """extern "C" {
            void add_constant_1(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 1;
                }
            }
            void add_constant_2(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 2;
                }
            }
        }"""
    # Create ExternalFunction for adding one (first stage)
    tile_size = 16
    add_one_function = ExternalFunction(
        "add_constant_1",
        source_string=source_code,
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[np.int32]],
            np.ndarray[(tile_size,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    # Create ExternalFunction for adding two (second stage)
    add_two_function = ExternalFunction(
        "add_constant_2",
        source_string=source_code,
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[np.int32]],
            np.ndarray[(tile_size,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    # Apply the transform: input -> add_constant_1 -> add_constant_2 -> output
    # This will apply: input + 1 + 2 = input + 3
    apply_pipeline_transform(
        input_tensor, output_tensor, add_one_function, add_two_function
    )

    # Verify results: input + 1 + 2 = input + 3
    expected = initial_tensor + 3
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)


def test_add_pipeline_with_source_file():
    """Test ExternalFunction with source_file instead of source_string."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Create temporary source file with both functions
    with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
        f.write(
            """extern "C" {
            void add_constant_1(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 1;
                }
            }
            void add_constant_2(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 2;
                }
            }
        }"""
        )
        source_file_path = f.name

    try:
        # Test with source_file - both functions are in the same file
        tile_size = 16
        add_one_function = ExternalFunction(
            "add_constant_1",
            source_file=source_file_path,
            arg_types=[
                np.ndarray[(tile_size,), np.dtype[np.int32]],
                np.ndarray[(tile_size,), np.dtype[np.int32]],
                np.int32,
            ],
        )

        add_two_function = ExternalFunction(
            "add_constant_2",
            source_file=source_file_path,
            arg_types=[
                np.ndarray[(tile_size,), np.dtype[np.int32]],
                np.ndarray[(tile_size,), np.dtype[np.int32]],
                np.int32,
            ],
        )

        # Apply the transform
        apply_pipeline_transform(
            input_tensor, output_tensor, add_one_function, add_two_function
        )

        # Verify results: input + 1 + 2 = input + 3
        expected = initial_tensor + 3
        actual = output_tensor.numpy()
        np.testing.assert_array_equal(actual, expected)

    finally:
        # Clean up temporary file
        os.unlink(source_file_path)


@jit(is_placed=False)
def apply_pipeline_transform_with_internal_functions(input_tensor, output_tensor):
    """Transform kernel that defines two functions internally and applies them in sequence."""
    _validate_tensor_compatibility(input_tensor, output_tensor)
    num_elements = np.size(input_tensor)
    tile_size = 16
    num_tiles = num_elements // tile_size

    if num_elements % tile_size != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {tile_size}."
        )

    dtype = input_tensor.dtype

    # Define tensor types
    tensor_type = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_type = np.ndarray[(tile_size,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    input_fifo = ObjectFifo(tile_type, name="input")
    intermediate_fifo = ObjectFifo(tile_type, name="intermediate")
    output_fifo = ObjectFifo(tile_type, name="output")

    # Define kernels internally within the transform function
    source_code = """extern "C" {
            void add_constant_1(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 1;
                }
            }
            void add_constant_2(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 2;
                }
            }
        }"""

    # Create ExternalFunction for adding one (first stage)
    add_one_function = ExternalFunction(
        "add_constant_1",
        source_string=source_code,
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[np.int32]],
            np.ndarray[(tile_size,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    # Create ExternalFunction for adding two (second stage)
    add_two_function = ExternalFunction(
        "add_constant_2",
        source_string=source_code,
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[np.int32]],
            np.ndarray[(tile_size,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    # Define a task that will run on a compute tile for first function
    def process_first_stage(input_fifo, intermediate_fifo, function_to_apply):
        for i in range_(num_tiles):
            input_element = input_fifo.acquire(1)
            intermediate_element = intermediate_fifo.acquire(1)
            function_to_apply(input_element, intermediate_element, tile_size)
            input_fifo.release(1)
            intermediate_fifo.release(1)

    def process_second_stage(intermediate_fifo, output_fifo, function_to_apply):
        for i in range_(num_tiles):
            intermediate_element = intermediate_fifo.acquire(1)
            output_element = output_fifo.acquire(1)
            function_to_apply(intermediate_element, output_element, tile_size)
            intermediate_fifo.release(1)
            output_fifo.release(1)

    # Create workers to run the tasks on compute tiles
    first_stage_worker = Worker(
        process_first_stage,
        fn_args=[input_fifo.cons(), intermediate_fifo.prod(), add_one_function],
    )
    second_stage_worker = Worker(
        process_second_stage,
        fn_args=[intermediate_fifo.cons(), output_fifo.prod(), add_two_function],
    )

    # Runtime operations to move data to/from the AIE-array
    runtime = Runtime()
    with runtime.sequence(tensor_type, tensor_type) as (
        input_sequence,
        output_sequence,
    ):
        runtime.start(first_stage_worker, second_stage_worker)
        runtime.fill(input_fifo.prod(), input_sequence)
        runtime.drain(output_fifo.cons(), output_sequence, wait=True)

    # Place program components and generate an MLIR module
    return Program(iron.get_current_device(), runtime).resolve_program(
        SequentialPlacer()
    )


def test_add_pipeline_with_internal_functions():
    """Test ExternalFunction with two add operations defined internally in the transform function."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Apply the transform: input -> add_constant_1 -> add_constant_2 -> output
    # This will apply: input + 1 + 2 = input + 3
    apply_pipeline_transform_with_internal_functions(input_tensor, output_tensor)

    # Verify results: input + 1 + 2 = input + 3
    expected = initial_tensor + 3
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)


@jit(is_placed=False)
def apply_pipeline_transform_with_internal_file_functions(input_tensor, output_tensor):
    """Transform kernel that defines two functions internally using a source file and applies them in sequence."""
    _validate_tensor_compatibility(input_tensor, output_tensor)
    num_elements = np.size(input_tensor)
    tile_size = 16
    num_tiles = num_elements // tile_size

    if num_elements % tile_size != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {tile_size}."
        )

    dtype = input_tensor.dtype

    # Define tensor types
    tensor_type = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_type = np.ndarray[(tile_size,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    input_fifo = ObjectFifo(tile_type, name="input")
    intermediate_fifo = ObjectFifo(tile_type, name="intermediate")
    output_fifo = ObjectFifo(tile_type, name="output")

    # Create source file with both functions
    with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
        f.write(
            """extern "C" {
            void add_constant_1(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 1;
                }
            }
            void add_constant_2(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 2;
                }
            }
        }"""
        )
        source_file_path = f.name

    try:
        # Create ExternalFunction for adding one (first stage) using source file
        add_one_function = ExternalFunction(
            "add_constant_1",
            source_file=source_file_path,
            arg_types=[
                np.ndarray[(tile_size,), np.dtype[np.int32]],
                np.ndarray[(tile_size,), np.dtype[np.int32]],
                np.int32,
            ],
        )

        # Create ExternalFunction for adding two (second stage) using source file
        add_two_function = ExternalFunction(
            "add_constant_2",
            source_file=source_file_path,
            arg_types=[
                np.ndarray[(tile_size,), np.dtype[np.int32]],
                np.ndarray[(tile_size,), np.dtype[np.int32]],
                np.int32,
            ],
        )

        # Define a task that will run on a compute tile for first function
        def process_first_stage(input_fifo, intermediate_fifo, function_to_apply):
            for i in range_(num_tiles):
                input_element = input_fifo.acquire(1)
                intermediate_element = intermediate_fifo.acquire(1)
                function_to_apply(input_element, intermediate_element, tile_size)
                input_fifo.release(1)
                intermediate_fifo.release(1)

        def process_second_stage(intermediate_fifo, output_fifo, function_to_apply):
            for i in range_(num_tiles):
                intermediate_element = intermediate_fifo.acquire(1)
                output_element = output_fifo.acquire(1)
                function_to_apply(intermediate_element, output_element, tile_size)
                intermediate_fifo.release(1)
                output_fifo.release(1)

        # Create workers to run the tasks on compute tiles
        first_stage_worker = Worker(
            process_first_stage,
            fn_args=[input_fifo.cons(), intermediate_fifo.prod(), add_one_function],
        )
        second_stage_worker = Worker(
            process_second_stage,
            fn_args=[intermediate_fifo.cons(), output_fifo.prod(), add_two_function],
        )

        # Runtime operations to move data to/from the AIE-array
        runtime = Runtime()
        with runtime.sequence(tensor_type, tensor_type) as (
            input_sequence,
            output_sequence,
        ):
            runtime.start(first_stage_worker, second_stage_worker)
            runtime.fill(input_fifo.prod(), input_sequence)
            runtime.drain(output_fifo.cons(), output_sequence, wait=True)

        # Place program components and generate an MLIR module
        return Program(iron.get_current_device(), runtime).resolve_program(
            SequentialPlacer()
        )
    finally:
        # Clean up temporary file
        os.unlink(source_file_path)


def test_add_pipeline_with_internal_file_functions():
    """Test ExternalFunction with two add operations defined internally using a source file."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Apply the transform: input -> add_constant_1 -> add_constant_2 -> output
    # This will apply: input + 1 + 2 = input + 3
    apply_pipeline_transform_with_internal_file_functions(input_tensor, output_tensor)

    # Verify results: input + 1 + 2 = input + 3
    expected = initial_tensor + 3
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)
