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
import shutil
import pytest

import aie.iron as iron
from aie.iron import ExternalKernel, jit
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_


@jit(is_placed=False)
def transform_with_internal_func_with_options(input, output):
    """Transform kernel that creates ExternalKernel internally with compiler options."""
    if input.shape != output.shape:
        raise ValueError(
            f"Input shapes are not the equal ({input.shape} != {output.shape})."
        )
    num_elements = np.size(input)

    # Create ExternalKernel inside the transform with compiler options
    internal_func = ExternalKernel(
        "internal_add_value",
        source_string="""extern "C" {
            void internal_add_value(int* input, int* output, int tile_size) {
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
        compile_flags=["-DADD_VALUE=1"],
    )

    # Extract tile size from ExternalKernel
    tile_size = internal_func.tile_size(0)

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
        # Extract tile size from ExternalKernel
        tile_size = func_to_apply.tile_size(0)

        # Number of sub-vector "tile" iterations
        for i in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            func_to_apply(elem_in, elem_out, tile_size)
            of_in.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), internal_func])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, B):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    # Place program components and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


@jit(is_placed=False)
def transform_with_internal_func_from_file(input, output):
    """Transform kernel that creates ExternalKernel internally from a file."""
    if input.shape != output.shape:
        raise ValueError(
            f"Input shapes are not the equal ({input.shape} != {output.shape})."
        )
    num_elements = np.size(input)

    # Create a temporary file with the source code inside the function
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cc", delete=False) as f:
        f.write(
            """extern "C" {
            void internal_add_from_file(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 42;
                }
            }
        }"""
        )
        temp_file_path = f.name

    # Create ExternalKernel inside the transform from a file
    internal_func = ExternalKernel(
        "internal_add_from_file",
        source_file=temp_file_path,
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    # Extract tile size from ExternalKernel
    tile_size = internal_func.tile_size(0)

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
        # Extract tile size from ExternalKernel
        tile_size = func_to_apply.tile_size(0)

        # Number of sub-vector "tile" iterations
        for i in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            func_to_apply(elem_in, elem_out, tile_size)
            of_in.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), internal_func])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, B):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    # Place program components and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


@jit(is_placed=False)
def transform_with_internal_func(input, output):
    """Transform kernel that creates ExternalKernel internally."""
    if input.shape != output.shape:
        raise ValueError(
            f"Input shapes are not the equal ({input.shape} != {output.shape})."
        )
    num_elements = np.size(input)

    # Create ExternalKernel inside the transform
    internal_func = ExternalKernel(
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

    # Extract tile size from ExternalKernel
    tile_size = internal_func.tile_size(0)

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
        # Extract tile size from ExternalKernel
        tile_size = func_to_apply.tile_size(0)

        # Number of sub-vector "tile" iterations
        for i in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            func_to_apply(elem_in, elem_out, tile_size)
            of_in.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), internal_func])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, B):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    # Place program components and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def test_transform_with_internal_func_with_options_inside():
    """Test transform function that creates ExternalKernel internally with compiler options."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Apply the transform (ExternalKernel is created inside with hardcoded compiler options)
    transform_with_internal_func_with_options(input_tensor, output_tensor)

    # Verify results
    expected = initial_tensor + 1
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)


def test_transform_with_internal_func_inside():
    """Test transform function that creates ExternalKernel internally."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Apply the transform (ExternalKernel is created inside)
    transform_with_internal_func(input_tensor, output_tensor)

    # Verify results
    expected = initial_tensor + 1
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)


def test_transform_with_internal_func_from_file():
    """Test transform function that creates ExternalKernel from a file."""
    # Create input and output tensors
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    initial_tensor = input_tensor.numpy().copy()

    # Apply the transform (ExternalKernel is created inside from file)
    transform_with_internal_func_from_file(input_tensor, output_tensor)

    # Verify results
    expected = initial_tensor + 42
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)
