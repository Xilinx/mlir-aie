# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import os
import numpy as np

import aie.iron as iron
from aie.iron import ExternalFunction, ObjectFifo, Worker, Runtime, Program
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, NPU1Col1, NPU2, NPU2Col1

from aie.iron.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
from aie.iron.compile import compile_mlir_module, compile_external_kernel


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


def test_simple_add_one(tmp_path):
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

    # Get the MLIR module
    mlir_module = transform(input_tensor, output_tensor, add_one)

    # Compile the module
    xclbin_path = tmp_path / "final.xclbin"
    insts_path = tmp_path / "insts.bin"

    current_device = iron.get_current_device()
    if isinstance(current_device, (NPU2, NPU2Col1)):
        target_arch = "aie2p"
    elif isinstance(current_device, (NPU1, NPU1Col1)):
        target_arch = "aie2"
    else:
        raise RuntimeError(f"Unsupported device type: {type(current_device)}")

    compile_external_kernel(add_one, str(tmp_path), target_arch)
    compile_mlir_module(
        mlir_module,
        str(insts_path),
        str(xclbin_path),
        work_dir=str(tmp_path),
        verbose=True,
    )

    # Run the kernel
    runtime = XRTHostRuntime()
    handle = runtime.load(str(xclbin_path))
    runtime.run(handle, input_tensor, output_tensor)
    runtime.cleanup()

    # Verify results
    expected = initial_tensor + 1
    actual = output_tensor.numpy()
    np.testing.assert_array_equal(actual, expected)
