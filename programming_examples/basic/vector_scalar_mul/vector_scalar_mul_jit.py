# vector_scalar_mul/vector_scalar_mul_jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2025 Advanced Micro Devices, Inc. or its affiliates

import argparse
import sys
import numpy as np
import aie.iron as iron
import os

from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype
import argparse
import sys
import numpy as np
import aie.iron as iron

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_
from aie.iron import trace


@iron.jit(is_placed=False)
def vector_scalar_mul(input0, input1, output):
    if input0.shape != output.shape:
        raise ValueError(
            f"Input and output shapes are not the same ({input0.shape} != {output.shape})."
        )
    if len(np.shape(input0)) != 1:
        raise ValueError("Function only supports vectors.")

    num_elements = np.size(input0)

    # Add size validation like in reference code
    # Assert that input1 (factor) is size 4 bytes (1 integer)
    if np.size(input1) != 1:
        raise ValueError("2nd input buffer must be size 1 (1 integer).")

    # Assert output size matches input size
    if output.numel() != input0.numel():
        raise ValueError("Output buffer size must match input buffer size.")

    num_sub_vectors = 4
    tile_size = num_elements // num_sub_vectors

    if num_elements % num_sub_vectors != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {num_sub_vectors}."
        )

    if input0.dtype != output.dtype:
        raise ValueError(
            f"Input and output data types are not the same ({input0.dtype} != {output.dtype})."
        )
    dtype = input0.dtype

    # Define tensor types - factor should be scalar_ty (np.int32), not tile_ty
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

    # Create a handle to an externally-defined kernel
    # Construct path to kernel source file
    current_dir = os.path.dirname(__file__)
    kernel_path = os.path.join(current_dir, "../../../aie_kernels/aie2", "scale.cc")
    # Get the bit width directly from the dtype
    bit_width = np.dtype(input0.dtype).itemsize * 8

    # Use the same kernel function name as reference code
    scale = ExternalFunction(
        "vector_scalar_mul_vector",
        source_file=kernel_path,
        arg_types=[
            tile_ty,  # input tensor
            tile_ty,  # output tensor
            scalar_ty,  # scalar factor
            np.int32,  # N
        ],
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
        include_dirs=[os.path.join(current_dir, "../../../aie_kernels/aie2")],
    )

    # AIE-array data movement with object fifos
    # Factor should be scalar_ty, not tensor_ty
    of_in = ObjectFifo(tile_ty, name="in")
    of_factor = ObjectFifo(scalar_ty, name="infactor")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task that will run on a compute tile
    def core_body(of_in, of_factor, of_out, scale_fn):
        # Acquire factor once outside the loop, like in reference code
        elem_factor = of_factor.acquire(1)

        # Number of sub-vector "tile" iterations
        for _ in range_(num_sub_vectors):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            scale_fn(elem_in, elem_out, elem_factor, tile_size)
            of_in.release(1)
            of_out.release(1)
        # Release factor once after the loop
        of_factor.release(1)

    # Create a worker to run the task on a compute tile
    # enable_trace = 1 if trace.get_trace_size() > 0 else 0
    worker = Worker(
        core_body,
        fn_args=[of_in.cons(), of_factor.cons(), of_out.prod(), scale],
        trace=1 if trace.get_trace_size() > 0 else 0,
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()

    with rt.sequence(tensor_ty, scalar_ty, tensor_ty) as (A, F, C):
        if trace.get_trace_size() > 0:
            rt.enable_trace(trace.get_trace_size())
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.fill(of_factor.prod(), F)
        rt.drain(of_out.cons(), C, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-n",
        "--num-elements",
        type=int,
        default=1024,
        help="Number of elements (default: 1024, must be multiple of 128 and >= 1024)",
    )
    parser.add_argument(
        "-t",
        "--trace-size",
        type=int,
        default=1024,
        help="Trace buffer size (0 = no tracing, default: 0)",
    )
    parser.add_argument(
        "-z",
        "--data_type",
        choices=["i16", "i32"],
        default="i16",
        help="Data type (default: i16)",
    )
    args = parser.parse_args()

    # Buffer size validation like reference code
    if args.num_elements % 128 != 0 or args.num_elements < 1024:
        print(
            "Number of elements must be a multiple of 128 (so len is multiple of 64) and greater than or equal to 1024 (so len >= 512)"
        )
        raise ValueError

    # Construct input random tensors and an output zeroed tensor
    # The tensors are in memory accessible to the NPU
    datatype = str_to_dtype(args.data_type)
    input0 = iron.randint(0, 100, (args.num_elements,), dtype=datatype, device="npu")
    scalar = iron.randint(0, 100, (1,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input0)

    # Enable tracing if requested
    if args.trace_size > 0:
        trace.set_trace_size(args.trace_size)
        trace.start_trace()

    # JIT-compile the kernel then launches the kernel with the given arguments
    vector_scalar_mul(input0, scalar, output)

    # Stop tracing and save results if tracing was enabled
    if args.trace_size > 0:
        trace_filename = f"trace_output_{args.num_elements}_{args.data_type}.json"
        trace.stop_trace(trace_filename)
        print(f"Tracing completed and saved to {trace_filename}")

    # Check the correctness of the result - use scalar multiplication
    expected = input0.numpy() * scalar.numpy()[0]
    actual = output.numpy()
    e = np.equal(expected, actual)
    errors = np.size(e) - np.count_nonzero(e)

    # Optionally, print the results
    if args.verbose:
        print(f"{'input0':>4} * {'factor':>4} = {'output':>4}")
        print("-" * 34)
        count = input0.numel()
        factor = scalar.numpy()[0]
        for idx, (a, c) in enumerate(zip(input0[:count], output[:count])):
            print(f"{idx:2}: {a:4} * {factor:4} = {c:4}")

    # If the result is correct, exit with a success code.
    # Otherwise, exit with a failure code
    if not errors:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        sys.exit(-1)


if __name__ == "__main__":
    main()
