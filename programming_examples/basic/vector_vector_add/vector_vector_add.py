# vector_vector_add/vector_vector_add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates

import argparse
import sys
import numpy as np
import aie.iron as iron

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_


@iron.jit(is_placed=False)
def vector_vector_add(input0, input1, output):
    if input0.shape != input1.shape:
        raise ValueError(
            f"Input shapes are not the equal ({input0.shape} != {input1.shape})."
        )
    if input0.shape != output.shape:
        raise ValueError(
            f"Input and output shapes are not the equal ({input0.shape} != {output.shape})."
        )
    if len(np.shape(input0)) != 1:
        raise ValueError("Function only supports vectors.")
    num_elements = np.size(input0)
    n = 16
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    N_div_n = num_elements // n

    if input0.dtype != input1.dtype:
        raise ValueError(
            f"Input data types are not the same ({input0.dtype} != {input1.dtype})."
        )
    if input0.dtype != output.dtype:
        raise ValueError(
            f"Input and output data types are not the same ({input0.dtype} != {output.dtype})."
        )
    dtype = input0.dtype

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(n,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in1 = ObjectFifo(tile_ty, name="in1")
    of_in2 = ObjectFifo(tile_ty, name="in2")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_in2, of_out):
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(n):
                elem_out[i] = elem_in1[i] + elem_in2[i]
            of_in1.release(1)
            of_in2.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in1.cons(), of_in2.cons(), of_out.prod()])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(of_in1.prod(), A)
        rt.fill(of_in2.prod(), B)
        rt.drain(of_out.cons(), C, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-n",
        "--num-elements",
        type=int,
        default=32,
        help="Number of elements (default: 32)",
    )
    args = parser.parse_args()

    # Construct two input random tensors and an output zeroed tensor
    # The three tensor are in memory accessible to the NPU
    input0 = iron.randint(0, 100, (args.num_elements,), dtype=np.int32, device="npu")
    input1 = iron.randint(0, 100, (args.num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input0)

    # JIT-compile the kernel then launches the kernel with the given arguments. Future calls
    # to the kernel will use the same compiled kernel and loaded code objects
    vector_vector_add(input0, input1, output)

    # Check the correctness of the result
    e = np.equal(input0.numpy() + input1.numpy(), output.numpy())
    errors = np.size(e) - np.count_nonzero(e)

    # Optionally, print the results
    if args.verbose:
        print(f"{'input0':>4} + {'input1':>4} = {'output':>4}")
        print("-" * 34)
        count = input0.numel()
        for idx, (a, b, c) in enumerate(
            zip(input0[:count], input1[:count], output[:count])
        ):
            print(f"{idx:2}: {a:4} + {b:4} = {c:4}")

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
