# vector_vector_add/vector_vector_add_placed.py -*- Python -*-
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

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_


@iron.jit
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

    buffer_depth = 2

    @device(iron.get_current_device().resolve())
    def device_body():
        tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
        tile_ty = np.ndarray[(n,), np.dtype[dtype]]

        # AIE Core Function declarations

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        of_in1 = object_fifo("in1", ShimTile, ComputeTile2, buffer_depth, tile_ty)
        of_in2 = object_fifo("in2", ShimTile, ComputeTile2, buffer_depth, tile_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, tile_ty)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2)
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                # Number of sub-vector "tile" iterations
                for _ in range_(N_div_n):
                    elem_in1 = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    elem_in2 = of_in2.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(n):
                        elem_out[i] = elem_in1[i] + elem_in2[i]
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    of_in2.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
        def sequence(A, B, C):
            in1_task = shim_dma_single_bd_task(of_in1, A, sizes=[1, 1, 1, num_elements])
            in2_task = shim_dma_single_bd_task(of_in2, B, sizes=[1, 1, 1, num_elements])
            out_task = shim_dma_single_bd_task(
                of_out, C, sizes=[1, 1, 1, num_elements], issue_token=True
            )

            dma_start_task(in1_task, in2_task, out_task)
            dma_await_task(out_task)
            dma_free_task(in1_task, in2_task)


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
