# vector_vector_add/vector_vector_add_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys
import argparse


from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

from aie.utils import tensor
import aie.iron as iron

# The JIT-compiled kernel relies on inputs from the command line.
# Because of that, we need to parse the arguments before the JIT is invoked.
parser = argparse.ArgumentParser()
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose output"
)
parser.add_argument(
    "-d",
    "--device",
    choices=["npu", "npu2", "xcvc1902"],
    default="npu",
    help="Target device",
)
parser.add_argument(
    "-c", "--column", type=int, default=0, help="Column index (default: 0)"
)
parser.add_argument(
    "-n",
    "--num-elements",
    type=int,
    default=32,
    help="Number of elements (default: 32)",
)
args = parser.parse_args()

device_map = {
    "npu": AIEDevice.npu1_1col,
    "npu2": AIEDevice.npu2_1col,
    "xcvc1902": AIEDevice.xcvc1902,
}
dev = device_map[args.device]
column_id = args.column
num_elements = args.num_elements
data_type = np.int32


@iron.jit(debug=True, verify=True)
def vector_vector_add():

    N = num_elements
    n = 16
    N_div_n = N // n

    buffer_depth = 2

    @device(dev)
    def device_body():
        tensor_ty = np.ndarray[(N,), np.dtype[data_type]]
        tile_ty = np.ndarray[(n,), np.dtype[data_type]]

        # AIE Core Function declarations

        # Tile declarations
        ShimTile = tile(column_id, 0)
        ComputeTile2 = tile(column_id, 2)

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
            in1_task = shim_dma_single_bd_task(of_in1, A, sizes=[1, 1, 1, N])
            in2_task = shim_dma_single_bd_task(of_in2, B, sizes=[1, 1, 1, N])
            out_task = shim_dma_single_bd_task(
                of_out, C, sizes=[1, 1, 1, N], issue_token=True
            )

            dma_start_task(in1_task, in2_task, out_task)
            dma_await_task(out_task)
            dma_free_task(in1_task, in2_task)


def main():

    input0 = tensor.random(
        (num_elements,), low=0, high=num_elements, dtype=data_type, device="npu"
    )
    input1 = tensor.random(
        (num_elements,), low=0, high=num_elements, dtype=data_type, device="npu"
    )
    output = tensor.zerolike(input0)

    vector_vector_add(input0, input1, output)

    e = np.equal(input0.numpy() + input1.numpy(), output.numpy())
    errors = np.size(e) - np.count_nonzero(e)

    if args.verbose:
        print(f"{'input0':>4} + {'input1':>4} = {'output':>4}")
        print("-" * 34)
        count = input0.numel()
        for idx, (a, b, c) in enumerate(
            zip(input0[:count], input1[:count], output[:count])
        ):
            print(f"{idx:2}: {a:4} + {b:4} = {c:4}")

    if not errors:
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    main()
