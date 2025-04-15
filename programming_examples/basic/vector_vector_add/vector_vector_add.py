# vector_vector_add/vector_vector_add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys
import argparse

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1, XCVC1902
from aie.iron.controlflow import range_

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

if args.num_elements % 16 != 0:
    raise ValueError("num_elements must be a multiple of 16.")

device_map = {
    "npu": NPU1Col1(),
    "npu2": NPU2Col1(),
    # "xcvc1902": XCVC1902(),
    # Commented out because of the error:
    # TypeError: Can't instantiate abstract class XCVC1902 without an
    # implementation for abstract methods 'get_compute_tiles',
    # 'get_mem_tiles', 'get_shim_tiles'
}
dev = device_map[args.device]
num_elements = args.num_elements
data_type = np.int32


@iron.jit(is_placed=False)
def vector_vector_add():
    n = 16
    N_div_n = num_elements // n

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(n,), np.dtype[np.int32]]

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
    return Program(dev, rt).resolve_program(SequentialPlacer())


def main():

    input0 = iron.rand((num_elements,), dtype=data_type, device="npu")
    input1 = iron.rand((num_elements,), dtype=data_type, device="npu")
    output = iron.zeros_like(input0)

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
