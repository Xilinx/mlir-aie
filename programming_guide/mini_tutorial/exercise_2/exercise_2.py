# exercise_2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import sys
import numpy as np

from aie.iron import Program, Runtime, Worker, ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_

import aie.iron as iron


@iron.jit(is_placed=False)
def exercise_2(input0, output):
    data_size = output.numel()
    element_type = output.dtype
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]

    # Dataflow with ObjectFifos
    of_in = ObjectFifo(data_ty, name="in")
    of_out = ObjectFifo(data_ty, name="out")

    # Task for the core to perform
    def core_fn(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(data_size):
            elem_out[i] = elem_in[i]
        of_in.release(1)
        of_out.release(1)

    # Create a worker to perform the task
    my_worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    # To/from AIE-array runtime data movement
    rt = Runtime()
    with rt.sequence(data_ty, data_ty) as (a_in, c_out):
        rt.start(my_worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    # Create the program from the device type and runtime
    my_program = Program(iron.get_current_device(), rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    return my_program.resolve_program(SequentialPlacer())


def main():
    # Define tensor shapes and data types
    data_size = 48
    element_type = np.int32

    # Construct an input tensor and an output zeroed tensor
    # The two tensors are in memory accessible to the NPU
    input0 = iron.arange(data_size, dtype=element_type, device="npu")
    output = iron.zeros_like(input0)

    # JIT-compile the kernel then launches the kernel with the given arguments. Future calls
    # to the kernel will use the same compiled kernel and loaded code objects
    exercise_2(input0, output)

    # Check the correctness of the result
    e = np.equal(input0.numpy(), output.numpy())
    errors = np.size(e) - np.count_nonzero(e)

    # Print the results
    print(f"{'input0':>4} = {'output':>4}")
    print("-" * 34)
    count = input0.numel()
    for idx, (a, c) in enumerate(zip(input0[:count], output[:count])):
        print(f"{idx:2}: {a:4} = {c:4}")

    # If the result is correct, exit with a success code.
    # Otherwise, exit with a failure code
    if not errors:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print("\nError count: ", errors)
        print("\nfailed.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
