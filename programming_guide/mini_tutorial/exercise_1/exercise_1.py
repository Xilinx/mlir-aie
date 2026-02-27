# exercise_1.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import sys
import numpy as np

from aie.iron import Program, Runtime, Worker, ObjectFifo, Buffer
from aie.iron.controlflow import range_

import aie.iron as iron


@iron.jit(is_placed=False)
def exercise_1(output):
    data_size = output.numel()
    element_type = output.dtype
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]

    # Dataflow with ObjectFifos
    of_out = ObjectFifo(data_ty, name="out")

    buffer = Buffer(
        data_ty,
        name="buff",
        initial_value=np.array(range(data_size), dtype=element_type),
    )

    # Task for the core to perform
    def core_fn(of_out, buff):
        elem_out = of_out.acquire(1)
        for i in range_(data_size):
            elem_out[i] = buff[i]
        of_out.release(1)

    # Create a worker to perform the task
    my_worker = Worker(core_fn, [of_out.prod(), buffer])

    # To/from AIE-array runtime data movement
    rt = Runtime()
    with rt.sequence(data_ty) as (c_out):
        rt.start(my_worker)
        rt.drain(of_out.cons(), c_out, wait=True)

    # Create the program from the device type and runtime
    my_program = Program(iron.get_current_device(), rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    return my_program.resolve_program()


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
    exercise_1(output)

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
