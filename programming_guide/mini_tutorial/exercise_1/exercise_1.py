# exercise_1.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import sys

from aie.iron import Program, Runtime, Worker, ObjectFifo, GlobalBuffer
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_

import aie.iron as iron

# Define tensor types
num_elements = 48
data_type = np.int32
tile_ty = np.ndarray[(num_elements,), np.dtype[data_type]]


@iron.jit(is_placed=False)
def exercise_1(output):
    # Dataflow with ObjectFifos
    of_out = ObjectFifo(tile_ty, name="out")

    buff = GlobalBuffer(
        tile_ty,
        name="buff",
        initial_value=np.array(range(num_elements), dtype=data_type),
    )

    # Task for the core to perform
    def core_fn(buff_in, of_out):
        elem_out = of_out.acquire(1)
        for i in range_(num_elements):
            elem_out[i] = buff_in[i]
        of_out.release(1)

    # Create a worker to perform the task
    my_worker = Worker(core_fn, [buff, of_out.prod()])

    # To/from AIE-array runtime data movement
    rt = Runtime()
    with rt.sequence(tile_ty) as (c_out):
        rt.start(my_worker)
        rt.drain(of_out.cons(), c_out, wait=True)

    # Create the program from the device type and runtime
    my_program = Program(iron.get_current_device(), rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    return my_program.resolve_program(SequentialPlacer())


def main():

    # Construct an input tensor and an output zeroed tensor
    # The two tensors are in memory accessible to the NPU
    input0 = iron.arange(num_elements, dtype=data_type, device="npu")
    output = iron.zeros_like(input0)

    iron.set_current_device(NPU2())

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
