# exercise_4a.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np

from aie.iron import Program, Runtime, Worker, ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern

import aie.iron as iron

dev = NPU2()

# Define tensor types
data_height = 3
data_width = 16
tile_height = 3
tile_width = 8
data_size = data_height * data_width
tile_size = tile_height * tile_width
element_type = np.int32
data_ty = np.ndarray[(data_size,), np.dtype[element_type]]
tile_ty = np.ndarray[(tile_size,), np.dtype[element_type]]


@iron.jit(is_placed=False)
def exercise_4a():
    # Dataflow with ObjectFifos
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    # Task for the core to perform
    def core_fn(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(tile_size):
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
    my_program = Program(dev, rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    return my_program.resolve_program(SequentialPlacer())


def main():

    # Construct an input tensor and an output zeroed tensor
    # The two tensors are in memory accessible to the NPU
    input0 = iron.arange(data_size, dtype=element_type, device="npu")
    output = iron.zeros(data_size, dtype=element_type, device="npu")

    # Generate reference pattern
    ref_vec = [k * 8 + j * 16 + i for k in range(2) for j in range(3) for i in range(8)]

    # JIT-compile the kernel then launches the kernel with the given arguments. Future calls
    # to the kernel will use the same compiled kernel and loaded code objects
    exercise_4a(input0, output)

    # Check the correctness of the result
    USE_REF_VEC = True  # Set to False to switch to output for user testing

    test_source = ref_vec if USE_REF_VEC else output
    errors = 0

    for index, (actual, ref) in enumerate(
        zip(
            test_source,
            [k * 8 + j * 16 + i for k in range(2) for j in range(3) for i in range(8)],
        )
    ):
        if actual != ref:
            print(f"Error in output {actual} != {ref}")
            errors += 1
        else:
            print(f"Correct output {actual} == {ref}")

    # If the result is correct, exit with a success code.
    # Otherwise, exit with a failure code
    if not errors:
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nError count: ", errors)
        print("\nfailed.\n")
        exit(1)


if __name__ == "__main__":
    main()
