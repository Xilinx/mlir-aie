# exercise_3.py -*- Python -*-
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
<<<<<<< HEAD
def exercise_3(input0, output):
    num_elements = output.numel()
    data_type = output.dtype
    tile_ty = np.ndarray[(num_elements,), np.dtype[data_type]]

    # Dataflow with ObjectFifos
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

=======
def exercise_3(output):
    data_size = output.numel()
    element_type = output.dtype
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]

    # Dataflow with ObjectFifos
    of_out = ObjectFifo(data_ty, name="out")

    # Runtime parameters
    rtps = []
    rtps.append(
        GlobalBuffer(
            data_ty,
            name=f"rtp",
            use_write_rtp=True,
        )
    )

>>>>>>> 2cf6582df15d38ac491407ff2826bda153656e43
    # Task for the core to perform
    def core_fn(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
<<<<<<< HEAD
        for i in range_(num_elements):
            elem_out[i] = elem_in[i]
        of_in.release(1)
=======
        for i in range_(data_size):
            elem_out[i] = rtp[i]
>>>>>>> 2cf6582df15d38ac491407ff2826bda153656e43
        of_out.release(1)

    # Create a worker to perform the task
    my_worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    # To/from AIE-array runtime data movement
    rt = Runtime()
<<<<<<< HEAD
    with rt.sequence(tile_ty, tile_ty) as (a_in, c_out):
=======
    with rt.sequence(data_ty) as (c_out):
        # Set runtime parameters
        def set_rtps(*args):
            for rtp in args:
                for i in range(
                    data_size
                ):  # note difference with range_ in the Worker
                    rtp[i] = i

        rt.inline_ops(set_rtps, rtps)
>>>>>>> 2cf6582df15d38ac491407ff2826bda153656e43
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
    exercise_3(input0, output)

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
