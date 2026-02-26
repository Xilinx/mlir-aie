# answer_2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import sys
import numpy as np

from aie.iron import Program, Runtime, Worker, ObjectFifo
from aie.iron.controlflow import range_

import aie.iron as iron


@iron.jit(is_placed=False)
def exercise_2(input0, output):
    data_size = output.numel()
    element_type = output.dtype
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]

    n_workers = 3
    tile_sizes = [8, 24, 16]
    tile_types = []
    for i in range(n_workers):
        tile_types.append(np.ndarray[(tile_sizes[i],), np.dtype[element_type]])

    # Dataflow with ObjectFifos
    of_offsets = [0, 8, 32]

    of_in = ObjectFifo(data_ty, name="in")
    of_ins = of_in.cons().split(
        of_offsets,
        obj_types=tile_types,
        names=[f"in{worker}" for worker in range(n_workers)],
    )

    of_out = ObjectFifo(data_ty, name="out")
    of_outs = of_out.prod().join(
        of_offsets,
        obj_types=tile_types,
        names=[f"out{worker}" for worker in range(n_workers)],
    )

    # Task for the core to perform
    def core_fn(of_in, of_out, num_elem):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(num_elem):
            elem_out[i] = elem_in[i]
        of_in.release(1)
        of_out.release(1)

    # Create workers to perform the task
    workers = []
    for i in range(n_workers):
        workers.append(
            Worker(
                core_fn,
                [
                    of_ins[i].cons(),
                    of_outs[i].prod(),
                    tile_sizes[i],
                ],
            )
        )

    # To/from AIE-array runtime data movement
    rt = Runtime()
    with rt.sequence(data_ty, data_ty) as (a_in, c_out):
        rt.start(*workers)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    # Create the program from the device type and runtime
    my_program = Program(iron.get_current_device(), rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    return my_program.resolve_program()


def main():
    # Define tensor shapes and data types
    num_elements = 48
    element_type = np.int32

    # Construct an input tensor and an output zeroed tensor
    # The two tensors are in memory accessible to the NPU
    input0 = iron.arange(num_elements, dtype=element_type, device="npu")
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
