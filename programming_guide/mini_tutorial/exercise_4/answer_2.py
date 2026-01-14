# answer_2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import sys
import numpy as np

from aie.iron import (
    Program,
    Runtime,
    Worker,
    ObjectFifo,
    Buffer,
    WorkerRuntimeBarrier,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_

import aie.iron as iron


@iron.jit(is_placed=False)
def exercise_4(output):
    data_size = output.numel()
    element_type = output.dtype
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]

    # Dataflow with ObjectFifos
    of_out = ObjectFifo(data_ty, name="out")

    # Runtime parameters
    rtps = []
    rtps.append(
        Buffer(
            data_ty,
            name=f"rtp",
            use_write_rtp=True,
        )
    )
    # Worker runtime barriers
    workerBarrier = WorkerRuntimeBarrier()

    # Task for the core to perform
    def core_fn(rtp, of_out, barrier):
        barrier.wait_for_value(1)
        elem_out = of_out.acquire(1)
        for i in range_(data_size):
            elem_out[i] = rtp[i]
        of_out.release(1)

    # Create a worker to perform the task
    my_worker = Worker(core_fn, [rtps[0], of_out.prod(), workerBarrier])

    # To/from AIE-array runtime data movement
    rt = Runtime()
    with rt.sequence(data_ty) as (c_out):
        # Set runtime parameters
        def set_rtps(*args):
            for rtp in args:
                for i in range(data_size):  # note difference with range_ in the Worker
                    rtp[i] = i

        rt.inline_ops(set_rtps, rtps)
        rt.set_barrier(workerBarrier, 1)
        rt.start(my_worker)
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
    exercise_4(output)

    # Check the correctness of the result
    USE_INPUT_VEC = False  # Set to False to switch to output for user testing
    test_source = input0 if USE_INPUT_VEC else output
    e = np.equal(input0.numpy(), test_source.numpy())
    errors = np.size(e) - np.count_nonzero(e)

    # Print the results
    print(f"{'input0':>4} = {'output':>4}")
    print("-" * 34)
    count = input0.numel()
    for idx, (a, c) in enumerate(zip(input0[:count], test_source[:count])):
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
