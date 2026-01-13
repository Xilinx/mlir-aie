# aie2p.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import sys

from aie.iron import Program, Runtime, Worker, ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_

import aie.iron as iron

# Define tensor types
# These represent the size and datatype of the data tensors on which we do compute or move
# with the ObjectFifo primitive.
num_elements = 48
data_type = np.int32
tile_ty = np.ndarray[(num_elements,), np.dtype[data_type]]


# JIT decorator for IRON
# Decorator to compile an IRON kernel into a binary to run on the NPU.
# Parameters:
#     - use_cache (bool): Use cached MLIR module if available. Defaults to True.
@iron.jit
def aie2p(input0, output):
    # Dataflow with ObjectFifos
    # ObjectFifos represent a dataflow connection between endpoints in the AIE array.
    # The IRON placement step relies on ObjectFifoHandles to infer the endpoints of ObjectFifos.
    # Each ObjectFifo can have one producer and one or multiple consumer ObjectFifoHandles.
    # These endpoints can be used by the Workers, the Runtime or to generate connecting ObjectFifos
    # (with forward(), split(), or join()).
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    # Task for the core to perform
    def core_fn(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(num_elements):
            elem_out[i] = elem_in[i]
        of_in.release(1)
        of_out.release(1)

    # Create a worker to perform the task
    # The arguments to a Worker are a task and then a list of arguments (inputs and outputs) for that task.
    # The same task can be assigned to multiple Workers, typically with different input arguments.
    my_worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    # To/from AIE-array runtime data movement
    # The arguments of the runtime sequence describe buffers that will be available on the host side;
    # the body of the sequence contains commands which describe how those buffers are moved into the AIE-array.
    # Runtime sequence commands are submitted to and executed by a dedicated command processor in order.
    # Commands that are set to 'wait' will block the dedicated command processor until a token associated
    # with their completion is generated.
    # The command processor waits for the runtime sequence to complete before returning by interrupting
    # the host processor.

    # Note: The task to start the workers currently only registers the workers as part of the Program, but
    # does not launch the Workers themselves. This means that this task can be added in the sequence at
    # any point and does not need to be the first one.
    rt = Runtime()
    with rt.sequence(tile_ty, tile_ty) as (a_in, c_out):
        rt.start(my_worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    # Create the program from the device type and runtime
    my_program = Program(iron.get_current_device(), rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    # The placer will use available information, such as the ObjectFifoHandles, to place the components.
    # After the placement is complete, the resolve_program() will check that each component has sufficient
    # information to be lowered to its MLIR equivalent.
    # At this point, the program is also verified and will report underlying MLIR errors, if any.
    # You can see a list of available placers in python/iron/placers.py
    return my_program.resolve_program(SequentialPlacer())


def main():

    # Construct an input tensor and an output zeroed tensor
    # The two tensors are in memory accessible to the NPU
    input0 = iron.arange(num_elements, dtype=data_type, device="npu")
    output = iron.zeros_like(input0)

    # JIT-compile the kernel then launches the kernel with the given arguments. Future calls
    # to the kernel will use the same compiled kernel and loaded code objects
    aie2p(input0, output)

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
