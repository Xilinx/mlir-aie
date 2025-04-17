# aie2.py -*- Python -*-
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
from aie.iron.device import NPU2
from aie.iron.controlflow import range_

# The device on which the Program will be executed. Should match available hardware architecture.
# You can see a list of available devices in python/iron/device/__init__.py
dev = NPU2()

# Define tensor types
# These represent the size and datatype of the data tensors on which we do compute or move
# with the ObjectFifo primitive.
data_size = 48
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

# Dataflow with ObjectFifos
# ObjectFifos represent a dataflow connection between endpoints in the AIE array.
# The IRON placement step relies on ObjectFifoHandles to infer the endpoints of ObjectFifos.
# Each ObjectFifo can have one producer and one or multiple consumer ObjectFifoHandles.
# These endpoints can be used by the Workers, the Runtime or to generate connecting ObjectFifos
# (with forward(), split(), or join()).
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
# The Worker takes as input a task and the input arguments it should run it with.
# The same task can be duplicated in multiple Workers, typically with different input arguments.
my_worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

# To/from AIE-array runtime data movement
# The arguments of the runtime sequence describe buffers that will be available on the host side;
# the body of the sequence contains tasks which describe how those buffers are moved into the AIE-array.
# Runtime sequence tasks are submitted to and executed by a dedicated command processor in order.
# Tasks which are set to 'wait' will block the dedicated command processor until a token associated
# with their completion is generated.
# The host processor waits for the full runtime sequence to complete before reading the buffers.

# Note: The blocking instruction submitted to the dedicated command processor will be satisfied by
# any token, not necessarily the one emitted by the task that generated the instruction.

# Note: The task to start the workers currently only registers the workers as part of the Program, but
# does not launch the Workers themselves. This means that this task can be added in the sequence at
# any point and does not need to be the first one.
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (a_in, _, c_out):
    rt.start(my_worker)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), c_out, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
# The placer will use available information, such as the ObjectFifoHandles, to place the components.
# After the placement is complete, the resolve_program() will check that each component has sufficient
# information to be lowered to its MLIR equivalent.
# At this point, the program is also verified and will report underlying MLIR errors, if any.
# You can see a list of available placers in python/iron/placers.py
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
