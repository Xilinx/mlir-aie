# passthrough_kernel/passthrough_kernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, XCVC1902

try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = NPU1Col1()
    elif device_name == "npu2":
        dev = XCVC1902()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
    vector_size = int(sys.argv[2])
    if vector_size % 64 != 0 or vector_size < 512:
        print("Vector size must be a multiple of 64 and greater than or equal to 512")
        raise ValueError
    trace_size = 0 if (len(sys.argv) != 4) else int(sys.argv[3])
except ValueError:
    print("Argument has inappropriate value")

# Define tensor types
line_size = vector_size // 4
line_type = np.ndarray[(line_size,), np.dtype[np.uint8]]
vector_type = np.ndarray[(vector_size,), np.dtype[np.uint8]]

# Dataflow with ObjectFifos
of_in = ObjectFifo(line_type, name="in")
of_out = ObjectFifo(line_type, name="out")

# External, binary kernel definition
passthrough_fn = Kernel(
    "passThroughLine",
    "passThrough.cc.o",
    [line_type, line_type, np.int32],
)


# Task for the core to perform
def core_fn(of_in, of_out, passThroughLine):
    elemOut = of_out.acquire(1)
    elemIn = of_in.acquire(1)
    passThroughLine(elemIn, elemOut, line_size)
    of_in.release(1)
    of_out.release(1)


# Create a worker to perform the task
my_worker = Worker(core_fn, [of_in.cons(), of_out.prod(), passthrough_fn])

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(vector_type, vector_type, vector_type) as (a_in, b_out, _):
    rt.start(my_worker)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), b_out, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
