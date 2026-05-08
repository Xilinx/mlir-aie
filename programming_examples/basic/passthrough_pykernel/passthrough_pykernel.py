# passthrough_pykernel/passthrough_pykernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.dialects.func import func

dev = NPU1Col1()

if len(sys.argv) > 2:
    if sys.argv[2] == "npu":
        dev = NPU1Col1()
    elif sys.argv[2] == "npu2":
        dev = NPU2()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

try:
    vector_size = int(sys.argv[1])
    if vector_size % 64 != 0 or vector_size < 512:
        print("Vector size must be a multiple of 64 and greater than or equal to 512")
        raise ValueError
except ValueError:
    print("Argument has inappropriate value")

# Define tensor types
line_size = vector_size // 4
line_type = np.ndarray[(line_size,), np.dtype[np.uint8]]
vector_type = np.ndarray[(vector_size,), np.dtype[np.uint8]]

# Dataflow with ObjectFifos
of_in = ObjectFifo(line_type, name="in")
of_out = ObjectFifo(line_type, name="out")


# A python function which will be treated as a callable function on the AIE
# e.g., a kernel written in python
@func
def passthrough_fn(input: line_type, output: line_type, lineWidth: np.int32):
    for i in range_(lineWidth):
        output[i] = input[i]


# The task for the core to perform (the core entry point, if you will)
def core_fn(of_in, of_out, passthrough_fn):
    elemOut = of_out.acquire(1)
    elemIn = of_in.acquire(1)
    passthrough_fn(elemIn, elemOut, line_size)
    of_in.release(1)
    of_out.release(1)


# Create a worker to run the task
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
module = my_program.resolve_program()

# Print the generated MLIR
print(module)
