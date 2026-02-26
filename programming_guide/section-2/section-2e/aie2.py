# section-2/section-2d/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_

if len(sys.argv) > 1:
    if sys.argv[1] == "npu":
        dev = NPU1Col1()
    elif sys.argv[1] == "npu2":
        dev = NPU2Col1()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

data_size = 48

# Define tensor types
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

# Input data movement
of_in = ObjectFifo(data_ty, name="in")
of_in1 = of_in.cons().forward(obj_type=data_ty, name="in1")

# Output data movement
of_out1 = ObjectFifo(data_ty, name="out1")
of_out = of_out1.cons().forward(obj_type=data_ty, name="out")


# Task for the core to perform
def core_fn(of_in, of_out):
    elem_in = of_in.acquire(1)
    elem_out = of_out.acquire(1)
    for i in range_(data_size):
        elem_out[i] = elem_in[i] + 1
    of_in.release(1)
    of_out.release(1)


# Create a worker to perform the task
my_worker = Worker(core_fn, [of_in1.cons(), of_out1.prod()])

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (a_in, b_out, _):
    rt.start(my_worker)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), b_out, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program()

# Print the generated MLIR
print(module)
