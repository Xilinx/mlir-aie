# single_buffer.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_

if len(sys.argv) > 1:
    if sys.argv[1] == "npu":
        dev = NPU1Col1()
    elif sys.argv[1] == "npu2":
        dev = NPU2Col1()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

# Define tensor types
data_ty = np.ndarray[(16,), np.dtype[np.int32]]

# Dataflow with ObjectFifos
of_in = ObjectFifo(data_ty, name="in", depth=1)  # single buffer
of_out = ObjectFifo(data_ty, name="out", depth=1)  # single buffer


# Task for the core to perform
def core_fn(of_in):
    elem_out = of_in.acquire(1)
    for i in range_(16):
        elem_out[i] = 1
    of_in.release(1)


def core_fn2(of_in, of_out):
    elem_in = of_in.acquire(1)
    elem_out = of_out.acquire(1)
    for i in range_(16):
        elem_out[i] = elem_in[i]
    of_in.release(1)
    of_out.release(1)


# Create a worker to perform the task
my_worker = Worker(core_fn, [of_in.prod()])
my_worker2 = Worker(core_fn2, [of_in.cons(), of_out.prod()])

# To/from AIE-array runtime data movement
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (_, _, c_out):
    rt.start(my_worker)
    rt.start(my_worker2)
    rt.drain(of_out.cons(), c_out, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
