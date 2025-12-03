# from_stream.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron import ObjectFifo, Program, Runtime, Worker
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
n_workers = 2

# Define tensor types
data_ty = np.ndarray[(48,), np.dtype[np.int32]]
tile24_ty = np.ndarray[(24,), np.dtype[np.int32]]

# Dataflow with ObjectFifos
# Input
of_in0 = ObjectFifo(tile24_ty, name="in0")
of_in1 = of_in0.cons().forward(
    name="in1", obj_type=tile24_ty, dims_from_stream=[(8, 3), (1, 8)]
)

# Output
of_out1 = ObjectFifo(tile24_ty, name="out1")
of_out0 = of_out1.cons().forward(name="out0", obj_type=tile24_ty)


# Task for the core to perform
def core_fn(of_in, of_out):
    elem_in = of_in.acquire(1)
    elem_out = of_out.acquire(1)
    for i in range_(24):
        elem_out[i] = elem_in[i]
    of_in.release(1)
    of_out.release(1)


# Create a worker to perform the task
my_worker = Worker(core_fn, [of_in1.cons(), of_out1.prod()])
    
# To/from AIE-array runtime data movement
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (a_in, _, c_out):
    rt.start(my_worker)
    rt.fill(of_in0.prod(), a_in)
    rt.drain(of_out0.cons(), c_out, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
