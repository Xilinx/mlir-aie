# aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np

from aie.iron import Program, Runtime, Worker, ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_

dev = NPU2()

# Define tensor types
data_height = 3
data_width = 16
data_size = data_height * data_width
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

# Dataflow with ObjectFifos
size_2 = TODO
stride_2 = TODO
size_1 = TODO
stride_1 = TODO
size_0 = TODO
stride_0 = TODO
dims = [(size_2, stride_2), (size_1, stride_1), (size_0, stride_0)]
of_in = ObjectFifo(data_ty, name="in")
of_out = ObjectFifo(data_ty, name="out", dims_to_stream=dims)


# Task for the core to perform
def core_fn(of_in, of_out):
    elem_in = of_in.acquire(1)
    elem_out = of_out.acquire(1)
    for i in range_(data_size):
        elem_out[i] = elem_in[i]
    of_in.release(1)
    of_out.release(1)


# Create a worker to perform the task
my_worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

# To/from AIE-array runtime data movement
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (a_in, _, c_out):
    rt.start(my_worker)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), c_out, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
