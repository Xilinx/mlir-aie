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
from aie.iron.device import NPU1Col1
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern

dev = NPU1Col1()

# Define tensor types
data_height = 3
data_width = 16
tile_height = 3
tile_width = 8
data_size = data_height * data_width
tile_size = tile_height * tile_width
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]
tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]

# Define runtime tensor access pattern (tap)
tensor_dims = (data_height, data_width)
offset = 0
sizes = [1, 2, 3, 8]
strides = [0, 8, 16, 1]
tap = TensorAccessPattern(tensor_dims, offset, sizes, strides)

tap.visualize(show_arrows=True, file_path="plot.png")

# Dataflow with ObjectFifos
of_in = ObjectFifo(tile_ty, name="in")
of_out = ObjectFifo(tile_ty, name="out")

# Task for the core to perform
def core_fn(of_in, of_out):
    elem_in = of_in.acquire(1)
    elem_out = of_out.acquire(1)
    for i in range_(tile_size):
        elem_out[i] = elem_in[i]
    of_in.release(1)
    of_out.release(1)

# Create a worker to perform the task
my_worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

# To/from AIE-array runtime data movement
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (a_in, _, c_out):
    rt.start(my_worker)
    rt.fill(of_in.prod(), a_in, tap)
    rt.drain(of_out.cons(), c_out, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
