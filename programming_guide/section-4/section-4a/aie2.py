# section-3/aie2.py -*- Python -*-
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

tensor_size = 4096
tile_size = tensor_size // 4

# Define tensor types
tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int32]]
tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

# External, binary kernel definition
scale_fn = Kernel(
    "vector_scalar_mul_aie_scalar",
    "scale.o",
    [tile_ty, tile_ty, scalar_ty, np.int32],
)

# Input data movement
of_in = ObjectFifo(tile_ty, name="in")
of_factor = ObjectFifo(scalar_ty, name="infactor")

# Output data movement
of_out = ObjectFifo(tile_ty, name="out")


# Task for the core to perform
def core_fn(of_in, of_factor, of_out, scale_scalar):
    elem_factor = of_factor.acquire(1)
    for _ in range_(4):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        scale_scalar(elem_in, elem_out, elem_factor, 1024)
        of_in.release(1)
        of_out.release(1)
    of_factor.release(1)


# Create a worker to perform the task
my_worker = Worker(core_fn, [of_in.cons(), of_factor.cons(), of_out.prod(), scale_fn])

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(tensor_ty, scalar_ty, tensor_ty) as (a_in, f_in, c_out):
    rt.start(my_worker)
    rt.fill(of_in.prod(), a_in)
    rt.fill(of_factor.prod(), f_in)
    rt.drain(of_out.cons(), c_out, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
