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
from aie.iron.device import NPU1Col1
from aie.iron.controlflow import range_

dev = NPU1Col1()
n_workers = 3

# Define tensor types
data_ty = np.ndarray[(48,), np.dtype[np.int32]]
tile24_ty = np.ndarray[(24,), np.dtype[np.int32]]
tile8_ty = np.ndarray[(8,), np.dtype[np.int32]]

# Dataflow with ObjectFifos
# Input
of_offsets = [8 * worker for worker in range(n_workers)]

of_in = ObjectFifo(tile24_ty, name="in")
of_ins = of_in.cons().split(
    of_offsets,
    obj_types=[tile8_ty] * n_workers,
    names=[f"in{worker}" for worker in range(n_workers)],
)


# Task for the core to perform
def core_fn(of_in):
    elem_in = of_in.acquire(1)
    of_in.release(1)


# Create a worker to perform the task
workers = []
for worker in range(n_workers):
    workers.append(
        Worker(
            core_fn,
            [
                of_ins[worker].cons(),
            ],
        )
    )

# To/from AIE-array runtime data movement
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (a_in, _, _):
    rt.start(*workers)
    rt.fill(of_in.prod(), a_in)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
