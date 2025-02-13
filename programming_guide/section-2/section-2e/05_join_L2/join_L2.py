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
# Output
of_offsets = [8 * worker for worker in range(n_workers)]

of_out = ObjectFifo(tile24_ty, name="out")
of_outs = of_out.prod().join(
    of_offsets,
    obj_types=[tile8_ty] * n_workers,
    names=[f"out{worker}" for worker in range(n_workers)],
)


    elem_out = of_out.acquire(1)
def core_fn(of_out):
    elem_out = of_out.acquire(1)
    of_out.release(1)


# Create a worker to perform the task
workers = []
for worker in range(n_workers):
    workers.append(
        Worker(
            core_fn,
            [
                of_outs[worker].prod(),
            ],
        )
    )

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (_, _, c_out):
    rt.start(*workers)
    rt.drain(of_out.cons(), c_out, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
