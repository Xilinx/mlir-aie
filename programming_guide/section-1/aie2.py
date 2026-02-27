# section-1/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np

from aie.iron import Program, Runtime, Worker, Buffer
from aie.iron.device import NPU1Col1, Tile
from aie.iron.controlflow import range_

data_size = 48
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

# Dataflow configuration
# described in a future section of the guide...

buffer = Buffer(data_ty, name="buff")


# Task for the worker to perform
def core_fn(buff):
    for i in range_(data_size):
        buff[i] = 0


# Create a worker to perform the task
my_worker = Worker(core_fn, [buffer], placement=Tile(0, 2), while_true=False)

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (_, _, _):
    rt.start(my_worker)

# Create the program from the device type and runtime
my_program = Program(NPU1Col1(), rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program()

# Print the generated MLIR
print(module)
