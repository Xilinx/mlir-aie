# section-1/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import sys

from aie.iron import Program, Runtime, Worker, GlobalBuffer
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col4, Tile
from aie.iron.controlflow import range_

data_size = 48
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

# Dataflow configuration
# described in a future section of the guide...


buff = GlobalBuffer(data_ty, name="buff")


def core_fn(buff_in):
    for i in range_(data_size):
        buff_in[i] = buff_in[i] + 1


# Create a worker to perform the task
my_worker = Worker(core_fn, [buff], placement=Tile(0, 2))

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (_, _, _):
    rt.start(my_worker)

# Create the program from the device type and runtime
my_program = Program(NPU1Col4(), rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
