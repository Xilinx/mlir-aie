# section-1/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

from aie.iron import Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col4

# Task for the core to perform
def core_fn():
    # compute task

# Create a worker to perform the task
my_worker = Worker(core_fn, [])

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(vector_type, vector_type, vector_type) as (a_in, b_out, _):
    # runtime sequence tasks

# Create the program from the device type and runtime
my_program = Program(NPU1Col4(), rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
