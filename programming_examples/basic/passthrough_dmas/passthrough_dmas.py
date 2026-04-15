# passthrough_dmas/passthrough_dmas.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU1Col1, NPU2Col1, XCVC1902

N = 4096
line_size = 1024

if len(sys.argv) > 1:
    N = int(sys.argv[1])
    assert N % line_size == 0

if len(sys.argv) > 2:
    if sys.argv[2] == "npu":
        dev = NPU1Col1()
    elif sys.argv[2] == "npu2":
        dev = NPU2Col1()
    elif sys.argv[2] == "xcvc1902":
        dev = XCVC1902()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

# Define tensor types
vector_ty = np.ndarray[(N,), np.dtype[np.int32]]
line_ty = np.ndarray[(line_size,), np.dtype[np.int32]]

# Data movement with ObjectFifos
of_in = ObjectFifo(line_ty, name="in")
of_out = of_in.cons().forward()

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(vector_ty, vector_ty, vector_ty) as (a_in, _, c_out):
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), c_out, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program()

# Print the generated MLIR
print(module)
