# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import sys
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

devices = {
    "npu": NPU1Col1(),
    "npu2": NPU2Col1()
}
if len(sys.argv) != 2 or sys.argv[1] not in devices:
    print(f"Usage {sys.argv[0]} <{'|'.join(devices.keys())}>")
    sys.exit(1)
device = devices[sys.argv[1]]

N = 4096 # Tensor size


# --------------------------------------------------------------------------
# In-Array Data Movement
# --------------------------------------------------------------------------

in_ty = np.ndarray[(N,), np.dtype[bfloat16]]
out_ty = np.ndarray[(N,), np.dtype[bfloat16]]

of_x = ObjectFifo(in_ty, name="x")
of_y = ObjectFifo(in_ty, name="y")
of_z = ObjectFifo(out_ty, name="z")


# --------------------------------------------------------------------------
# Task each core will run
# --------------------------------------------------------------------------

# The kernel acquires input tensors X and Y, and output tensor Z, performs the
# SAXPY operation on X and Y, and writes the result in Z.

saxpy_kernel = Kernel("saxpy", "saxpy.o", [in_ty, in_ty, out_ty])
def core_body(of_x, of_y, of_z, saxpy_kernel):
    elem_x = of_x.acquire(1)
    elem_y = of_y.acquire(1)
    elem_z = of_z.acquire(1)
    saxpy_kernel(elem_x, elem_y, elem_z)
    of_x.release(1)
    of_y.release(1)
    of_z.release(1)
worker = Worker(core_body, fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), saxpy_kernel], trace=1)


# --------------------------------------------------------------------------
# DRAM-NPU data movement and work dispatch
# --------------------------------------------------------------------------

rt = Runtime()
with rt.sequence(in_ty, in_ty, out_ty) as (a_x, a_y, c_z):
    rt.enable_trace(16384)
    rt.start(worker)
    rt.fill(of_x.prod(), a_x)
    rt.fill(of_y.prod(), a_y)
    rt.drain(of_z.cons(), c_z, wait=True)


# --------------------------------------------------------------------------
# Place and generate MLIR program
# --------------------------------------------------------------------------

program = Program(device, rt)
module = program.resolve_program(SequentialPlacer())
print(module)
