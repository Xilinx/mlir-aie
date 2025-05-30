# saxpy/saxpy.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1


if len(sys.argv) > 2:
    if sys.argv[1] == "npu":
        dev = NPU1Col1()
    elif sys.argv[1] == "npu2":
        dev = NPU2Col1()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))


def my_saxpy():
    N = 4096

    in_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(N,), np.dtype[bfloat16]]

    # AIE-array data movement with object fifos
    of_x = ObjectFifo(in_ty, name="x")
    of_y = ObjectFifo(in_ty, name="y")
    of_z = ObjectFifo(out_ty, name="z")

    # AIE Core Function declarations
    saxpy_kernel = Kernel("saxpy", "saxpy.o", [in_ty, in_ty, out_ty])

    # A task for a core to perform
    def core_body(of_x, of_y, of_z, saxpy_kernel):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        saxpy_kernel(elem_x, elem_y, elem_z)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)

    # Create a worker to run a task on a core
    worker = Worker(core_body, fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), saxpy_kernel], trace=1)

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(in_ty, in_ty, out_ty) as (a_x, a_y, c_z):
        rt.enable_trace(16384)
        rt.start(worker)
        rt.fill(of_x.prod(), a_x)
        rt.fill(of_y.prod(), a_y)
        rt.drain(of_z.cons(), c_z, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


print(my_saxpy())
