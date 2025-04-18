# aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import sys

from aie.iron import Program, Runtime, Worker, ObjectFifo, GlobalBuffer, WorkerRuntimeBarrier
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1
from aie.iron.controlflow import range_

from aie.extras.dialects.ext import arith
from aie.helpers.util import np_ndarray_type_get_shape
from aie.dialects.aie import T

dev = NPU1Col1()

# Define tensor types
data_size = 48
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]
rtp_size = data_size // 2
rtp_ty = np.ndarray[(rtp_size,), np.dtype[np.int32]]

# Dataflow with ObjectFifos
of_out = ObjectFifo(data_ty, name="out")

# Runtime parameters
rtps = []
rtps.append(
    GlobalBuffer(
        rtp_ty,
        name=f"rtp",
        use_write_rtp=True,
    )
)

# Worker runtime barriers
workerBarrier = WorkerRuntimeBarrier()
workerBarrier2 = WorkerRuntimeBarrier()


# Task for the core to perform
def core_fn(rtp, of_out, barrier, barrier2):
    barrier.wait_for_value(1)
    elem_out = of_out.acquire(1)
    for i in range_(rtp_size):
        elem_out[i] = rtp[i]
    barrier2.wait_for_value(1)
    for i in range_(rtp_size):
        elem_out[rtp_size + i] = rtp[i]
    of_out.release(1)


# Create a worker to perform the task
my_worker = Worker(core_fn, [rtps[0], of_out.prod(), workerBarrier, workerBarrier2])

# To/from AIE-array runtime data movement
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (_, _, c_out):
    # Set runtime parameters
    def set_rtps_0(*rtps):
        for rtp in rtps:
            for i in range(rtp_size):
                rtp[i] = i

    def set_rtps_1(*rtps):
        for rtp in rtps:
            for i in range(rtp_size):
                rtp[i] = rtp_size + i

    rt.inline_ops(set_rtps_0, rtps)
    rt.set_barrier(workerBarrier, 1)
    rt.inline_ops(set_rtps_1, rtps)
    rt.set_barrier(workerBarrier2, 1)
    rt.start(my_worker)
    rt.drain(of_out.cons(), c_out, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
