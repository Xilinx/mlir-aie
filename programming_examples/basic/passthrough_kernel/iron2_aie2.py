# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron2.io.iocoordinator import IOCoordinator
from aie.iron2.dataflow.objectfifo import ObjectFifo
from aie.iron2.kernels.binkernel import BinKernel
from aie.iron2.program import Program
from aie.iron2.worker import Worker
from aie.iron2.phys.device import NPU1Col1
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.util import DataTiler

try:
    vector_size = int(sys.argv[1])
    if vector_size % 64 != 0 or vector_size < 512:
        print("Vector size must be a multiple of 64 and greater than or equal to 512")
        raise ValueError
except ValueError:
    print("Argument has inappropriate value")

line_size = vector_size // 4
line_type = np.ndarray[(line_size,), np.dtype[np.uint8]]
vector_type = np.ndarray[(vector_size,), np.dtype[np.uint8]]

io = IOCoordinator()
a_in = io.inout_data(vector_type)
b_out = io.inout_data(vector_type)
_unused = io.inout_data(vector_type)

of_in = ObjectFifo(2, line_type, "in")
of_out = ObjectFifo(2, line_type, "out")

tiler = DataTiler(vector_type)
for t in io.tile_loop(tiler):
    io.fill(of_in.first, t, a_in, coords=(0, 0))
    io.drain(of_out.second, t, b_out, coords=(0, 0), wait=True)

passthrough_fn = BinKernel(
    "passThroughLine",
    "passThrough.cc.o",
    [line_type, line_type, np.int32],
)


def core_fn(of_in, of_out, passThroughLine):
    for _ in range_(sys.maxsize):
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        passThroughLine(elemIn, elemOut, line_size)
        of_in.release(1)
        of_out.release(1)


my_worker = Worker(core_fn, [of_in.second, of_out.first, passthrough_fn], coords=(0, 2))

my_program = Program(NPU1Col1(), io, workers=[my_worker])
my_program.resolve_program()

# A: np.ndarray[(vector_size,), np.dtype[np.uint8]] = ...
# C: np.ndarray[(vector_size,), np.dtype[np.uint8]] = ...
# my_program.run(inputs=A, outputs=C)