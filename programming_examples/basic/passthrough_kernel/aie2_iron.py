# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron.io.iocoordinator import IOCoordinator
from aie.iron.dataflow.objectfifo import ObjectFifo
from aie.iron.kernels.binkernel import BinKernel
from aie.iron.placers import SequentialPlacer
from aie.iron.program import Program
from aie.iron.worker import Worker
from aie.iron.phys.device import NPU1Col1
from aie.helpers.taplib import TensorTiler2D

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

of_in = ObjectFifo(2, line_type, "in")
of_out = ObjectFifo(2, line_type, "out")

tap = TensorTiler2D.simple_tiler((1, vector_size))[0]

io = IOCoordinator()
with io.runtime_sequence(vector_type, vector_type, vector_type) as (a_in, b_out, _):
    io.fill(of_in.prod, tap, a_in)
    io.drain(of_out.cons, tap, b_out, wait=True)

passthrough_fn = BinKernel(
    "passThroughLine",
    "passThrough.cc.o",
    [line_type, line_type, np.int32],
)


def core_fn(of_in, of_out, passThroughLine):
    elemOut = of_out.acquire(1)
    elemIn = of_in.acquire(1)
    passThroughLine(elemIn, elemOut, line_size)
    of_in.release(1)
    of_out.release(1)


my_worker = Worker(core_fn, [of_in.cons, of_out.prod, passthrough_fn], while_true=True)

my_program = Program(NPU1Col1(), io, workers=[my_worker])
my_program.resolve_program(SequentialPlacer())
