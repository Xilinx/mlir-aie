# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import sys
import numpy as np

# TODO: move maybe to aie.api.controlflow
from aie.extras.dialects.ext.scf import _for as range_

from aie.api.dataflow.inout.simplefifoinout import SimpleFifoInOutProgram
from aie.api.dataflow.objectfifo import MyObjectFifo
from aie.api.kernels.binkernel import BinKernel
from aie.api.phys.device import NPU1Col1
from aie.api.program import MyProgram
from aie.api.worker import MyWorker

try:
    vector_size = int(sys.argv[1])
    if vector_size % 64 != 0 or vector_size < 512:
        print("Vector size must be a multiple of 64 and greater than or equal to 512")
        raise ValueError
except ValueError:
    print("Argument has inappropriate value")

assert vector_size % 4 == 0
line_size = vector_size // 4
line_type = np.ndarray[np.uint8, (line_size,)]

# TODO: rely on depth inference
of_in = MyObjectFifo(2, line_type, shim_endpoint=(0, 0))
of_out = MyObjectFifo(2, line_type, shim_endpoint=(0, 0))

passthrough_fn = BinKernel(
    "passThroughLine",
    "passThrough.cc.o",
    [line_type, line_type, np.int32],
)


def core_fn(of_in, of_out, passThroughLine):
    for _ in range_(vector_size // line_size):
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        passThroughLine(elemIn, elemOut, line_size)
        of_in.release(1)
        of_out.release(1)


# TODO: clean up placement
worker_program = MyWorker(
    core_fn, [of_in.second, of_out.first, passthrough_fn], coords=(0, 2)
)
inout_program = SimpleFifoInOutProgram(
    of_in.first, vector_size, of_out.second, vector_size
)

my_program = MyProgram(
    NPU1Col1(), worker_programs=[worker_program], inout_program=inout_program
)
my_program.resolve_program()
