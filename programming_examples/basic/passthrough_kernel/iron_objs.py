# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
"""
Problems for clarify/conciseness:
* ObjectFifo needs (ordered) endpoints at instantiation
* Need introspection to declare functions/fifos on-the-fly so they still land in the symbol table
* Can remove type data if we're okay with inferring it through use (also required introspection) => but less verification if we go this route
    - Could we fix this somehow? e.g. loop emulation or something like that?
"""

from aie.dialects.scf import for_ as range_
from aie.dialects.scf import yield_

from aie.api.dataflow.inout.simplefifoinout import SimpleFifoInOutProgram
from aie.api.dataflow.objectfifo import MyObjectFifo
from aie.api.kernels.binkernel import BinKernel
from aie.api.phys.device import NPU1Col1
from aie.api.program import MyProgram
from aie.api.worker import MyWorker

import sys
import numpy as np

try:
    vector_size = int(sys.argv[1])
    if vector_size % 64 != 0 or vector_size < 512:
        print("Vector size must be a multiple of 64 and greater than or equal to 512")
        raise ValueError
except ValueError:
    print("Argument has inappropriate value")

assert vector_size % 4 == 0
line_size = vector_size // 4

inout_type = ((vector_size,), np.uint8)
fifo_memref_type = ((line_size,), np.uint8)

of0 = MyObjectFifo(2, memref_type=fifo_memref_type, name="out")
of1 = MyObjectFifo(2, memref_type=fifo_memref_type, name="in")

passthrough_fn = BinKernel(
    "passThroughLine",
    "passThrough.cc.o",
    [fifo_memref_type, fifo_memref_type, np.int32],
)


def core_fn(ofs_end1, ofs_end2, external_functions):
    of_out = ofs_end1[0]
    of_in = ofs_end2[0]
    passThroughLine = external_functions[0]

    for _ in range_(vector_size // line_size):
        elemOut = of_out.acquire_produce(1)
        elemIn = of_in.acquire_consume(1)
        passThroughLine(elemIn, elemOut, line_size)
        of_in.release_consume(1)
        of_out.release_produce(1)
        yield_([])


worker_program = MyWorker(core_fn, [of0], [of1], [passthrough_fn], coords=(0, 2))
inout_program = SimpleFifoInOutProgram(of0, vector_size, of1, vector_size)


my_program = MyProgram(
    NPU1Col1(), worker_programs=[worker_program], inout_program=inout_program
)
my_program.resolve_program()
