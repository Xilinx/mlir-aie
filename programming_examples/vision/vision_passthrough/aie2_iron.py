#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
import numpy as np
import sys

from aie.iron.runtime import Runtime
from aie.iron.dataflow import ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.program import Program
from aie.iron.worker import Worker
from aie.iron.kernels import BinKernel
from aie.iron.phys.device import NPU1Col1

width = 512  # 1920 // 8
height = 9  # 1080 // 8
if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])

lineWidthInBytes = width
tensorSize = width * height

enableTrace = False
traceSizeInBytes = 8192
traceSizeInInt32s = traceSizeInBytes // 4


def passThroughAIE2():
    # define types
    tensor_ty = np.ndarray[(tensorSize,), np.dtype[np.int8]]
    line_ty = np.ndarray[(lineWidthInBytes,), np.dtype[np.uint8]]

    # AIE Core Function declarations
    passThroughLineKernel = BinKernel(
        "passThroughLine", "passThrough.cc.o", [line_ty, line_ty, np.int32]
    )

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(line_ty, name="in")
    of_out = ObjectFifo(line_ty, name="out")

    def passthrough_fn(of_in, of_out, passThroughLine):
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        passThroughLine(elemIn, elemOut, width)
        of_in.release(1)
        of_out.release(1)

    worker = Worker(
        passthrough_fn, [of_in.cons(), of_out.prod(), passThroughLineKernel]
    )

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (inTensor, _, outTensor):
        rt.start(worker)
        rt.fill(of_in.prod(), inTensor)
        rt.drain(of_out.cons(), outTensor, wait=True)

    return Program(NPU1Col1(), rt).resolve_program(SequentialPlacer())


module = passThroughAIE2()
print(module)
