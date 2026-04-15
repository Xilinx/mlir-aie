#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1, NPU2Col1


def passThroughAIE2(dev, width, height):
    lineWidthInBytes = width
    tensorSize = width * height

    # define types
    tensor_ty = np.ndarray[(tensorSize,), np.dtype[np.int8]]
    line_ty = np.ndarray[(lineWidthInBytes,), np.dtype[np.uint8]]

    # AIE Core Function declarations
    passThroughLineKernel = Kernel(
        "passThroughLine", "passThrough.cc.o", [line_ty, line_ty, np.int32]
    )

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(line_ty, name="in")
    of_out = ObjectFifo(line_ty, name="out")

    # Task for the core to perform
    def passthrough_fn(of_in, of_out, passThroughLine):
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        passThroughLine(elemIn, elemOut, width)
        of_in.release(1)
        of_out.release(1)

    # Create a worker to perform the task
    worker = Worker(
        passthrough_fn, [of_in.cons(), of_out.prod(), passThroughLineKernel]
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (inTensor, _, outTensor):
        rt.start(worker)
        rt.fill(of_in.prod(), inTensor)
        rt.drain(of_out.cons(), outTensor, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program()


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = NPU1Col1()
    elif device_name == "npu2":
        dev = NPU2Col1()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
    width = 512 if (len(sys.argv) != 4) else int(sys.argv[2])
    height = 9 if (len(sys.argv) != 4) else int(sys.argv[3])
except ValueError:
    print("Argument has inappropriate value")
module = passThroughAIE2(dev, width, height)
print(module)
