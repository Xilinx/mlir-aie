#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1, NPU2


def color_detect(dev, width, height):
    lineWidth = width
    lineWidthInBytes = width * 4
    tensorSize = width * height * 4  # 4 channels

    traceSize = 1024

    # Define types
    line_bytes_ty = np.ndarray[(lineWidthInBytes,), np.dtype[np.uint8]]
    line_ty = np.ndarray[(lineWidth,), np.dtype[np.uint8]]
    tensor_ty = np.ndarray[(tensorSize,), np.dtype[np.int8]]
    tensor_16x16_ty = np.ndarray[(16, 16), np.dtype[np.int32]]

    # AIE Core Function declarations
    rgba2hueLine = Kernel(
        "rgba2hueLine", "rgba2hue.cc.o", [line_bytes_ty, line_ty, np.int32]
    )
    thresholdLine = Kernel(
        "thresholdLine",
        "threshold.cc.o",
        [line_ty, line_ty, np.int32, np.int16, np.int16, np.int8],
    )
    bitwiseORLine = Kernel(
        "bitwiseORLine",
        "combined_bitwiseOR_gray2rgba_bitwiseAND.a",
        [line_ty, line_ty, line_ty, np.int32],
    )
    gray2rgbaLine = Kernel(
        "gray2rgbaLine",
        "combined_bitwiseOR_gray2rgba_bitwiseAND.a",
        [line_ty, line_bytes_ty, np.int32],
    )
    bitwiseANDLine = Kernel(
        "bitwiseANDLine",
        "combined_bitwiseOR_gray2rgba_bitwiseAND.a",
        [line_bytes_ty, line_bytes_ty, line_bytes_ty, np.int32],
    )

    # AIE-array data movement with object fifos
    # Input
    inOF_L3L2 = ObjectFifo(line_bytes_ty, name="inOF_L3L2")
    inOF_L2L1 = inOF_L3L2.cons(6).forward(depth=6, name="inOF_L2L1")

    # Output
    outOF_L1L2 = ObjectFifo(line_bytes_ty, name="outOF_L1L2")
    outOF_L2L3 = outOF_L1L2.cons().forward(name="outOF_L2L3")

    # Intermediate
    OF_2to34 = ObjectFifo(line_ty, name="OF_2to34")
    OF_3to3 = ObjectFifo(line_ty, name="OF_3to3", depth=1)
    OF_3to5 = ObjectFifo(line_ty, name="OF_3to5")
    OF_4to4 = ObjectFifo(line_ty, name="OF_4to4", depth=1)
    OF_4to5 = ObjectFifo(line_ty, name="OF_4to5")
    OF_5to5a = ObjectFifo(line_ty, name="OF_5to5a", depth=1)
    OF_5to5b = ObjectFifo(line_bytes_ty, name="OF_5to5b", depth=1)

    # Compute task for cores to perform
    def rgba2hue_fn(of_in, of_out, rgba2hueLine_kernel):
        elemIn = of_in.acquire(1)
        elemOut = of_out.acquire(1)
        rgba2hueLine_kernel(elemIn, elemOut, lineWidth)
        of_in.release(1)
        of_out.release(1)

    # worker to perform the task
    worker2 = Worker(rgba2hue_fn, [inOF_L3L2.cons(), OF_2to34.prod(), rgba2hueLine])

    # Compute task for cores to perform
    def threshold_fn(of_in, of_in3, of_out3, of_out5, threshold_kernel, is_first=True):
        if is_first:
            thresholdValueUpper1 = 40
            thresholdValueLower1 = 30
        else:
            thresholdValueUpper1 = 160
            thresholdValueLower1 = 90
        thresholdMaxvalue = 255
        thresholdModeToZeroInv = 4
        thresholdModeBinary = 0

        elemIn = of_in.acquire(1)
        elemOutTmp = of_in3.acquire(1)
        threshold_kernel(
            elemIn,
            elemOutTmp,
            lineWidth,
            thresholdValueUpper1,
            thresholdMaxvalue,
            thresholdModeToZeroInv,
        )
        of_in.release(1)
        of_in3.release(1)
        elemInTmp = of_out3.acquire(1)
        elemOut = of_out5.acquire(1)
        threshold_kernel(
            elemInTmp,
            elemOut,
            lineWidth,
            thresholdValueLower1,
            thresholdMaxvalue,
            thresholdModeBinary,
        )
        of_out3.release(1)
        of_out5.release(1)

    # worker to perform the task
    worker3 = Worker(
        threshold_fn,
        [
            OF_2to34.cons(),
            OF_3to3.prod(),
            OF_3to3.cons(),
            OF_3to5.prod(),
            thresholdLine,
            True,
        ],
    )

    # worker to perform the task
    worker4 = Worker(
        threshold_fn,
        [
            OF_2to34.cons(),
            OF_4to4.prod(),
            OF_4to4.cons(),
            OF_4to5.prod(),
            thresholdLine,
            False,
        ],
    )

    # Compute task for cores to perform
    def or_gray2rgba_and_fn(
        of_in,
        of_in2,
        of_in_self,
        of_out_self,
        of_in_self2,
        of_out_self2,
        of_in3,
        of_out,
        bitwiseORLine_kernel,
        gray2rgbaLine_kernel,
        bitwiseANDLine_kernel,
    ):
        # bitwise OR
        elemIn1 = of_in.acquire(1)
        elemIn2 = of_in2.acquire(1)
        elemOutTmpA = of_in_self.acquire(1)
        bitwiseORLine_kernel(elemIn1, elemIn2, elemOutTmpA, lineWidth)
        of_in.release(1)
        of_in2.release(1)
        of_in_self.release(1)
        # gray2rgba
        elemInTmpA = of_out_self.acquire(1)
        elemOutTmpB = of_in_self2.acquire(1)
        gray2rgbaLine_kernel(elemInTmpA, elemOutTmpB, lineWidth)
        of_out_self.release(1)
        of_in_self2.release(1)
        # bitwise AND
        elemInTmpB1 = of_out_self2.acquire(1)
        elemInTmpB2 = of_in3.acquire(1)
        elemOut = of_out.acquire(1)
        bitwiseANDLine_kernel(elemInTmpB1, elemInTmpB2, elemOut, lineWidthInBytes)
        of_out_self2.release(1)
        of_in3.release(1)
        of_out.release(1)

    # worker to perform the task
    worker5 = Worker(
        or_gray2rgba_and_fn,
        [
            OF_3to5.cons(),
            OF_4to5.cons(),
            OF_5to5a.prod(),
            OF_5to5a.cons(),
            OF_5to5b.prod(),
            OF_5to5b.cons(),
            inOF_L2L1.cons(),
            outOF_L1L2.prod(),
            bitwiseORLine,
            gray2rgbaLine,
            bitwiseANDLine,
        ],
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_16x16_ty, tensor_ty) as (I, B, O):
        rt.start(worker2, worker3, worker4, worker5)
        rt.fill(inOF_L3L2.prod(), I)
        rt.drain(outOF_L2L3.cons(), O, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program()


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = NPU1Col1()
    elif device_name == "npu2":
        dev = NPU2()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
    width = 36 if (len(sys.argv) != 4) else int(sys.argv[2])
    height = 64 if (len(sys.argv) != 4) else int(sys.argv[3])
except ValueError:
    print("Argument has inappropriate value")
module = color_detect(dev, width, height)
print(module)
