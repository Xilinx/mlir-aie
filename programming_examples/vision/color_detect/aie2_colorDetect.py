#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Xilinx Inc.
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext.scf import _for as range_

width = 64
height = 36
if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])

lineWidth = width
lineWidthInBytes = width * 4
tensorSize = width * height * 4  # 4 channels

enableTrace = False
traceSize = 1024


def color_detect():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def deviceBody():
            line_bytes_ty = np.ndarray[(lineWidthInBytes,), np.dtype[np.uint8]]
            line_ty = np.ndarray[(lineWidth,), np.dtype[np.uint8]]

            tensor_ty = np.ndarray[(tensorSize,), np.dtype[np.int8]]
            tensor_16x16_ty = np.ndarray[(16, 16), np.dtype[np.int32]]

            # AIE Core Function declarations
            rgba2hueLine = external_func(
                "rgba2hueLine", inputs=[line_bytes_ty, line_ty, np.int32]
            )
            thresholdLine = external_func(
                "thresholdLine",
                inputs=[line_ty, line_ty, np.int32, np.int16, np.int16, np.int8],
            )
            bitwiseORLine = external_func(
                "bitwiseORLine", inputs=[line_ty, line_ty, line_ty, np.int32]
            )
            gray2rgbaLine = external_func(
                "gray2rgbaLine", inputs=[line_ty, line_bytes_ty, np.int32]
            )
            bitwiseANDLine = external_func(
                "bitwiseANDLine",
                inputs=[line_bytes_ty, line_bytes_ty, line_bytes_ty, np.int32],
            )

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)
            ComputeTile3 = tile(0, 3)
            ComputeTile4 = tile(0, 4)
            ComputeTile5 = tile(0, 5)

            # AIE-array data movement with object fifos

            # Input
            inOF_L3L2 = object_fifo(
                "inOF_L3L2",
                ShimTile,
                [ComputeTile2, MemTile],
                [2, 2, 6],
                line_bytes_ty,
            )
            inOF_L2L1 = object_fifo(
                "inOF_L2L1", MemTile, ComputeTile5, 6, line_bytes_ty
            )
            object_fifo_link(inOF_L3L2, inOF_L2L1)

            # Output
            outOF_L2L3 = object_fifo("outOF_L2L3", MemTile, ShimTile, 2, line_bytes_ty)
            outOF_L1L2 = object_fifo(
                "outOF_L1L2", ComputeTile5, MemTile, 2, line_bytes_ty
            )
            object_fifo_link(outOF_L1L2, outOF_L2L3)

            # Intermediate
            OF_2to34 = object_fifo(
                "OF_2to34", ComputeTile2, [ComputeTile3, ComputeTile4], 2, line_ty
            )
            OF_3to3 = object_fifo("OF_3to3", ComputeTile3, ComputeTile3, 1, line_ty)
            OF_3to5 = object_fifo("OF_3to5", ComputeTile3, ComputeTile5, 2, line_ty)
            OF_4to4 = object_fifo("OF_4to4", ComputeTile4, ComputeTile4, 1, line_ty)
            OF_4to5 = object_fifo("OF_4to5", ComputeTile4, ComputeTile5, 2, line_ty)
            OF_5to5a = object_fifo("OF_5to5a", ComputeTile5, ComputeTile5, 1, line_ty)
            OF_5to5b = object_fifo(
                "OF_5to5b", ComputeTile5, ComputeTile5, 1, line_bytes_ty
            )

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2, "rgba2hue.cc.o")
            def coreBody():
                for _ in range_(sys.maxsize):
                    elemIn = inOF_L3L2.acquire(ObjectFifoPort.Consume, 1)
                    elemOut = OF_2to34.acquire(ObjectFifoPort.Produce, 1)
                    rgba2hueLine(elemIn, elemOut, lineWidth)
                    inOF_L3L2.release(ObjectFifoPort.Consume, 1)
                    OF_2to34.release(ObjectFifoPort.Produce, 1)

            # Compute tile 3
            @core(ComputeTile3, "threshold.cc.o")
            def coreBody():
                thresholdValueUpper1 = 40
                thresholdValueLower1 = 30
                thresholdMaxvalue = 255
                thresholdModeToZeroInv = 4
                thresholdModeBinary = 0
                for _ in range_(sys.maxsize):
                    elemIn = OF_2to34.acquire(ObjectFifoPort.Consume, 1)
                    elemOutTmp = OF_3to3.acquire(ObjectFifoPort.Produce, 1)
                    thresholdLine(
                        elemIn,
                        elemOutTmp,
                        lineWidth,
                        thresholdValueUpper1,
                        thresholdMaxvalue,
                        thresholdModeToZeroInv,
                    )
                    OF_2to34.release(ObjectFifoPort.Consume, 1)
                    OF_3to3.release(ObjectFifoPort.Produce, 1)
                    elemInTmp = OF_3to3.acquire(ObjectFifoPort.Consume, 1)
                    elemOut = OF_3to5.acquire(ObjectFifoPort.Produce, 1)
                    thresholdLine(
                        elemInTmp,
                        elemOut,
                        lineWidth,
                        thresholdValueLower1,
                        thresholdMaxvalue,
                        thresholdModeBinary,
                    )
                    OF_3to3.release(ObjectFifoPort.Consume, 1)
                    OF_3to5.release(ObjectFifoPort.Produce, 1)

            # Compute tile 4
            @core(ComputeTile4, "threshold.cc.o")
            def coreBody():
                thresholdValueUpper1 = 160
                thresholdValueLower1 = 90
                thresholdMaxvalue = 255
                thresholdModeToZeroInv = 4
                thresholdModeBinary = 0
                for _ in range_(sys.maxsize):
                    elemIn = OF_2to34.acquire(ObjectFifoPort.Consume, 1)
                    elemOutTmp = OF_4to4.acquire(ObjectFifoPort.Produce, 1)
                    thresholdLine(
                        elemIn,
                        elemOutTmp,
                        lineWidth,
                        thresholdValueUpper1,
                        thresholdMaxvalue,
                        thresholdModeToZeroInv,
                    )
                    OF_2to34.release(ObjectFifoPort.Consume, 1)
                    OF_4to4.release(ObjectFifoPort.Produce, 1)
                    elemInTmp = OF_4to4.acquire(ObjectFifoPort.Consume, 1)
                    elemOut = OF_4to5.acquire(ObjectFifoPort.Produce, 1)
                    thresholdLine(
                        elemInTmp,
                        elemOut,
                        lineWidth,
                        thresholdValueLower1,
                        thresholdMaxvalue,
                        thresholdModeBinary,
                    )
                    OF_4to4.release(ObjectFifoPort.Consume, 1)
                    OF_4to5.release(ObjectFifoPort.Produce, 1)

            # Compute tile 5
            @core(ComputeTile5, "combined_bitwiseOR_gray2rgba_bitwiseAND.a")
            def coreBody():
                for _ in range_(sys.maxsize):
                    # bitwise OR
                    elemIn1 = OF_3to5.acquire(ObjectFifoPort.Consume, 1)
                    elemIn2 = OF_4to5.acquire(ObjectFifoPort.Consume, 1)
                    elemOutTmpA = OF_5to5a.acquire(ObjectFifoPort.Produce, 1)
                    bitwiseORLine(elemIn1, elemIn2, elemOutTmpA, lineWidth)
                    OF_3to5.release(ObjectFifoPort.Consume, 1)
                    OF_4to5.release(ObjectFifoPort.Consume, 1)
                    OF_5to5a.release(ObjectFifoPort.Produce, 1)
                    # gray2rgba
                    elemInTmpA = OF_5to5a.acquire(ObjectFifoPort.Consume, 1)
                    elemOutTmpB = OF_5to5b.acquire(ObjectFifoPort.Produce, 1)
                    gray2rgbaLine(elemInTmpA, elemOutTmpB, lineWidth)
                    OF_5to5a.release(ObjectFifoPort.Consume, 1)
                    OF_5to5b.release(ObjectFifoPort.Produce, 1)
                    # bitwise AND
                    elemInTmpB1 = OF_5to5b.acquire(ObjectFifoPort.Consume, 1)
                    elemInTmpB2 = inOF_L2L1.acquire(ObjectFifoPort.Consume, 1)
                    elemOut = outOF_L1L2.acquire(ObjectFifoPort.Produce, 1)
                    bitwiseANDLine(elemInTmpB1, elemInTmpB2, elemOut, lineWidthInBytes)
                    OF_5to5b.release(ObjectFifoPort.Consume, 1)
                    inOF_L2L1.release(ObjectFifoPort.Consume, 1)
                    outOF_L1L2.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement
            @runtime_sequence(tensor_ty, tensor_16x16_ty, tensor_ty)
            def sequence(I, B, O):
                npu_dma_memcpy_nd(
                    metadata=inOF_L3L2,
                    bd_id=1,
                    mem=I,
                    sizes=[1, 1, 1, height * lineWidthInBytes],
                )
                npu_dma_memcpy_nd(
                    metadata=outOF_L2L3,
                    bd_id=0,
                    mem=O,
                    sizes=[1, 1, 1, height * lineWidthInBytes],
                )
                # outOF_L2L3 will only complete after inOF_L3L2 completes, so we just wait on outOF_L2L3 instead of all
                dma_wait(outOF_L2L3)

    print(ctx.module)


color_detect()
