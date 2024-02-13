#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Xilinx Inc.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.dialects.ext import arith
from aie.extras.dialects.ext.scf import range_, yield_
from aie.extras.context import mlir_mod_ctx
from aie.ir import MemRefType, TypeAttr

width = 64
height = 36
if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])

lineWidth = width
lineWidthInBytes = width * 4
lineWidthInInt32s = lineWidthInBytes // 4

enableTrace = False
traceSizeInBytes = 8192
traceSizeInInt32s = traceSizeInBytes // 4


def color_detect():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def deviceBody():
            line_bytes_ty = MemRefType.get((lineWidthInBytes,), T.ui8())
            line_ty = MemRefType.get((lineWidth,), T.ui8())

            ofifo_line_bytes_ty = TypeAttr.get(ObjectFifoType.get(line_bytes_ty))
            ofifo_line_ty = TypeAttr.get(ObjectFifoType.get(line_ty))

            # AIE Core Function declarations
            rgba2hueLine = external_func(
                "rgba2hueLine", inputs=[line_bytes_ty, line_ty, T.i32()]
            )
            thresholdLine = external_func(
                "thresholdLine",
                inputs=[line_ty, line_ty, T.i32(), T.i16(), T.i16(), T.i8()],
            )
            bitwiseORLine = external_func(
                "bitwiseORLine", inputs=[line_ty, line_ty, line_ty, T.i32()]
            )
            gray2rgbaLine = external_func(
                "gray2rgbaLine", inputs=[line_ty, line_bytes_ty, T.i32()]
            )
            bitwiseANDLine = external_func(
                "bitwiseANDLine",
                inputs=[line_bytes_ty, line_bytes_ty, line_bytes_ty, T.i32()],
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
            objectfifo(
                "inOF_L3L2",
                ShimTile,
                [ComputeTile2, MemTile],
                [2, 2, 6],
                ofifo_line_bytes_ty,
                [],
                [],
            )
            objectfifo(
                "inOF_L2L1",
                MemTile,
                [ComputeTile5],
                6,
                ofifo_line_bytes_ty,
                [],
                [],
            )
            objectfifo_link(["inOF_L3L2"], ["inOF_L2L1"])

            # Output
            objectfifo(
                "outOF_L2L3",
                MemTile,
                [ShimTile],
                2,
                ofifo_line_bytes_ty,
                [],
                [],
            )
            objectfifo(
                "outOF_L1L2",
                ComputeTile5,
                [MemTile],
                2,
                ofifo_line_bytes_ty,
                [],
                [],
            )
            objectfifo_link(["outOF_L1L2"], ["outOF_L2L3"])

            # Intermediate
            objectfifo(
                "OF_2to34",
                ComputeTile2,
                [ComputeTile3, ComputeTile4],
                2,
                ofifo_line_ty,
                [],
                [],
            )
            objectfifo(
                "OF_3to3",
                ComputeTile3,
                [ComputeTile3],
                1,
                ofifo_line_ty,
                [],
                [],
            )
            objectfifo(
                "OF_3to5",
                ComputeTile3,
                [ComputeTile5],
                2,
                ofifo_line_ty,
                [],
                [],
            )
            objectfifo(
                "OF_4to4",
                ComputeTile4,
                [ComputeTile4],
                1,
                ofifo_line_ty,
                [],
                [],
            )
            objectfifo(
                "OF_4to5",
                ComputeTile4,
                [ComputeTile5],
                2,
                ofifo_line_ty,
                [],
                [],
            )
            objectfifo(
                "OF_5to5a",
                ComputeTile5,
                [ComputeTile5],
                1,
                ofifo_line_ty,
                [],
                [],
            )
            objectfifo(
                "OF_5to5b",
                ComputeTile5,
                [ComputeTile5],
                1,
                ofifo_line_bytes_ty,
                [],
                [],
            )

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2, "rgba2hue.cc.o")
            def coreBody():
                for _ in range_(sys.maxsize):
                    elemIn = acquire(
                        ObjectFifoPort.Consume, "inOF_L3L2", 1, line_bytes_ty
                    ).acquired_elem()
                    elemOut = acquire(
                        ObjectFifoPort.Produce, "OF_2to34", 1, line_ty
                    ).acquired_elem()
                    Call(rgba2hueLine, [elemIn, elemOut, arith.constant(lineWidth)])
                    objectfifo_release(ObjectFifoPort.Consume, "inOF_L3L2", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_2to34", 1)
                    yield_([])

            # Compute tile 3
            @core(ComputeTile3, "threshold.cc.o")
            def coreBody():
                thresholdValueUpper1 = arith.constant(40, T.i16())
                thresholdValueLower1 = arith.constant(30, T.i16())
                thresholdMaxvalue = arith.constant(255, T.i16())
                thresholdModeToZeroInv = arith.constant(4, T.i8())
                thresholdModeBinary = arith.constant(0, T.i8())
                for _ in range_(sys.maxsize):
                    elemIn = acquire(
                        ObjectFifoPort.Consume, "OF_2to34", 1, line_ty
                    ).acquired_elem()
                    elemOutTmp = acquire(
                        ObjectFifoPort.Produce, "OF_3to3", 1, line_ty
                    ).acquired_elem()
                    Call(
                        thresholdLine,
                        [
                            elemIn,
                            elemOutTmp,
                            arith.constant(lineWidth),
                            thresholdValueUpper1,
                            thresholdMaxvalue,
                            thresholdModeToZeroInv,
                        ],
                    )
                    objectfifo_release(ObjectFifoPort.Consume, "OF_2to34", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_3to3", 1)
                    elemInTmp = acquire(
                        ObjectFifoPort.Consume, "OF_3to3", 1, line_ty
                    ).acquired_elem()
                    elemOut = acquire(
                        ObjectFifoPort.Produce, "OF_3to5", 1, line_ty
                    ).acquired_elem()
                    Call(
                        thresholdLine,
                        [
                            elemInTmp,
                            elemOut,
                            arith.constant(lineWidth),
                            thresholdValueLower1,
                            thresholdMaxvalue,
                            thresholdModeBinary,
                        ],
                    )
                    objectfifo_release(ObjectFifoPort.Consume, "OF_3to3", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_3to5", 1)
                    yield_([])

            # Compute tile 4
            @core(ComputeTile4, "threshold.cc.o")
            def coreBody():
                thresholdValueUpper1 = arith.constant(160, T.i16())
                thresholdValueLower1 = arith.constant(90, T.i16())
                thresholdMaxvalue = arith.constant(255, T.i16())
                thresholdModeToZeroInv = arith.constant(4, T.i8())
                thresholdModeBinary = arith.constant(0, T.i8())
                for _ in range_(sys.maxsize):
                    elemIn = acquire(
                        ObjectFifoPort.Consume, "OF_2to34", 1, line_ty
                    ).acquired_elem()
                    elemOutTmp = acquire(
                        ObjectFifoPort.Produce, "OF_4to4", 1, line_ty
                    ).acquired_elem()
                    Call(
                        thresholdLine,
                        [
                            elemIn,
                            elemOutTmp,
                            arith.constant(lineWidth),
                            thresholdValueUpper1,
                            thresholdMaxvalue,
                            thresholdModeToZeroInv,
                        ],
                    )
                    objectfifo_release(ObjectFifoPort.Consume, "OF_2to34", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_4to4", 1)
                    elemInTmp = acquire(
                        ObjectFifoPort.Consume, "OF_4to4", 1, line_ty
                    ).acquired_elem()
                    elemOut = acquire(
                        ObjectFifoPort.Produce, "OF_4to5", 1, line_ty
                    ).acquired_elem()
                    Call(
                        thresholdLine,
                        [
                            elemInTmp,
                            elemOut,
                            arith.constant(lineWidth),
                            thresholdValueLower1,
                            thresholdMaxvalue,
                            thresholdModeBinary,
                        ],
                    )
                    objectfifo_release(ObjectFifoPort.Consume, "OF_4to4", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_4to5", 1)
                    yield_([])

            # Compute tile 5
            @core(ComputeTile5, "combined_bitwiseOR_gray2rgba_bitwiseAND.a")
            def coreBody():
                for _ in range_(sys.maxsize):
                    # bitwise OR
                    elemIn1 = acquire(
                        ObjectFifoPort.Consume, "OF_3to5", 1, line_ty
                    ).acquired_elem()
                    elemIn2 = acquire(
                        ObjectFifoPort.Consume, "OF_4to5", 1, line_ty
                    ).acquired_elem()
                    elemOutTmpA = acquire(
                        ObjectFifoPort.Produce, "OF_5to5a", 1, line_ty
                    ).acquired_elem()
                    Call(
                        bitwiseORLine,
                        [elemIn1, elemIn2, elemOutTmpA, arith.constant(lineWidth)],
                    )
                    objectfifo_release(ObjectFifoPort.Consume, "OF_3to5", 1)
                    objectfifo_release(ObjectFifoPort.Consume, "OF_4to5", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_5to5a", 1)
                    # gray2rgba
                    elemInTmpA = acquire(
                        ObjectFifoPort.Consume, "OF_5to5a", 1, line_ty
                    ).acquired_elem()
                    elemOutTmpB = acquire(
                        ObjectFifoPort.Produce, "OF_5to5b", 1, line_bytes_ty
                    ).acquired_elem()
                    Call(
                        gray2rgbaLine,
                        [elemInTmpA, elemOutTmpB, arith.constant(lineWidth)],
                    )
                    objectfifo_release(ObjectFifoPort.Consume, "OF_5to5a", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "OF_5to5b", 1)
                    # bitwise AND
                    elemInTmpB1 = acquire(
                        ObjectFifoPort.Consume, "OF_5to5b", 1, line_bytes_ty
                    ).acquired_elem()
                    elemInTmpB2 = acquire(
                        ObjectFifoPort.Consume, "inOF_L2L1", 1, line_bytes_ty
                    ).acquired_elem()
                    elemOut = acquire(
                        ObjectFifoPort.Produce, "outOF_L1L2", 1, line_bytes_ty
                    ).acquired_elem()
                    Call(
                        bitwiseANDLine,
                        [
                            elemInTmpB1,
                            elemInTmpB2,
                            elemOut,
                            arith.constant(lineWidthInBytes),
                        ],
                    )
                    objectfifo_release(ObjectFifoPort.Consume, "OF_5to5b", 1)
                    objectfifo_release(ObjectFifoPort.Consume, "inOF_L2L1", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "outOF_L1L2", 1)
                    yield_([])

            # To/from AIE-array data movement

            tensorSize = width * height * 4  # 4 channels
            tensorSizeInInt32s = tensorSize // 4
            tensor_ty = MemRefType.get((tensorSizeInInt32s,), T.i32())
            memRef_16x16_ty = MemRefType.get(
                (
                    16,
                    16,
                ),
                T.i32(),
            )

            @FuncOp.from_py_func(tensor_ty, memRef_16x16_ty, tensor_ty)
            def sequence(I, B, O):
                ipu_dma_memcpy_nd(
                    metadata="inOF_L3L2",
                    bd_id=1,
                    mem=I,
                    sizes=[1, 1, 1, height * lineWidthInInt32s],
                )
                ipu_dma_memcpy_nd(
                    metadata="outOF_L2L3",
                    bd_id=0,
                    mem=O,
                    sizes=[1, 1, 1, height * lineWidthInInt32s],
                )
                ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


color_detect()
