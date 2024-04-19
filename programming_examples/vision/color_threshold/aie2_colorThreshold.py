#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx

width = 512
height = 9
if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])

lineWidth = width
lineWidthChannels = width * 4  # 4 channels

enableTrace = False
traceSizeInBytes = 8192
traceSizeInInt32s = traceSizeInBytes // 4


def color_threshold():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu)
        def device_body():
            line_channels_ty = T.memref(lineWidthChannels, T.ui8())
            line_ty = T.memref(lineWidth, T.ui8())

            # AIE Core Function declarations
            thresholdLine = external_func(
                "thresholdLine",
                inputs=[line_ty, line_ty, T.i32(), T.i16(), T.i16(), T.i8()],
            )

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)
            ComputeTile3 = tile(0, 3)
            ComputeTile4 = tile(0, 4)
            ComputeTile5 = tile(0, 5)

            # AIE-array data movement with object fifos

            # Input RGBA broadcast + memtile for skip
            inOOB_L3L2 = object_fifo(
                "inOOB_L3L2", ShimTile, MemTile, 2, line_channels_ty
            )
            inOOB_L2L1_0 = object_fifo(
                "inOOB_L2L1_0", MemTile, ComputeTile2, 2, line_ty
            )
            inOOB_L2L1_1 = object_fifo(
                "inOOB_L2L1_1", MemTile, ComputeTile3, 2, line_ty
            )
            inOOB_L2L1_2 = object_fifo(
                "inOOB_L2L1_2", MemTile, ComputeTile4, 2, line_ty
            )
            inOOB_L2L1_3 = object_fifo(
                "inOOB_L2L1_3", MemTile, ComputeTile5, 2, line_ty
            )
            object_fifo_link(
                inOOB_L3L2,
                [inOOB_L2L1_0, inOOB_L2L1_1, inOOB_L2L1_2, inOOB_L2L1_3],
            )

            # Output RGBA
            outOOB_L2L3 = object_fifo(
                "outOOB_L2L3", MemTile, ShimTile, 2, line_channels_ty
            )
            outOOB_L1L2_0 = object_fifo(
                "outOOB_L1L2_0", ComputeTile2, MemTile, 2, line_ty
            )
            outOOB_L1L2_1 = object_fifo(
                "outOOB_L1L2_1", ComputeTile3, MemTile, 2, line_ty
            )
            outOOB_L1L2_2 = object_fifo(
                "outOOB_L1L2_2", ComputeTile4, MemTile, 2, line_ty
            )
            outOOB_L1L2_3 = object_fifo(
                "outOOB_L1L2_3", ComputeTile5, MemTile, 2, line_ty
            )
            object_fifo_link(
                [outOOB_L1L2_0, outOOB_L1L2_1, outOOB_L1L2_2, outOOB_L1L2_3],
                outOOB_L2L3,
            )

            # Runtime parameters
            rtpComputeTile2 = Buffer(ComputeTile2, [16], T.i32(), "rtpComputeTile2")
            rtpComputeTile3 = Buffer(ComputeTile3, [16], T.i32(), "rtpComputeTile3")
            rtpComputeTile4 = Buffer(ComputeTile4, [16], T.i32(), "rtpComputeTile4")
            rtpComputeTile5 = Buffer(ComputeTile5, [16], T.i32(), "rtpComputeTile5")

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2, "threshold.cc.o")
            def core_body():
                # for _ in for_(4096):
                for _ in for_(sys.maxsize):
                    elemIn = inOOB_L2L1_0.acquire(ObjectFifoPort.Consume, 1)
                    elemOut = outOOB_L1L2_0.acquire(ObjectFifoPort.Produce, 1)

                    # RTPs written from the instruction stream must be read right before the kernel
                    # after the ObjectFIFO acquires
                    thresholdValue = arith.trunci(
                        T.i16(), memref.load(rtpComputeTile2, [0])
                    )
                    maxValue = arith.trunci(T.i16(), memref.load(rtpComputeTile2, [1]))
                    thresholdType = arith.trunci(
                        T.i8(), memref.load(rtpComputeTile2, [2])
                    )
                    # maxValue = arith.constant(255, T.i16())
                    # thresholdValue = arith.constant(50, T.i16())
                    # thresholdType = arith.constant(0, T.i8())
                    call(
                        thresholdLine,
                        [
                            elemIn,
                            elemOut,
                            arith.constant(lineWidth),
                            thresholdValue,
                            maxValue,
                            thresholdType,
                        ],
                    )

                    inOOB_L2L1_0.release(ObjectFifoPort.Consume, 1)
                    outOOB_L1L2_0.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # Compute tile 3
            @core(ComputeTile3, "threshold.cc.o")
            def core_body():
                # for _ in for_(4096):
                for _ in for_(sys.maxsize):
                    elemIn = inOOB_L2L1_1.acquire(ObjectFifoPort.Consume, 1)
                    elemOut = outOOB_L1L2_1.acquire(ObjectFifoPort.Produce, 1)
                    # RTPs written from the instruction stream must be read right before the kernel
                    # after the ObjectFIFO acquires
                    thresholdValue = arith.trunci(
                        T.i16(), memref.load(rtpComputeTile3, [0])
                    )
                    maxValue = arith.trunci(T.i16(), memref.load(rtpComputeTile3, [1]))
                    thresholdType = arith.trunci(
                        T.i8(), memref.load(rtpComputeTile3, [2])
                    )
                    # maxValue = arith.constant(255, T.i16())
                    # thresholdValue = arith.constant(50, T.i16())
                    # thresholdType = arith.constant(0, T.i8())
                    call(
                        thresholdLine,
                        [
                            elemIn,
                            elemOut,
                            arith.constant(lineWidth),
                            thresholdValue,
                            maxValue,
                            thresholdType,
                        ],
                    )

                    inOOB_L2L1_1.release(ObjectFifoPort.Consume, 1)
                    outOOB_L1L2_1.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # Compute tile 4
            @core(ComputeTile4, "threshold.cc.o")
            def core_body():
                # for _ in for_(4096):
                for _ in for_(sys.maxsize):
                    elemIn = inOOB_L2L1_2.acquire(ObjectFifoPort.Consume, 1)
                    elemOut = outOOB_L1L2_2.acquire(ObjectFifoPort.Produce, 1)

                    # RTPs written from the instruction stream must be read right before the kernel
                    # after the ObjectFIFO acquires
                    thresholdValue = arith.trunci(
                        T.i16(), memref.load(rtpComputeTile4, [0])
                    )
                    maxValue = arith.trunci(T.i16(), memref.load(rtpComputeTile4, [1]))
                    thresholdType = arith.trunci(
                        T.i8(), memref.load(rtpComputeTile4, [2])
                    )
                    # maxValue = arith.constant(255, T.i16())
                    # thresholdValue = arith.constant(50, T.i16())
                    # thresholdType = arith.constant(0, T.i8())
                    call(
                        thresholdLine,
                        [
                            elemIn,
                            elemOut,
                            arith.constant(lineWidth),
                            thresholdValue,
                            maxValue,
                            thresholdType,
                        ],
                    )

                    inOOB_L2L1_2.release(ObjectFifoPort.Consume, 1)
                    outOOB_L1L2_2.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # Compute tile 5
            @core(ComputeTile5, "threshold.cc.o")
            def core_body():
                # for _ in for_(4096):
                for _ in for_(sys.maxsize):
                    elemIn = inOOB_L2L1_3.acquire(ObjectFifoPort.Consume, 1)
                    elemOut = outOOB_L1L2_3.acquire(ObjectFifoPort.Produce, 1)

                    # RTPs written from the instruction stream must be read right before the kernel
                    # after the ObjectFIFO acquires
                    thresholdValue = arith.trunci(
                        T.i16(), memref.load(rtpComputeTile5, [0])
                    )
                    maxValue = arith.trunci(T.i16(), memref.load(rtpComputeTile5, [1]))
                    thresholdType = arith.trunci(
                        T.i8(), memref.load(rtpComputeTile5, [2])
                    )
                    # maxValue = arith.constant(255, T.i16())
                    # thresholdValue = arith.constant(50, T.i16())
                    # thresholdType = arith.constant(0, T.i8()
                    call(
                        thresholdLine,
                        [
                            elemIn,
                            elemOut,
                            arith.constant(lineWidth),
                            thresholdValue,
                            maxValue,
                            thresholdType,
                        ],
                    )

                    inOOB_L2L1_3.release(ObjectFifoPort.Consume, 1)
                    outOOB_L1L2_3.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # To/from AIE-array data movement

            tensorSize = width * height
            tensorSizeInInt32s = tensorSize // 4

            @FuncOp.from_py_func(
                T.memref(tensorSizeInInt32s, T.i32()),
                T.memref(32, T.i32()),  # not used
                T.memref(tensorSizeInInt32s, T.i32()),
            )
            def sequence(inTensor, notUsed, outTensor):
                # thresholdValue, maxValue, thresholdType
                NpuWriteRTPOp("rtpComputeTile2", col=0, row=2, index=0, value=50)
                NpuWriteRTPOp("rtpComputeTile2", col=0, row=2, index=1, value=255)
                NpuWriteRTPOp("rtpComputeTile2", col=0, row=2, index=2, value=0)

                NpuWriteRTPOp("rtpComputeTile3", col=0, row=3, index=0, value=50)
                NpuWriteRTPOp("rtpComputeTile3", col=0, row=3, index=1, value=255)
                NpuWriteRTPOp("rtpComputeTile3", col=0, row=3, index=2, value=0)

                NpuWriteRTPOp("rtpComputeTile4", col=0, row=4, index=0, value=50)
                NpuWriteRTPOp("rtpComputeTile4", col=0, row=4, index=1, value=255)
                NpuWriteRTPOp("rtpComputeTile4", col=0, row=4, index=2, value=0)

                NpuWriteRTPOp("rtpComputeTile5", col=0, row=5, index=0, value=50)
                NpuWriteRTPOp("rtpComputeTile5", col=0, row=5, index=1, value=255)
                NpuWriteRTPOp("rtpComputeTile5", col=0, row=5, index=2, value=0)

                npu_dma_memcpy_nd(
                    metadata="inOOB_L3L2",
                    bd_id=1,
                    mem=inTensor,
                    sizes=[1, 1, 1, tensorSizeInInt32s],
                )
                npu_dma_memcpy_nd(
                    metadata="outOOB_L2L3",
                    bd_id=0,
                    mem=outTensor,
                    sizes=[1, 1, 1, tensorSizeInInt32s],
                )
                npu_sync(column=0, row=0, direction=0, channel=0)

    # print(ctx.module.operation.verify())
    print(ctx.module)


color_threshold()
