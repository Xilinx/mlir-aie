#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

import sys

from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.extras import memref, arith
from aie.util import mlir_mod_ctx

width = 128
if len(sys.argv) == 2:
    width = int(sys.argv[1])

lineWidth         = width
lineWidthInBytes  = width*4
lineWidthInInt32s = lineWidthInBytes // 4

enableTrace = False
traceSizeInBytes = 8192
traceSizeInInt32s = traceSizeInBytes // 4

def color_threshold():

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():
            # uint8_ty = IntegerType.get_unsigned(8)
            # int8_ty = IntegerType.get_signless(8)
            # int32_ty = IntegerType.get_signless(32)

            # line_bytes_ty = MemRefType.get((lineWidthInBytes,), uint8_ty)
            # line_ty = MemRefType.get((lineWidth,), uint8_ty)

            thresholdLine = external_func(
                "thresholdLine",
                inputs=[
                    T.memref(lineWidth, T.ui8()),
                    T.memref(lineWidth, T.ui8()),
                    T.i32(),
                    T.i16(),
                    T.i16(),
                    T.i8(),
                ],
            )

            # thresholdLine = privateFunc("thresholdLine", inputs = [lineWidth, lineWidth, int32_ty, int32_ty, int32_ty, int8_ty])
        
            ShimTile     = tile(0, 0)
            MemTile      = tile(0, 1)
            ComputeTile2 = tile(0, 2)
            ComputeTile3 = tile(0, 3)
            ComputeTile4 = tile(0, 4)
            ComputeTile5 = tile(0, 5)

            # set up AIE-array data movement with Ordered Object Buffers
            line_bytes_ty = TypeAttr.get(ObjectFifoType.get(T.memref(lineWidthInBytes, T.ui8())))
            line_ty       = TypeAttr.get(ObjectFifoType.get(T.memref(lineWidth, T.ui8())))

            # input RGBA broadcast + memtile for skip
            objectfifo("inOOB_L3L2", ShimTile, [MemTile], 2, line_bytes_ty, [], [])
            objectfifo("inOOB_L2L1_0", MemTile, [ComputeTile2], 2, line_ty, [], [])
            objectfifo("inOOB_L2L1_1", MemTile, [ComputeTile3], 2, line_ty, [], [])
            objectfifo("inOOB_L2L1_2", MemTile, [ComputeTile4], 2, line_ty, [], [])
            objectfifo("inOOB_L2L1_3", MemTile, [ComputeTile5], 2, line_ty, [], [])
            objectfifo_link(["inOOB_L3L2"], ["inOOB_L2L1_0", "inOOB_L2L1_1", "inOOB_L2L1_2", "inOOB_L2L1_3"])

            # output RGBA 
            objectfifo("outOOB_L2L3", MemTile, [ShimTile], 2, line_bytes_ty, [], [])
            objectfifo("outOOB_L1L2_0", ComputeTile2, [MemTile], 2, line_ty, [], [])
            objectfifo("outOOB_L1L2_1", ComputeTile3, [MemTile], 2, line_ty, [], [])
            objectfifo("outOOB_L1L2_2", ComputeTile4, [MemTile], 2, line_ty, [], [])
            objectfifo("outOOB_L1L2_3", ComputeTile5, [MemTile], 2, line_ty, [], [])
            objectfifo_link(["outOOB_L1L2_0", "outOOB_L1L2_1", "outOOB_L1L2_2", "outOOB_L1L2_3"], ["outOOB_L1L2"])

            # runtime parameters
            rtpComputeTile2 = Buffer(ComputeTile2, [16], T.i32(), "rtpComputeTile2")
            rtpComputeTile3 = Buffer(ComputeTile3, [16], T.i32(), "rtpComputeTile3")
            rtpComputeTile4 = Buffer(ComputeTile4, [16], T.i32(), "rtpComputeTile4")
            rtpComputeTile5 = Buffer(ComputeTile5, [16], T.i32(), "rtpComputeTile5")
    
            # set up compute tiles
            
            #compute tile 2
            @core(ComputeTile2, "threshold.cc.o")
            def core_body():
                maxValue = 255
                thresholdValue = memref.load(rtpComputeTile2, [0])
                thresholdType = memref.load(rtpComputeTile2, [1])
                # @forLoop(lowerBound = 0, upperBound = 4096, step = 1)
                for _ in for_(4096):
                # def loopBody():
                    elemIn  = acquire(
                        ObjectFifoPort.Consume, "inOOB_L2L1_0", 1, T.memref(lineWidth, T.ui8())
                    ).acquired_elem()
                    elemOut = acquire(
                        ObjectFifoPort.Produce, "outOOB_L1L2_0", 1, T.memref(lineWidth, T.ui8())
                    ).acquired_elem()

                    Call(thresholdLine, [elemIn, elemOut, lineWidth, thresholdValue, maxValue, thresholdType])

                    objectfifo_release(ObjectFifoPort.Consume, "inOOB_L2L1_0", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "outOOB_L1L2_0", 1)

            #compute tile 3
            @core(ComputeTile3, "threshold.cc.o")
            def core_body():
                maxValue = 255
                thresholdValue = memref.load(rtpComputeTile3, [0])
                thresholdType = memref.load(rtpComputeTile3, [1])
                # @forLoop(lowerBound = 0, upperBound = 4096, step = 1)
                for _ in for_(4096):
                # def loopBody():
                    elemIn  = acquire(
                        ObjectFifoPort.Consume, "inOOB_L2L1_1", 1, T.memref(lineWidth, T.ui8())
                    ).acquired_elem()
                    elemOut = acquire(
                        ObjectFifoPort.Produce, "outOOB_L1L2_1", 1, T.memref(lineWidth, T.ui8())
                    ).acquired_elem()

                    Call(thresholdLine, [elemIn, elemOut, lineWidth, thresholdValue, maxValue, thresholdType])

                    objectfifo_release(ObjectFifoPort.Consume, "inOOB_L2L1_1", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "outOOB_L1L2_1", 1)

            #compute tile 4
            @core(ComputeTile4, "threshold.cc.o")
            def core_body():
                maxValue = 255
                thresholdValue = memref.load(rtpComputeTile4, [0])
                thresholdType = memref.load(rtpComputeTile4, [1])
                # @forLoop(lowerBound = 0, upperBound = 4096, step = 1)
                for _ in for_(4096):
                # def loopBody():
                    elemIn  = acquire(
                        ObjectFifoPort.Consume, "inOOB_L2L1_2", 1, T.memref(lineWidth, T.ui8())
                    ).acquired_elem()
                    elemOut = acquire(
                        ObjectFifoPort.Produce, "outOOB_L1L2_2", 1, T.memref(lineWidth, T.ui8())
                    ).acquired_elem()

                    Call(thresholdLine, [elemIn, elemOut, lineWidth, thresholdValue, maxValue, thresholdType])

                    objectfifo_release(ObjectFifoPort.Consume, "inOOB_L2L1_2", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "outOOB_L1L2_2", 1)

            #compute tile 5
            @core(ComputeTile5, "threshold.cc.o")
            def core_body():
                maxValue = 255
                thresholdValue = memref.load(rtpComputeTile5, [0])
                thresholdType = memref.load(rtpComputeTile5, [1])
                # @forLoop(lowerBound = 0, upperBound = 4096, step = 1)
                for _ in for_(4096):
                # def loopBody():
                    elemIn  = acquire(
                        ObjectFifoPort.Consume, "inOOB_L2L1_3", 1, T.memref(lineWidth, T.ui8())
                    ).acquired_elem()
                    elemOut = acquire(
                        ObjectFifoPort.Produce, "outOOB_L1L2_3", 1, T.memref(lineWidth, T.ui8())
                    ).acquired_elem()

                    Call(thresholdLine, [elemIn, elemOut, lineWidth, thresholdValue, maxValue, thresholdType])

                    objectfifo_release(ObjectFifoPort.Consume, "inOOB_L2L1_3", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "outOOB_L1L2_3", 1)

            
            # to/from AIE-array data movement
            
            tensorSize = width*4 # 4 channels
            tensorSizeInInt32s = tensorSize // 4
            # tensor_ty =  MemRefType.get((tensorSizeInInt32s,), int32_ty)
            # memRef_16x16_ty = MemRefType.get((16,16,), int32_ty)
            @FuncOp.from_py_func(
                # tensor_ty, 
                # memRef_16x16_ty, 
                # tensor_ty
                T.memref(tensorSizeInInt32s, T.i32()),
                T.memref(16,16, T.i32()),
                T.memref(tensorSizeInInt32s, T.i32()),
            )
            def sequence(inTensor, notUsed, outTensor):

                IpuWriteRTPOp("rtpComputeTile2", col = 0, row = 2, index = 0, value = 50) # thresholdValue
                IpuWriteRTPOp("rtpComputeTile2", col = 0, row = 2, index = 1, value = 0)  # thresholdType
                IpuWriteRTPOp("rtpComputeTile3", col = 0, row = 3, index = 0, value = 50) # thresholdValue
                IpuWriteRTPOp("rtpComputeTile3", col = 0, row = 3, index = 1, value = 0)  # thresholdType
                IpuWriteRTPOp("rtpComputeTile4", col = 0, row = 4, index = 0, value = 50) # thresholdValue
                IpuWriteRTPOp("rtpComputeTile4", col = 0, row = 4, index = 1, value = 0)  # thresholdType
                IpuWriteRTPOp("rtpComputeTile5", col = 0, row = 5, index = 0, value = 50) # thresholdValue
                IpuWriteRTPOp("rtpComputeTile5", col = 0, row = 5, index = 1, value = 0)  # thresholdType

                ipu_dma_memcpy_nd(
                    metadata = "inOOB_L3L2", 
                    bd_id = 0, 
                    mem = inTensor, 
                    lengths = [1, 4, 4, lineWidthInInt32s], 
                    strides = [0, lineWidthInInt32s, lineWidthInBytes]
                ) 
                ipu_dma_memcpy_nd(
                    metadata = "outOOB_L2L3", 
                    bd_id = 1,
                    mem = outTensor, 
                    lengths = [1, 4, 4, lineWidthInInt32s], 
                    strides = [0, lineWidthInInt32s, lineWidthInBytes]
                ) 
                ipu_sync(column = 0, row = 0, direction = 0, channel = 0)

    # print(ctx.module)
    print(ctx.module.operation.verify())

color_threshold()
