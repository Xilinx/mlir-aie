#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *

width = 640 #64 // 8
height = 480 #36 // 8
heightMinus1 = 479

lineWidth         = width
lineWidthInBytes  = width*4
lineWidthInInt32s = lineWidthInBytes // 4

enableTrace = False
traceSizeInBytes = 8192
traceSizeInInt32s = traceSizeInBytes // 4

@constructAndPrintInModule
def edge_detect():
    @device(AIEDevice.ipu)
    def deviceBody():
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int16_ty = IntegerType.get_signless(16)
        int32_ty = IntegerType.get_signless(32)

        line_bytes_ty = MemRefType.get((lineWidthInBytes,), uint8_ty)
        line_ty = MemRefType.get((lineWidth,), uint8_ty)
        memRef_3x3_ty = MemRefType.get((3,3,), int16_ty)

        rgba2grayLine   = privateFunc("rgba2grayLine", inputs = [line_bytes_ty, line_ty, int32_ty])
        filter2dLine    = privateFunc("filter2dLine", inputs = [line_ty, line_ty, line_ty, line_ty, int32_ty, memRef_3x3_ty])
        thresholdLine   = privateFunc("thresholdLine", inputs = [line_ty, line_ty, int32_ty, int16_ty, int16_ty, int8_ty])
        gray2rgbaLine   = privateFunc("gray2rgbaLine", inputs = [line_ty, line_bytes_ty, int32_ty])
        addWeightedLine = privateFunc("addWeightedLine", inputs = [line_bytes_ty, line_bytes_ty, line_bytes_ty, int32_ty, int16_ty, int16_ty, int8_ty])
    
        ShimTile = Tile(0, 0)
        MemTile = Tile(0, 1)
        ComputeTile2 = Tile(0, 2)
        ComputeTile3 = Tile(0, 3)
        ComputeTile4 = Tile(0, 4)
        ComputeTile5 = Tile(0, 5)

        # OrderedObjectBuffer("inOF_L3L2", ShimTile, MemTile, 2, line_bytes_ty)
        # OrderedObjectBuffer("inOF_L2L1", MemTile, [ComputeTile2, ComputeTile5], [2, 2, 7], line_bytes_ty)
        OrderedObjectBuffer("inOF_L3L2", ShimTile, [ComputeTile2, MemTile], [2, 2, 7], line_bytes_ty)
        OrderedObjectBuffer("inOF_L2L1", MemTile, [ComputeTile5], 7, line_bytes_ty)
        Link(["inOF_L3L2"], ["inOF_L2L1"])

        OrderedObjectBuffer("outOF_L2L3", MemTile, ShimTile, 2, line_bytes_ty)
        OrderedObjectBuffer("outOF_L1L2", ComputeTile5, MemTile, 2, line_bytes_ty)
        Link(["outOF_L1L2"], ["outOF_L2L3"])

        OrderedObjectBuffer("OF_2to3", ComputeTile2, ComputeTile3, 4, line_ty)
        OrderedObjectBuffer("OF_3to4", ComputeTile3, ComputeTile4, 2, line_ty)
        OrderedObjectBuffer("OF_4to5", ComputeTile4, ComputeTile5, 2, line_ty)
        OrderedObjectBuffer("OF_5to5", ComputeTile5, ComputeTile5, 1, line_bytes_ty)

        @core(ComputeTile2, "rgba2gray.cc.o")
        def coreBody():
            @forLoop(lowerBound = 0, upperBound = 4294967295, step = 1)
            def loopBody():
                elemIn = Acquire("inOF_L3L2", ObjectFifoPort.Consume, 1, line_bytes_ty).acquiredElem()
                elemOut = Acquire("OF_2to3", ObjectFifoPort.Produce, 1, line_ty).acquiredElem()

                Call(rgba2grayLine, [elemIn, elemOut, lineWidth])

                Release(ObjectFifoPort.Consume, "inOF_L3L2", 1)
                Release(ObjectFifoPort.Produce, "OF_2to3", 1)

        @core(ComputeTile3, "filter2d.cc.o")
        def coreBody():  
            kernel = memref.AllocOp(memRef_3x3_ty, [], [])
            v0 = integerConstant(0, int16_ty)
            v1 = integerConstant(4096, int16_ty)
            vMinus4 = integerConstant(-16384, int16_ty)
            Store(v0, kernel, [0, 0])
            Store(v1, kernel, [0, 1])
            Store(v0, kernel, [0, 2])
            Store(v1, kernel, [1, 0])
            Store(vMinus4, kernel, [1, 1])
            Store(v1, kernel, [1, 2])
            Store(v0, kernel, [2, 0])
            Store(v1, kernel, [2, 1])
            Store(v0, kernel, [2, 2])
            
            @forLoop(lowerBound = 0, upperBound = 4294967295, step = 1)
            def loopBody():

                # Preamble : Top Border
                elemsInPre = Acquire("OF_2to3", ObjectFifoPort.Consume, 2, line_ty).acquiredElem()
                elemPreOut = Acquire("OF_3to4", ObjectFifoPort.Produce, 1, line_ty).acquiredElem()
                Call(filter2dLine, [elemsInPre[0], elemsInPre[0], elemsInPre[1], elemPreOut, lineWidth, kernel])
                Release(ObjectFifoPort.Produce, "OF_3to4", 1) 

                # Steady State : Middle
                @forLoop(lowerBound = 1, upperBound = heightMinus1, step = 1)
                def loopBody():
                    elemsIn = Acquire("OF_2to3", ObjectFifoPort.Consume, 3, line_ty).acquiredElem()
                    elemOut = Acquire("OF_3to4", ObjectFifoPort.Produce, 1, line_ty).acquiredElem()
                    Call(filter2dLine, [elemsIn[0], elemsIn[1], elemsIn[2], elemOut, lineWidth, kernel])
                    Release(ObjectFifoPort.Consume, "OF_2to3", 1)
                    Release(ObjectFifoPort.Produce, "OF_3to4", 1)

                # Postamble : Bottom Border
                elemsInPost = Acquire("OF_2to3", ObjectFifoPort.Consume, 2, line_ty).acquiredElem()
                elemPostOut = Acquire("OF_3to4", ObjectFifoPort.Produce, 1, line_ty).acquiredElem()
                Call(filter2dLine, [elemsInPost[0], elemsInPost[1], elemsInPost[1], elemPostOut, lineWidth, kernel])
                Release(ObjectFifoPort.Consume, "OF_2to3", 2)
                Release(ObjectFifoPort.Produce, "OF_3to4", 1) 

        @core(ComputeTile4, "threshold.cc.o")
        def coreBody():  
            vThr = integerConstant(10, int16_ty)
            vMax = integerConstant(255, int16_ty)
            vTyp = integerConstant(0, int8_ty)
            @forLoop(lowerBound = 0, upperBound = 4294967295, step = 1)
            def loopBody():
                elemIn = Acquire("OF_3to4", ObjectFifoPort.Consume, 1, line_ty).acquiredElem()
                elemOut = Acquire("OF_4to5", ObjectFifoPort.Produce, 1, line_ty).acquiredElem()

                Call(thresholdLine, [elemIn, elemOut, lineWidth, vThr, vMax, vTyp])

                Release(ObjectFifoPort.Consume, "OF_3to4", 1)
                Release(ObjectFifoPort.Produce, "OF_4to5", 1)

        @core(ComputeTile5, "combined_gray2rgba_addWeighted.a")
        def coreBody():
            @forLoop(lowerBound = 0, upperBound = 4294967295, step = 1)
            def loopBody():
                elemIn = Acquire("OF_4to5", ObjectFifoPort.Consume, 1, line_ty).acquiredElem()
                elemOut = Acquire("OF_5to5", ObjectFifoPort.Produce, 1, line_bytes_ty).acquiredElem()

                Call(gray2rgbaLine, [elemIn, elemOut, lineWidth])

                Release(ObjectFifoPort.Consume, "OF_4to5", 1)
                Release(ObjectFifoPort.Produce, "OF_5to5", 1)

                elemIn1 = Acquire("OF_5to5", ObjectFifoPort.Consume, 1, line_bytes_ty).acquiredElem()
                elemIn2 = Acquire("inOF_L2L1", ObjectFifoPort.Consume, 1, line_bytes_ty).acquiredElem()
                elemOut2 = Acquire("outOF_L1L2", ObjectFifoPort.Produce, 1, line_bytes_ty).acquiredElem()

                alpha = integerConstant(16384, int16_ty)
                beta = integerConstant(16384, int16_ty)
                gamma = integerConstant(0, int8_ty)

                Call(addWeightedLine, [elemIn1, elemIn2, elemOut2, lineWidthInBytes, alpha, beta, gamma])

                Release(ObjectFifoPort.Consume, "OF_5to5", 1)
                Release(ObjectFifoPort.Consume, "inOF_L2L1", 1)
                Release(ObjectFifoPort.Produce, "outOF_L1L2", 1)

        tensorSize = width*height*4 # 4 channels
        tensorSizeInInt32s = tensorSize // 4
        # memRef_mem_ty =  MemRefType.get((2304,), int32_ty)
        tensor_ty =  MemRefType.get((tensorSizeInInt32s,), int32_ty)
        memRef_16x16_ty = MemRefType.get((16,16,), int32_ty)
        @FuncOp.from_py_func(tensor_ty, memRef_16x16_ty, tensor_ty)
        def sequence(inTensor, notUsed, outTensor):
            IpuDmaMemcpyNd(metadata = "inOF_L3L2", bd_id = 1, mem = inTensor, lengths = [1, 1, 1, tensorSizeInInt32s]) 
            IpuDmaMemcpyNd(metadata = "outOF_L2L3", bd_id = 0, mem = outTensor, lengths = [1, 1, 1, tensorSizeInInt32s]) 
            IpuSync(column = 0, row = 0, direction = 0, channel = 0)
