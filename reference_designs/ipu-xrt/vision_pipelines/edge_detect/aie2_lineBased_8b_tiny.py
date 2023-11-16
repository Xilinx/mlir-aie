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

@constructAndPrintInModule
def edge_detect():
    @device(AIEDevice.ipu)
    def deviceBody():
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int16_ty = IntegerType.get_signless(16)
        int32_ty = IntegerType.get_signless(32)
        memRef_256_ty = MemRefType.get((256,), uint8_ty)
        memRef_64_ty = MemRefType.get((64,), uint8_ty)
        memRef_3x3_ty = MemRefType.get((3,3,), int16_ty)

        rgba2grayLine   = privateFunc("rgba2grayLine", inputs = [memRef_256_ty, memRef_64_ty, int32_ty])
        filter2dLine    = privateFunc("filter2dLine", inputs = [memRef_64_ty, memRef_64_ty, memRef_64_ty, memRef_64_ty, int32_ty, memRef_3x3_ty])
        thresholdLine   = privateFunc("thresholdLine", inputs = [memRef_64_ty, memRef_64_ty, int32_ty, int16_ty, int16_ty, int8_ty])
        gray2rgbaLine   = privateFunc("gray2rgbaLine", inputs = [memRef_64_ty, memRef_256_ty, int32_ty])
        addWeightedLine = privateFunc("addWeightedLine", inputs = [memRef_256_ty, memRef_256_ty, memRef_256_ty, int32_ty, int16_ty, int16_ty, int8_ty])
    
        S = Tile(0, 0)
        M = Tile(0, 1)
        T2 = Tile(0, 2)
        T3 = Tile(0, 3)
        T4 = Tile(0, 4)
        T5 = Tile(0, 5)

        OrderedObjectBuffer("inOF_L3L2", S, M, 2, memRef_256_ty)
        OrderedObjectBuffer("inOF_L2L1", M, [T2, T5], [2, 2, 7], memRef_256_ty)
        Link(["inOF_L3L2"], ["inOF_L2L1"])

        OrderedObjectBuffer("outOF_L2L3", M, S, 2, memRef_256_ty)
        OrderedObjectBuffer("outOF_L1L2", T5, M, 2, memRef_256_ty)
        Link(["outOF_L1L2"], ["outOF_L2L3"])

        OrderedObjectBuffer("OF_2to3", T2, T3, 4, memRef_64_ty)
        OrderedObjectBuffer("OF_3to4", T3, T4, 2, memRef_64_ty)
        OrderedObjectBuffer("OF_4to5", T4, T5, 2, memRef_64_ty)
        OrderedObjectBuffer("OF_5to5", T5, T5, 1, memRef_256_ty)

        @core(T2, "rgba2gray.cc.o")
        def coreBody():
            @forLoop(lowerBound = 0, upperBound = 36, step = 1)
            def loopBody():
                elemIn = Acquire(ObjectFifoPort.Consume, "inOF_L2L1", 1, memRef_256_ty).acquiredElem()
                elemOut = Acquire(ObjectFifoPort.Produce, "OF_2to3", 1, memRef_64_ty).acquiredElem()

                Call(rgba2grayLine, [elemIn, elemOut, integerConstant(64)])

                Release(ObjectFifoPort.Consume, "inOF_L2L1", 1)
                Release(ObjectFifoPort.Produce, "OF_2to3", 1)

        @core(T3, "filter2d.cc.o")
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
            
            # Preamble : Top Border
            elemsInPre = Acquire(ObjectFifoPort.Consume, "OF_2to3", 2, memRef_64_ty).acquiredElem()
            elemPreOut = Acquire(ObjectFifoPort.Produce, "OF_3to4", 1, memRef_64_ty).acquiredElem()
            Call(filter2dLine, [elemsInPre[0], elemsInPre[0], elemsInPre[1], elemPreOut, integerConstant(64), kernel])
            Release(ObjectFifoPort.Produce, "OF_3to4", 1) 

            # Steady State : Middle
            @forLoop(lowerBound = 1, upperBound = 35, step = 1)
            def loopBody():
                elemsIn = Acquire(ObjectFifoPort.Consume, "OF_2to3", 3, memRef_64_ty).acquiredElem()
                elemOut = Acquire(ObjectFifoPort.Produce, "OF_3to4", 1, memRef_64_ty).acquiredElem()
                Call(filter2dLine, [elemsIn[0], elemsIn[1], elemsIn[2], elemOut, integerConstant(64), kernel])
                Release(ObjectFifoPort.Consume, "OF_2to3", 1)
                Release(ObjectFifoPort.Produce, "OF_3to4", 1)

            # Postamble : Bottom Border
            elemsInPost = Acquire(ObjectFifoPort.Consume, "OF_2to3", 2, memRef_64_ty).acquiredElem()
            elemPostOut = Acquire(ObjectFifoPort.Produce, "OF_3to4", 1, memRef_64_ty).acquiredElem()
            Call(filter2dLine, [elemsInPost[0], elemsInPost[1], elemsInPost[1], elemPostOut, integerConstant(64), kernel])
            Release(ObjectFifoPort.Consume, "OF_2to3", 2)
            Release(ObjectFifoPort.Produce, "OF_3to4", 1) 

        @core(T4, "threshold.cc.o")
        def coreBody():  
            vThr = integerConstant(10, int16_ty)
            vMax = integerConstant(255, int16_ty)
            vTyp = integerConstant(0, int8_ty)
            @forLoop(lowerBound = 0, upperBound = 36, step = 1)
            def loopBody():
                elemIn = Acquire(ObjectFifoPort.Consume, "OF_3to4", 1, memRef_64_ty).acquiredElem()
                elemOut = Acquire(ObjectFifoPort.Produce, "OF_4to5", 1, memRef_64_ty).acquiredElem()

                Call(thresholdLine, [elemIn, elemOut, integerConstant(64), vThr, vMax, vTyp])

                Release(ObjectFifoPort.Consume, "OF_3to4", 1)
                Release(ObjectFifoPort.Produce, "OF_4to5", 1)

        @core(T5, "combined_gray2rgba_addWeighted.a")
        def coreBody():
            @forLoop(lowerBound = 0, upperBound = 36, step = 1)
            def loopBody():
                elemIn = Acquire(ObjectFifoPort.Consume, "OF_4to5", 1, memRef_64_ty).acquiredElem()
                elemOut = Acquire(ObjectFifoPort.Produce, "OF_5to5", 1, memRef_256_ty).acquiredElem()

                Call(gray2rgbaLine, [elemIn, elemOut, integerConstant(64)])

                Release(ObjectFifoPort.Consume, "OF_4to5", 1)
                Release(ObjectFifoPort.Produce, "OF_5to5", 1)

                elemIn1 = Acquire(ObjectFifoPort.Consume, "OF_5to5", 1, memRef_256_ty).acquiredElem()
                elemIn2 = Acquire(ObjectFifoPort.Consume, "inOF_L2L1", 1, memRef_256_ty).acquiredElem()
                elemOut2 = Acquire(ObjectFifoPort.Produce, "outOF_L1L2", 1, memRef_256_ty).acquiredElem()

                alpha = integerConstant(16384, int16_ty)
                beta = integerConstant(16384, int16_ty)
                gamma = integerConstant(0, int8_ty)

                Call(addWeightedLine, [elemIn1, elemIn2, elemOut2, integerConstant(256), alpha, beta, gamma])

                Release(ObjectFifoPort.Consume, "OF_5to5", 1)
                Release(ObjectFifoPort.Consume, "inOF_L2L1", 1)
                Release(ObjectFifoPort.Produce, "outOF_L1L2", 1)

        memRef_mem_ty =  MemRefType.get((2304,), int32_ty)
        @FuncOp.from_py_func(memRef_mem_ty, memRef_mem_ty, memRef_mem_ty)
        def sequence(I, B, O):
            IpuDmaMemcpyNd(metadata = "outOF_L2L3", bd_id = 0, mem = O, lengths = [1, 1, 36, 64], strides = [0, 0, 64]) 
            IpuDmaMemcpyNd(metadata = "inOF_L3L2", bd_id = 1, mem = I, lengths = [1, 1, 36, 64], strides = [0, 0, 64]) 
            IpuSync(column = 0, row = 0, direction = 0, channel = 0)
