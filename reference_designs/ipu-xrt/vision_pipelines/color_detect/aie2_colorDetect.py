#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Xilinx Inc.

import sys

from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *

width = 64
height = 36
if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])

lineWidth         = width
lineWidthInBytes  = width*4
lineWidthInInt32s = lineWidthInBytes // 4

enableTrace = False
traceSizeInBytes = 8192
traceSizeInInt32s = traceSizeInBytes // 4

@constructAndPrintInModule
def color_detect():
    @device(AIEDevice.ipu)
    def deviceBody():
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int16_ty = IntegerType.get_signless(16)
        int32_ty = IntegerType.get_signless(32)

        line_bytes_ty = MemRefType.get((lineWidthInBytes,), uint8_ty)
        line_ty = MemRefType.get((lineWidth,), uint8_ty)

        # AIE Core Function declarations
        rgba2hueLine   = privateFunc("rgba2grayLine", inputs = [line_bytes_ty, line_ty, int32_ty])
        thresholdLine   = privateFunc("thresholdLine", inputs = [line_ty, line_ty, int32_ty, int16_ty, int16_ty, int8_ty])
        bitwiseORLine = privateFunc("bitwiseORLine", inputs = [line_ty, line_ty, line_ty, int32_ty])
        gray2rgbaLine   = privateFunc("gray2rgbaLine", inputs = [line_ty, line_bytes_ty, int32_ty])
        bitwiseANDLine = privateFunc("bitwiseORLine", inputs = [line_bytes_ty, line_bytes_ty, line_bytes_ty, int32_ty])
    
        # tile declarations
        ShimTile = Tile(0, 0)
        MemTile = Tile(0, 1)
        ComputeTile2 = Tile(0, 2)
        ComputeTile3 = Tile(0, 3)
        ComputeTile4 = Tile(0, 4)
        ComputeTile5 = Tile(0, 5)

       # set up AIE-array data movement with Ordered Object Bufferss

        # input RGBA broadcast + memtile for skip
        OrderedObjectBuffer("inOOB_L3L2", ShimTile, [ComputeTile2, MemTile], [2, 2, 6], line_bytes_ty)
        OrderedObjectBuffer("inOOB_L2L1", MemTile, [ComputeTile5], 6, line_bytes_ty)
        Link(["inOOB_L3L2"], ["inOOB_L2L1"])

        # output RGBA 
        OrderedObjectBuffer("outOOB_L2L3", MemTile, ShimTile, 2, line_bytes_ty)
        OrderedObjectBuffer("outOOB_L1L2", ComputeTile5, MemTile, 2, line_bytes_ty)
        Link(["outOOB_L1L2"], ["outOOB_L2L3"])

        # between computeTiles
        OrderedObjectBuffer("OOB_2to34", ComputeTile2, [ComputeTile3, ComputeTile4], 2, line_ty)
        OrderedObjectBuffer("OOB_3to3", ComputeTile3, ComputeTile3, 1, line_ty)
        OrderedObjectBuffer("OOB_3to5", ComputeTile3, ComputeTile5, 2, line_ty)
        OrderedObjectBuffer("OOB_4to4", ComputeTile4, ComputeTile4, 1, line_ty)
        OrderedObjectBuffer("OOB_4to5", ComputeTile4, ComputeTile5, 2, line_ty)
        OrderedObjectBuffer("OOB_5to5a", ComputeTile5, ComputeTile5, 1, line_ty)
        OrderedObjectBuffer("OOB_5to5b", ComputeTile5, ComputeTile5, 1, line_ty)

        # set up compute tiles
        
        #compute tile 2
        @core(ComputeTile2, "rgba2hue.cc.o")
        def coreBody():
            @forLoop(lowerBound = 0, upperBound = sys.maxsize, step = 1)
            def loopBody():
                elemIn = Acquire(ObjectFifoPort.Consume, "inOOB_L3L2", 1, line_bytes_ty).acquiredElem()
                elemOut = Acquire(ObjectFifoPort.Produce, "OOB_2to34", 1, line_ty).acquiredElem()

                Call(rgba2hueLine, [elemIn, elemOut, lineWidth])

                Release(ObjectFifoPort.Consume, "inOOB_L3L2", 1)
                Release(ObjectFifoPort.Produce, "OOB_2to34", 1)

        #compute tile 3
        @core(ComputeTile3, "threshold.cc.o")
        def coreBody():  
            thresholdValueUpper1 = integerConstant(40, int16_ty)
            thresholdValueLower1 = integerConstant(30, int16_ty)
            thresholdMaxvalue = integerConstant(255, int16_ty)
            thresholdModeToZeroInv = integerConstant(4, int8_ty)
            thresholdModeBinary = integerConstant(0, int8_ty)
            
            @forLoop(lowerBound = 0, upperBound = sys.maxsize, step = 1)
            def loopBody():
                elemIn = Acquire(ObjectFifoPort.Consume, "OOB_2to34",  1, line_ty).acquiredElem()
                elemOutTmp = Acquire(ObjectFifoPort.Produce, "OOB_3to3", 1, line_ty).acquiredElem()

                Call(thresholdLine, [elemIn, elemOutTmp, lineWidth, thresholdValueUpper1, thresholdMaxvalue, thresholdModeToZeroInv])

                Release(ObjectFifoPort.Consume, "OOB_2to34", 1)
                Release(ObjectFifoPort.Produce, "OOB_3to3", 1)

                elemInTmp = Acquire(ObjectFifoPort.Consume, "OOB_3to3",  1, line_ty).acquiredElem()
                elemOut = Acquire(ObjectFifoPort.Produce, "OOB_3to5", 1, line_ty).acquiredElem()

                Call(thresholdLine, [elemInTmp, elemOut, lineWidth, thresholdValueLower1, thresholdMaxvalue, thresholdModeBinary])

                Release(ObjectFifoPort.Consume, "OOB_3to3", 1)
                Release(ObjectFifoPort.Produce, "OOB_3to5", 1)


        #compute tile 4
        @core(ComputeTile4, "threshold.cc.o")
        def coreBody():  
            thresholdValueUpper1 = integerConstant(160, int16_ty)
            thresholdValueLower1 = integerConstant(90, int16_ty)
            thresholdMaxvalue = integerConstant(255, int16_ty)
            thresholdModeToZeroInv = integerConstant(4, int8_ty)
            thresholdModeBinary = integerConstant(0, int8_ty)
            
            @forLoop(lowerBound = 0, upperBound = sys.maxsize, step = 1)
            def loopBody():
                elemIn = Acquire(ObjectFifoPort.Consume, "OOB_2to34",  1, line_ty).acquiredElem()
                elemOutTmp = Acquire(ObjectFifoPort.Produce, "OOB_4to4", 1, line_ty).acquiredElem()

                Call(thresholdLine, [elemIn, elemOutTmp, lineWidth, thresholdValueUpper1, thresholdMaxvalue, thresholdModeToZeroInv])

                Release(ObjectFifoPort.Consume, "OOB_2to34", 1)
                Release(ObjectFifoPort.Produce, "OOB_4to4", 1)

                elemInTmp = Acquire(ObjectFifoPort.Consume, "OOB_4to4",  1, line_ty).acquiredElem()
                elemOut = Acquire(ObjectFifoPort.Produce, "OOB_4to5", 1, line_ty).acquiredElem()

                Call(thresholdLine, [elemInTmp, elemOut, lineWidth, thresholdValueLower1, thresholdMaxvalue, thresholdModeBinary])

                Release(ObjectFifoPort.Consume, "OOB_4to4", 1)
                Release(ObjectFifoPort.Produce, "OOB_4to5", 1)

        #compute tile 5
        @core(ComputeTile5, "combined_bitwiseOR_gray2rgba_bitwiseAND.a")
        def coreBody():
            @forLoop(lowerBound = 0, upperBound = sys.maxsize, step = 1)
            def loopBody():
                # bitwise OR
                elemIn1 = Acquire(ObjectFifoPort.Consume, "OOB_3to5", 1, line_ty).acquiredElem()
                elemIn2 = Acquire(ObjectFifoPort.Consume, "OOB_4to5", 1, line_ty).acquiredElem()
                elemOutTmpA = Acquire(ObjectFifoPort.Produce, "OOB_5to5a", 1, line_ty).acquiredElem()

                Call(bitwiseORLine, [elemIn1, elemIn2, elemOutTmpA, lineWidth])

                Release(ObjectFifoPort.Consume, "OOB_3to5", 1)
                Release(ObjectFifoPort.Consume, "OOB_4to5", 1)
                Release(ObjectFifoPort.Produce, "OOB_5to5a", 1)

                # gray2rgba
                elemInTmpA = Acquire(ObjectFifoPort.Consume, "OOB_5to5a", 1, line_ty).acquiredElem()
                elemOutTmpB = Acquire(ObjectFifoPort.Produce, "OOB_5to5b", 1, line_bytes_ty).acquiredElem()
                
                Call(gray2rgbaLine, [elemInTmpA, elemOutTmpB, lineWidth])
                
                Release(ObjectFifoPort.Consume, "OOB_5to5a", 1)
                Release(ObjectFifoPort.Produce, "OOB_5to5b", 1)

                # # bitwise AND
                elemInTmpB1 = Acquire(ObjectFifoPort.Consume, "OOB_5to5b", 1, line_bytes_ty).acquiredElem()
                elemInTmpB2 = Acquire(ObjectFifoPort.Consume, "inOOB_L2L1", 1, line_bytes_ty).acquiredElem()
                elemOut = Acquire(ObjectFifoPort.Produce, "outOOB_L1L2", 1, line_bytes_ty).acquiredElem()

                Call(bitwiseANDLine, [elemInTmpB1, elemInTmpB2, elemOut, lineWidthInBytes])

                Release(ObjectFifoPort.Consume, "OOB_5to5b", 1)
                Release(ObjectFifoPort.Consume, "inOOB_L2L1", 1)
                Release(ObjectFifoPort.Produce, "outOOB_L1L2", 1)

        
        # to/from AIE-array data movement
        
        tensorSize = width*height*4 # 4 channels
        tensorSizeInInt32s = tensorSize // 4
        tensor_ty =  MemRefType.get((tensorSizeInInt32s,), int32_ty)
        memRef_16x16_ty = MemRefType.get((16,16,), int32_ty)
        @FuncOp.from_py_func(tensor_ty, memRef_16x16_ty, tensor_ty)
        def sequence(inTensor, notUsed, outTensor):
            IpuDmaMemcpyNd(metadata = "inOOB_L3L2", bd_id = 1, mem = inTensor, lengths = [1, 1, 1, height * lineWidthInInt32s]) 
            IpuDmaMemcpyNd(metadata = "outOOB_L2L3", bd_id = 0, mem = outTensor, lengths = [1, 1, 1, height * lineWidthInInt32s]) 
            IpuSync(column = 0, row = 0, direction = 0, channel = 0)
