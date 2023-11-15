from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *

@constructAndPrintInModule
def edge_detect():
    @device("ipu")
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
                elemIn = Acquire("inOF_L2L1", "Consume", 1, memRef_256_ty).acquiredElem()
                elemOut = Acquire("OF_2to3", "Produce", 1, memRef_64_ty).acquiredElem()

                call(rgba2grayLine, [elemIn, elemOut, integerConstant(64)])

                Release("inOF_L2L1", "Consume", 1)
                Release("OF_2to3", "Produce", 1)

        @core(T3, "filter2d.cc.o")
        def coreBody():  
            kernel = memref.AllocOp(memRef_3x3_ty, [], [])
            v0 = integerConstant(0, int16_ty)
            v1 = integerConstant(4096, int16_ty)
            vMinus4 = integerConstant(-16384, int16_ty)
            c0 = indexConstant(0)     
            c1 = indexConstant(1)   
            c2 = indexConstant(2)
            store(v0, kernel, [c0, c0])
            store(v1, kernel, [c0, c1])
            store(v0, kernel, [c0, c2])
            store(v1, kernel, [c1, c0])
            store(vMinus4, kernel, [c1, c1])
            store(v1, kernel, [c1, c2])
            store(v0, kernel, [c2, c0])
            store(v1, kernel, [c2, c1])
            store(v0, kernel, [c2, c2])
            
            # Preamble : Top Border
            elemsInPre = Acquire("OF_2to3", "Consume", 2, memRef_64_ty).acquiredElem()
            elemPreOut = Acquire("OF_3to4", "Produce", 1, memRef_64_ty).acquiredElem()
            call(filter2dLine, [elemsInPre[0], elemsInPre[0], elemsInPre[1], elemPreOut, integerConstant(64), kernel])
            Release("OF_3to4", "Produce", 1) 

            # Steady State : Middle
            @forLoop(lowerBound = 1, upperBound = 35, step = 1)
            def loopBody():
                elemsIn = Acquire("OF_2to3", "Consume", 3, memRef_64_ty).acquiredElem()
                elemOut = Acquire("OF_3to4", "Produce", 1, memRef_64_ty).acquiredElem()
                call(filter2dLine, [elemsIn[0], elemsIn[1], elemsIn[2], elemOut, integerConstant(64), kernel])
                Release("OF_2to3", "Consume", 1)
                Release("OF_3to4", "Produce", 1)

            # Postamble : Bottom Border
            elemsInPost = Acquire("OF_2to3", "Consume", 2, memRef_64_ty).acquiredElem()
            elemPostOut = Acquire("OF_3to4", "Produce", 1, memRef_64_ty).acquiredElem()
            call(filter2dLine, [elemsInPost[0], elemsInPost[1], elemsInPost[1], elemPostOut, integerConstant(64), kernel])
            Release("OF_2to3", "Consume", 2)
            Release("OF_3to4", "Produce", 1) 

        @core(T4, "threshold.cc.o")
        def coreBody():  
            vThr = integerConstant(10, int16_ty)
            vMax = integerConstant(255, int16_ty)
            vTyp = integerConstant(0, int8_ty)
            @forLoop(lowerBound = 0, upperBound = 36, step = 1)
            def loopBody():
                elemIn = Acquire("OF_3to4", "Consume", 1, memRef_64_ty).acquiredElem()
                elemOut = Acquire("OF_4to5", "Produce", 1, memRef_64_ty).acquiredElem()

                call(thresholdLine, [elemIn, elemOut, integerConstant(64), vThr, vMax, vTyp])

                Release("OF_3to4", "Consume", 1)
                Release("OF_4to5", "Produce", 1)

        @core(T5, "combined_gray2rgba_addWeighted.a")
        def coreBody():
            @forLoop(lowerBound = 0, upperBound = 36, step = 1)
            def loopBody():
                elemIn = Acquire("OF_4to5", "Consume", 1, memRef_64_ty).acquiredElem()
                elemOut = Acquire("OF_5to5", "Produce", 1, memRef_256_ty).acquiredElem()

                call(gray2rgbaLine, [elemIn, elemOut, integerConstant(64)])

                Release("OF_4to5", "Consume", 1)
                Release("OF_5to5", "Produce", 1)

                elemIn1 = Acquire("OF_5to5", "Consume", 1, memRef_256_ty).acquiredElem()
                elemIn2 = Acquire("inOF_L2L1", "Consume", 1, memRef_256_ty).acquiredElem()
                elemOut2 = Acquire("outOF_L1L2", "Produce", 1, memRef_256_ty).acquiredElem()

                alpha = integerConstant(16384, int16_ty)
                beta = integerConstant(16384, int16_ty)
                gamma = integerConstant(0, int8_ty)

                call(addWeightedLine, [elemIn1, elemIn2, elemOut2, integerConstant(256), alpha, beta, gamma])

                Release("OF_5to5", "Consume", 1)
                Release("inOF_L2L1", "Consume", 1)
                Release("outOF_L1L2", "Produce", 1)

        memRef_mem_ty =  MemRefType.get((2304,), int32_ty)
        @FuncOp.from_py_func(memRef_mem_ty, memRef_mem_ty, memRef_mem_ty)
        def sequence(I, B, O):
            IpuDmaMemcpyNd(metadata = "outOF_L2L3", bd_id = 0, mem = O, lengths = [1, 1, 36, 64], strides = [0, 0, 64]) 
            IpuDmaMemcpyNd(metadata = "inOF_L3L2", bd_id = 1, mem = I, lengths = [1, 1, 36, 64], strides = [0, 0, 64]) 
            IpuSync(column = 0, row = 0, direction = 0, channel = 0)
