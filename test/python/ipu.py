# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *


@constructAndPrintInModule
def my_vector_scalar():
    N = 4095
    n = 1023
    N_div_n = N // n
    N_in_bytes = N * 3

    buffer_depth = 1

    @device(AIEDevice.ipu)
    def deviceBody():
        int32_ty = IntegerType.get_signless(32)
        memRef_ty = MemRefType.get((n,), int32_ty)

        scale_int32 = privateFunc("scale_int32", inputs=[memRef_ty, memRef_ty])

        S = Tile(0, 0)
        T = Tile(0, 2)

        OrderedObjectBuffer("in", S, T, buffer_depth, memRef_ty)
        OrderedObjectBuffer("out", T, S, buffer_depth, memRef_ty)

        @core(T, "scale.o")
        def coreBody():
            # Effective while(1)
            @forLoop(lowerBound=0, upperBound=0xFFFFFFFF, step=1)
            def loopReps():
                # Number of sub-vector "tile" iterations
                @forLoop(lowerBound=0, upperBound=N_div_n, step=1)
                def loopTile():
                    elemOut = Acquire(
                        "out", ObjectFifoPort.Produce, 1, memRef_ty
                    ).acquiredElem()
                    elemIn = Acquire(
                        "in", ObjectFifoPort.Consume, 1, memRef_ty
                    ).acquiredElem()
                    Call(scale_int32, [elemIn, elemOut])
                    Release(ObjectFifoPort.Consume, "in", 1)
                    Release(ObjectFifoPort.Produce, "out", 1)

        memRef_mem_ty = MemRefType.get((N,), int32_ty)

        @FuncOp.from_py_func(memRef_mem_ty, memRef_mem_ty, memRef_mem_ty)
        def sequence(A, B, C):
            IpuDmaMemcpyNd(metadata="out", bd_id=0, mem=C, lengths=[1, 1, 1, N])
            IpuDmaMemcpyNd(metadata="in", bd_id=1, mem=A, lengths=[1, 1, 1, N])
            IpuSync(column=0, row=0, direction=0, channel=0)


@constructAndPrintInModule
def my_matmul():
    M = 128
    K = 128
    N = 128
    m = 64
    k = 32
    n = 64
    word_size_in = 2
    word_size_out = 2

    A_sz_in_i32s = M * K * word_size_in // 4
    B_sz_in_i32s = K * N * word_size_in // 4
    C_sz_in_bytes = M * N * word_size_out
    C_sz_in_i32s = C_sz_in_bytes // 4

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    # Matrix A: MxK, submatrices a: mxk
    k_in_i32s = k * word_size_in // 4
    K_in_i32s = K * word_size_in // 4

    # Matrix B: KxN, submatrices b: kxn
    n_in_i32s = n * word_size_in // 4
    N_in_i32s = N * word_size_in // 4
    k_x_N_in_i32s = k * N * word_size_in // 4

    # Output Matrix C: MxN
    n_in_i32s_out = n * word_size_out // 4
    N_in_i32s_out = N * word_size_out // 4
    m_x_N_in_i32s_out = m * N * word_size_out // 4

    vectorized = True

    @device(AIEDevice.ipu)
    def deviceBody():
        in_ty = IntegerType.get_signless(16)
        out_ty = IntegerType.get_signless(16)
        memRef_A_ty = MemRefType.get(
            (
                m,
                k,
            ),
            in_ty,
        )
        memRef_B_ty = MemRefType.get(
            (
                k,
                n,
            ),
            in_ty,
        )
        memRef_C_ty = MemRefType.get(
            (
                m,
                n,
            ),
            out_ty,
        )

        zero_scalar = privateFunc("zero_scalar_i16", inputs=[memRef_C_ty])
        zero = privateFunc("zero_i16", inputs=[memRef_C_ty])
        matmul_scalar = privateFunc(
            "matmul_scalar_i16_i16", inputs=[memRef_A_ty, memRef_B_ty, memRef_C_ty]
        )
        matmul = privateFunc(
            "matmul_i16_i16", inputs=[memRef_A_ty, memRef_B_ty, memRef_C_ty]
        )

        S = Tile(0, 0)
        M = Tile(0, 1)
        T = Tile(0, 2)

        OrderedObjectBuffer("inA", S, T, 2, memRef_A_ty)
        OrderedObjectBuffer("inB", S, T, 2, memRef_B_ty)
        OrderedObjectBuffer("outC", T, S, 2, memRef_C_ty)

        @core(T, "mm.o")
        def coreBody():
            @forLoop(lowerBound=0, upperBound=0xFFFFFFFF, step=1)
            def loopReps():
                @forLoop(lowerBound=0, upperBound=tiles, step=1)
                def loopTile():
                    elemOut = Acquire(
                        "outC", ObjectFifoPort.Produce, 1, memRef_C_ty
                    ).acquiredElem()
                    if vectorized:
                        Call(zero, [elemOut])
                    else:
                        Call(zero_scalar, [elemOut])

                    @forLoop(lowerBound=0, upperBound=K_div_k, step=1)
                    def loopK():
                        elemInA = Acquire(
                            "inA", ObjectFifoPort.Consume, 1, memRef_A_ty
                        ).acquiredElem()
                        elemInB = Acquire(
                            "inB", ObjectFifoPort.Consume, 1, memRef_B_ty
                        ).acquiredElem()
                        if vectorized:
                            Call(matmul, [elemInA, elemInB, elemOut])
                        else:
                            Call(matmul_scalar, [elemInA, elemInB, elemOut])
                        Release(ObjectFifoPort.Consume, "inA", 1)
                        Release(ObjectFifoPort.Consume, "inB", 1)

                    Release(ObjectFifoPort.Produce, "outC", 1)

        int32_ty = IntegerType.get_signless(32)
        memRef_Ain_ty = MemRefType.get((A_sz_in_i32s,), int32_ty)
        memRef_Bin_ty = MemRefType.get((B_sz_in_i32s,), int32_ty)
        memRef_Cout_ty = MemRefType.get((C_sz_in_i32s,), int32_ty)

        @FuncOp.from_py_func(memRef_Ain_ty, memRef_Bin_ty, memRef_Cout_ty)
        def sequence(A, B, C):
            # only do 5 tile rows at a time before synchronizing, so we can reuse BDs
            rows_per_block = 5
            for tile_row_block in range(
                (M_div_m + rows_per_block - 1) // rows_per_block
            ):
                C_row_offset_in_i32s = (
                    tile_row_block * rows_per_block * m * N * word_size_out // 4
                )
                num_tile_rows = min(
                    [rows_per_block, M_div_m - tile_row_block * rows_per_block]
                )
                IpuDmaMemcpyNd(
                    metadata="outC",
                    bd_id=0,
                    mem=C,
                    offsets=[0, 0, 0, C_row_offset_in_i32s],
                    lengths=[num_tile_rows, N_div_n, m, n_in_i32s_out],
                    strides=[m_x_N_in_i32s_out, n_in_i32s_out, N_in_i32s_out],
                )
                for tile_row in range(num_tile_rows):
                    A_row_offset_in_i32s = (
                        ((tile_row_block * rows_per_block) + tile_row)
                        * m
                        * K
                        * word_size_in
                        // 4
                    )
                    IpuDmaMemcpyNd(
                        metadata="inA",
                        bd_id=2 * tile_row + 1,
                        mem=A,
                        offsets=[0, 0, 0, A_row_offset_in_i32s],
                        lengths=[N_div_n, K_div_k, m, k_in_i32s],
                        strides=[0, k_in_i32s, K_in_i32s],
                    )
                    IpuDmaMemcpyNd(
                        metadata="inB",
                        bd_id=2 * tile_row + 2,
                        mem=B,
                        lengths=[N_div_n, K_div_k, k, n_in_i32s],
                        strides=[n_in_i32s, k_x_N_in_i32s, N_in_i32s],
                    )

                IpuSync(column=0, row=0, direction=0, channel=0)


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
        memRef_3x3_ty = MemRefType.get(
            (
                3,
                3,
            ),
            int16_ty,
        )

        rgba2grayLine = privateFunc(
            "rgba2grayLine", inputs=[memRef_256_ty, memRef_64_ty, int32_ty]
        )
        filter2dLine = privateFunc(
            "filter2dLine",
            inputs=[
                memRef_64_ty,
                memRef_64_ty,
                memRef_64_ty,
                memRef_64_ty,
                int32_ty,
                memRef_3x3_ty,
            ],
        )
        thresholdLine = privateFunc(
            "thresholdLine",
            inputs=[memRef_64_ty, memRef_64_ty, int32_ty, int16_ty, int16_ty, int8_ty],
        )
        gray2rgbaLine = privateFunc(
            "gray2rgbaLine", inputs=[memRef_64_ty, memRef_256_ty, int32_ty]
        )
        addWeightedLine = privateFunc(
            "addWeightedLine",
            inputs=[
                memRef_256_ty,
                memRef_256_ty,
                memRef_256_ty,
                int32_ty,
                int16_ty,
                int16_ty,
                int8_ty,
            ],
        )

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
            @forLoop(lowerBound=0, upperBound=36, step=1)
            def loopBody():
                elemIn = Acquire(
                    "inOF_L2L1", ObjectFifoPort.Consume, 1, memRef_256_ty
                ).acquiredElem()
                elemOut = Acquire(
                    "OF_2to3", ObjectFifoPort.Produce, 1, memRef_64_ty
                ).acquiredElem()

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
            elemsInPre = Acquire(
                "OF_2to3", ObjectFifoPort.Consume, 2, memRef_64_ty
            ).acquiredElem()
            elemPreOut = Acquire(
                "OF_3to4", ObjectFifoPort.Produce, 1, memRef_64_ty
            ).acquiredElem()
            Call(
                filter2dLine,
                [
                    elemsInPre[0],
                    elemsInPre[0],
                    elemsInPre[1],
                    elemPreOut,
                    integerConstant(64),
                    kernel,
                ],
            )
            Release(ObjectFifoPort.Produce, "OF_3to4", 1)

            # Steady State : Middle
            @forLoop(lowerBound=1, upperBound=35, step=1)
            def loopBody():
                elemsIn = Acquire(
                    "OF_2to3", ObjectFifoPort.Consume, 3, memRef_64_ty
                ).acquiredElem()
                elemOut = Acquire(
                    "OF_3to4", ObjectFifoPort.Produce, 1, memRef_64_ty
                ).acquiredElem()
                Call(
                    filter2dLine,
                    [
                        elemsIn[0],
                        elemsIn[1],
                        elemsIn[2],
                        elemOut,
                        integerConstant(64),
                        kernel,
                    ],
                )
                Release(ObjectFifoPort.Consume, "OF_2to3", 1)
                Release(ObjectFifoPort.Produce, "OF_3to4", 1)

            # Postamble : Bottom Border
            elemsInPost = Acquire(
                "OF_2to3", ObjectFifoPort.Consume, 2, memRef_64_ty
            ).acquiredElem()
            elemPostOut = Acquire(
                "OF_3to4", ObjectFifoPort.Produce, 1, memRef_64_ty
            ).acquiredElem()
            Call(
                filter2dLine,
                [
                    elemsInPost[0],
                    elemsInPost[1],
                    elemsInPost[1],
                    elemPostOut,
                    integerConstant(64),
                    kernel,
                ],
            )
            Release(ObjectFifoPort.Consume, "OF_2to3", 2)
            Release(ObjectFifoPort.Produce, "OF_3to4", 1)

        @core(T4, "threshold.cc.o")
        def coreBody():
            vThr = integerConstant(10, int16_ty)
            vMax = integerConstant(255, int16_ty)
            vTyp = integerConstant(0, int8_ty)

            @forLoop(lowerBound=0, upperBound=36, step=1)
            def loopBody():
                elemIn = Acquire(
                    "OF_3to4", ObjectFifoPort.Consume, 1, memRef_64_ty
                ).acquiredElem()
                elemOut = Acquire(
                    "OF_4to5", ObjectFifoPort.Produce, 1, memRef_64_ty
                ).acquiredElem()

                Call(
                    thresholdLine,
                    [elemIn, elemOut, integerConstant(64), vThr, vMax, vTyp],
                )

                Release(ObjectFifoPort.Consume, "OF_3to4", 1)
                Release(ObjectFifoPort.Produce, "OF_4to5", 1)

        @core(T5, "combined_gray2rgba_addWeighted.a")
        def coreBody():
            @forLoop(lowerBound=0, upperBound=36, step=1)
            def loopBody():
                elemIn = Acquire(
                    "OF_4to5", ObjectFifoPort.Consume, 1, memRef_64_ty
                ).acquiredElem()
                elemOut = Acquire(
                    "OF_5to5", ObjectFifoPort.Produce, 1, memRef_256_ty
                ).acquiredElem()

                Call(gray2rgbaLine, [elemIn, elemOut, integerConstant(64)])

                Release(ObjectFifoPort.Consume, "OF_4to5", 1)
                Release(ObjectFifoPort.Produce, "OF_5to5", 1)

                elemIn1 = Acquire(
                    "OF_5to5", ObjectFifoPort.Consume, 1, memRef_256_ty
                ).acquiredElem()
                elemIn2 = Acquire(
                    "inOF_L2L1", ObjectFifoPort.Consume, 1, memRef_256_ty
                ).acquiredElem()
                elemOut2 = Acquire(
                    "outOF_L1L2", ObjectFifoPort.Produce, 1, memRef_256_ty
                ).acquiredElem()

                alpha = integerConstant(16384, int16_ty)
                beta = integerConstant(16384, int16_ty)
                gamma = integerConstant(0, int8_ty)

                Call(
                    addWeightedLine,
                    [
                        elemIn1,
                        elemIn2,
                        elemOut2,
                        integerConstant(256),
                        alpha,
                        beta,
                        gamma,
                    ],
                )

                Release(ObjectFifoPort.Consume, "OF_5to5", 1)
                Release(ObjectFifoPort.Consume, "inOF_L2L1", 1)
                Release(ObjectFifoPort.Produce, "outOF_L1L2", 1)

        memRef_mem_ty = MemRefType.get((2304,), int32_ty)

        @FuncOp.from_py_func(memRef_mem_ty, memRef_mem_ty, memRef_mem_ty)
        def sequence(I, B, O):
            IpuDmaMemcpyNd(
                metadata="outOF_L2L3",
                bd_id=0,
                mem=O,
                lengths=[1, 1, 36, 64],
                strides=[0, 0, 64],
            )
            IpuDmaMemcpyNd(
                metadata="inOF_L3L2",
                bd_id=1,
                mem=I,
                lengths=[1, 1, 36, 64],
                strides=[0, 0, 64],
            )
            IpuSync(column=0, row=0, direction=0, channel=0)
