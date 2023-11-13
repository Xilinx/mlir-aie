# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

# RUN: %python %s | FileCheck %s

import aie
from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.passmanager import PassManager


def constructAndPrintInModule(f):
    with Context() as ctx, Location.unknown():
        aie.dialects.aie.register_dialect(ctx)
        module = Module.create()
        print("\nTEST:", f.__name__)
        with InsertionPoint(module.body):
            f()
        pm = PassManager("builtin.module")
        pm.add("canonicalize")
        pm.run(module.operation)
        print(module)


# CHECK-LABEL: my_vector_scalar
# CHECK: module {
# CHECK:   AIE.device(ipu) {
# CHECK:     func.func private @scale_int32(memref<1024xi32>, memref<1024xi32>)
# CHECK:     %tile_0_0 = AIE.tile(0, 0)
# CHECK:     %tile_0_2 = AIE.tile(0, 2)
# CHECK:     AIE.objectFifo @in(%tile_0_0, {%tile_0_2}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
# CHECK:     AIE.objectFifo @out(%tile_0_2, {%tile_0_0}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
# CHECK:     %core_0_2 = AIE.core(%tile_0_2) {
# CHECK:       %c4 = arith.constant 4 : index
# CHECK:       %c0 = arith.constant 0 : index
# CHECK:       %c4294967295 = arith.constant 4294967295 : index
# CHECK:       %c1 = arith.constant 1 : index
# CHECK:       scf.for %arg0 = %c0 to %c4294967295 step %c1 {
# CHECK:         scf.for %arg1 = %c0 to %c4 step %c1 {
# CHECK:           %0 = AIE.objectFifo.acquire @out(Produce, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
# CHECK:           %1 = AIE.objectFifo.subview.access %0[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
# CHECK:           %2 = AIE.objectFifo.acquire @in(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
# CHECK:           %3 = AIE.objectFifo.subview.access %2[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
# CHECK:           func.call @scale_int32(%3, %1) : (memref<1024xi32>, memref<1024xi32>) -> ()
# CHECK:           AIE.objectFifo.release @in(Consume, 1)
# CHECK:           AIE.objectFifo.release @out(Produce, 1)
# CHECK:         }
# CHECK:       }
# CHECK:       AIE.end
# CHECK:     } {link_with = "scale.o"}
# CHECK:     func.func @sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
# CHECK:       %c0_i32 = arith.constant 0 : i32
# CHECK:       %c1_i32 = arith.constant 1 : i32
# CHECK:       %c4096_i32 = arith.constant 4096 : i32
# CHECK:       AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg2[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c1_i32, %c1_i32, %c1_i32, %c4096_i32] [%c0_i32, %c0_i32, %c0_i32]) {id = 0 : i32, metadata = @out} : (i32, i32, memref<4096xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
# CHECK:       AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c1_i32, %c1_i32, %c1_i32, %c4096_i32] [%c0_i32, %c0_i32, %c0_i32]) {id = 1 : i32, metadata = @in} : (i32, i32, memref<4096xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
# CHECK:       AIEX.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
# CHECK:       return
# CHECK:     }
# CHECK:   }
# CHECK: }
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


# CHECK-LABEL: my_matmul
# CHECK: module {
# CHECK:   AIE.device(ipu) {
# CHECK:     func.func private @zero_scalar_i16(memref<64x64xi16>)
# CHECK:     func.func private @zero_i16(memref<64x64xi16>)
# CHECK:     func.func private @matmul_scalar_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
# CHECK:     func.func private @matmul_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
# CHECK:     %tile_0_0 = AIE.tile(0, 0)
# CHECK:     %tile_0_1 = AIE.tile(0, 1)
# CHECK:     %tile_0_2 = AIE.tile(0, 2)
# CHECK:     AIE.objectFifo @inA(%tile_0_0, {%tile_0_2}, 2 : i32) : !AIE.objectFifo<memref<64x32xi16>>
# CHECK:     AIE.objectFifo @inB(%tile_0_0, {%tile_0_2}, 2 : i32) : !AIE.objectFifo<memref<32x64xi16>>
# CHECK:     AIE.objectFifo @outC(%tile_0_2, {%tile_0_0}, 2 : i32) : !AIE.objectFifo<memref<64x64xi16>>
# CHECK:     %core_0_2 = AIE.core(%tile_0_2) {
# CHECK:       %c4 = arith.constant 4 : index
# CHECK:       %c0 = arith.constant 0 : index
# CHECK:       %c4294967295 = arith.constant 4294967295 : index
# CHECK:       %c1 = arith.constant 1 : index
# CHECK:       scf.for %arg0 = %c0 to %c4294967295 step %c1 {
# CHECK:         scf.for %arg1 = %c0 to %c4 step %c1 {
# CHECK:           %0 = AIE.objectFifo.acquire @outC(Produce, 1) : !AIE.objectFifoSubview<memref<64x64xi16>>
# CHECK:           %1 = AIE.objectFifo.subview.access %0[0] : !AIE.objectFifoSubview<memref<64x64xi16>> -> memref<64x64xi16>
# CHECK:           func.call @zero_i16(%1) : (memref<64x64xi16>) -> ()
# CHECK:           scf.for %arg2 = %c0 to %c4 step %c1 {
# CHECK:             %2 = AIE.objectFifo.acquire @inA(Consume, 1) : !AIE.objectFifoSubview<memref<64x32xi16>>
# CHECK:             %3 = AIE.objectFifo.subview.access %2[0] : !AIE.objectFifoSubview<memref<64x32xi16>> -> memref<64x32xi16>
# CHECK:             %4 = AIE.objectFifo.acquire @inB(Consume, 1) : !AIE.objectFifoSubview<memref<32x64xi16>>
# CHECK:             %5 = AIE.objectFifo.subview.access %4[0] : !AIE.objectFifoSubview<memref<32x64xi16>> -> memref<32x64xi16>
# CHECK:             func.call @matmul_i16_i16(%3, %5, %1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
# CHECK:             AIE.objectFifo.release @inA(Consume, 1)
# CHECK:             AIE.objectFifo.release @inB(Consume, 1)
# CHECK:           }
# CHECK:           AIE.objectFifo.release @outC(Produce, 1)
# CHECK:         }
# CHECK:       }
# CHECK:       AIE.end
# CHECK:     } {link_with = "mm.o"}
# CHECK:     func.func @sequence(%arg0: memref<8192xi32>, %arg1: memref<8192xi32>, %arg2: memref<8192xi32>) {
# CHECK:       %c2048_i32 = arith.constant 2048 : i32
# CHECK:       %c16_i32 = arith.constant 16 : i32
# CHECK:       %c4_i32 = arith.constant 4 : i32
# CHECK:       %c0_i32 = arith.constant 0 : i32
# CHECK:       %c2_i32 = arith.constant 2 : i32
# CHECK:       %c64_i32 = arith.constant 64 : i32
# CHECK:       %c32_i32 = arith.constant 32 : i32
# CHECK:       %c4096_i32 = arith.constant 4096 : i32
# CHECK:       AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg2[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c2_i32, %c2_i32, %c64_i32, %c32_i32] [%c4096_i32, %c32_i32, %c64_i32]) {id = 0 : i32, metadata = @outC} : (i32, i32, memref<8192xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
# CHECK:       AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c2_i32, %c4_i32, %c64_i32, %c16_i32] [%c0_i32, %c16_i32, %c64_i32]) {id = 1 : i32, metadata = @inA} : (i32, i32, memref<8192xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
# CHECK:       AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg1[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c2_i32, %c4_i32, %c32_i32, %c32_i32] [%c32_i32, %c2048_i32, %c64_i32]) {id = 2 : i32, metadata = @inB} : (i32, i32, memref<8192xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
# CHECK:       AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg0[%c0_i32, %c0_i32, %c0_i32, %c4096_i32] [%c2_i32, %c4_i32, %c64_i32, %c16_i32] [%c0_i32, %c16_i32, %c64_i32]) {id = 3 : i32, metadata = @inA} : (i32, i32, memref<8192xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
# CHECK:       AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg1[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c2_i32, %c4_i32, %c32_i32, %c32_i32] [%c32_i32, %c2048_i32, %c64_i32]) {id = 4 : i32, metadata = @inB} : (i32, i32, memref<8192xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
# CHECK:       AIEX.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
# CHECK:       return
# CHECK:     }
# CHECK:   }
# CHECK: }
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


# CHECK-LABEL: edge_detect
# CHECK: module {
# CHECK:   AIE.device(ipu) {
# CHECK:     func.func private @rgba2grayLine(memref<256xui8>, memref<64xui8>, i32)
# CHECK:     func.func private @filter2dLine(memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>)
# CHECK:     func.func private @thresholdLine(memref<64xui8>, memref<64xui8>, i32, i16, i16, i8)
# CHECK:     func.func private @gray2rgbaLine(memref<64xui8>, memref<256xui8>, i32)
# CHECK:     func.func private @addWeightedLine(memref<256xui8>, memref<256xui8>, memref<256xui8>, i32, i16, i16, i8)
# CHECK:     %tile_0_0 = AIE.tile(0, 0)
# CHECK:     %tile_0_1 = AIE.tile(0, 1)
# CHECK:     %tile_0_2 = AIE.tile(0, 2)
# CHECK:     %tile_0_3 = AIE.tile(0, 3)
# CHECK:     %tile_0_4 = AIE.tile(0, 4)
# CHECK:     %tile_0_5 = AIE.tile(0, 5)
# CHECK:     AIE.objectFifo @inOF_L3L2(%tile_0_0, {%tile_0_1}, 2 : i32) : !AIE.objectFifo<memref<256xui8>>
# CHECK:     AIE.objectFifo @inOF_L2L1(%tile_0_1, {%tile_0_2, %tile_0_5}, [2 : i32, 2 : i32, 7 : i32]) : !AIE.objectFifo<memref<256xui8>>
# CHECK:     AIE.objectFifo.link [@inOF_L3L2] -> [@inOF_L2L1]()
# CHECK:     AIE.objectFifo @outOF_L2L3(%tile_0_1, {%tile_0_0}, 2 : i32) : !AIE.objectFifo<memref<256xui8>>
# CHECK:     AIE.objectFifo @outOF_L1L2(%tile_0_5, {%tile_0_1}, 2 : i32) : !AIE.objectFifo<memref<256xui8>>
# CHECK:     AIE.objectFifo.link [@outOF_L1L2] -> [@outOF_L2L3]()
# CHECK:     AIE.objectFifo @OF_2to3(%tile_0_2, {%tile_0_3}, 4 : i32) : !AIE.objectFifo<memref<64xui8>>
# CHECK:     AIE.objectFifo @OF_3to4(%tile_0_3, {%tile_0_4}, 2 : i32) : !AIE.objectFifo<memref<64xui8>>
# CHECK:     AIE.objectFifo @OF_4to5(%tile_0_4, {%tile_0_5}, 2 : i32) : !AIE.objectFifo<memref<64xui8>>
# CHECK:     AIE.objectFifo @OF_5to5(%tile_0_5, {%tile_0_5}, 1 : i32) : !AIE.objectFifo<memref<256xui8>>
# CHECK:     %core_0_2 = AIE.core(%tile_0_2) {
# CHECK:       %c64_i32 = arith.constant 64 : i32
# CHECK:       %c0 = arith.constant 0 : index
# CHECK:       %c36 = arith.constant 36 : index
# CHECK:       %c1 = arith.constant 1 : index
# CHECK:       scf.for %arg0 = %c0 to %c36 step %c1 {
# CHECK:         %0 = AIE.objectFifo.acquire @inOF_L2L1(Consume, 1) : !AIE.objectFifoSubview<memref<256xui8>>
# CHECK:         %1 = AIE.objectFifo.subview.access %0[0] : !AIE.objectFifoSubview<memref<256xui8>> -> memref<256xui8>
# CHECK:         %2 = AIE.objectFifo.acquire @OF_2to3(Produce, 1) : !AIE.objectFifoSubview<memref<64xui8>>
# CHECK:         %3 = AIE.objectFifo.subview.access %2[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         func.call @rgba2grayLine(%1, %3, %c64_i32) : (memref<256xui8>, memref<64xui8>, i32) -> ()
# CHECK:         AIE.objectFifo.release @inOF_L2L1(Consume, 1)
# CHECK:         AIE.objectFifo.release @OF_2to3(Produce, 1)
# CHECK:       }
# CHECK:       AIE.end
# CHECK:     } {link_with = "rgba2gray.cc.o"}
# CHECK:     %core_0_3 = AIE.core(%tile_0_3) {
# CHECK:       %c35 = arith.constant 35 : index
# CHECK:       %c64_i32 = arith.constant 64 : i32
# CHECK:       %c2 = arith.constant 2 : index
# CHECK:       %c1 = arith.constant 1 : index
# CHECK:       %c0 = arith.constant 0 : index
# CHECK:       %c-16384_i16 = arith.constant -16384 : i16
# CHECK:       %c4096_i16 = arith.constant 4096 : i16
# CHECK:       %c0_i16 = arith.constant 0 : i16
# CHECK:       %alloc = memref.alloc() : memref<3x3xi16>
# CHECK:       memref.store %c0_i16, %alloc[%c0, %c0] : memref<3x3xi16>
# CHECK:       memref.store %c4096_i16, %alloc[%c0, %c1] : memref<3x3xi16>
# CHECK:       memref.store %c0_i16, %alloc[%c0, %c2] : memref<3x3xi16>
# CHECK:       memref.store %c4096_i16, %alloc[%c1, %c0] : memref<3x3xi16>
# CHECK:       memref.store %c-16384_i16, %alloc[%c1, %c1] : memref<3x3xi16>
# CHECK:       memref.store %c4096_i16, %alloc[%c1, %c2] : memref<3x3xi16>
# CHECK:       memref.store %c0_i16, %alloc[%c2, %c0] : memref<3x3xi16>
# CHECK:       memref.store %c4096_i16, %alloc[%c2, %c1] : memref<3x3xi16>
# CHECK:       memref.store %c0_i16, %alloc[%c2, %c2] : memref<3x3xi16>
# CHECK:       %0 = AIE.objectFifo.acquire @OF_2to3(Consume, 2) : !AIE.objectFifoSubview<memref<64xui8>>
# CHECK:       %1 = AIE.objectFifo.subview.access %0[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:       %2 = AIE.objectFifo.subview.access %0[1] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:       %3 = AIE.objectFifo.acquire @OF_3to4(Produce, 1) : !AIE.objectFifoSubview<memref<64xui8>>
# CHECK:       %4 = AIE.objectFifo.subview.access %3[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:       func.call @filter2dLine(%1, %1, %2, %4, %c64_i32, %alloc) : (memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>) -> ()
# CHECK:       AIE.objectFifo.release @OF_3to4(Produce, 1)
# CHECK:       scf.for %arg0 = %c1 to %c35 step %c1 {
# CHECK:         %10 = AIE.objectFifo.acquire @OF_2to3(Consume, 3) : !AIE.objectFifoSubview<memref<64xui8>>
# CHECK:         %11 = AIE.objectFifo.subview.access %10[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         %12 = AIE.objectFifo.subview.access %10[1] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         %13 = AIE.objectFifo.subview.access %10[2] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         %14 = AIE.objectFifo.acquire @OF_3to4(Produce, 1) : !AIE.objectFifoSubview<memref<64xui8>>
# CHECK:         %15 = AIE.objectFifo.subview.access %14[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         func.call @filter2dLine(%11, %12, %13, %15, %c64_i32, %alloc) : (memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>) -> ()
# CHECK:         AIE.objectFifo.release @OF_2to3(Consume, 1)
# CHECK:         AIE.objectFifo.release @OF_3to4(Produce, 1)
# CHECK:       }
# CHECK:       %5 = AIE.objectFifo.acquire @OF_2to3(Consume, 2) : !AIE.objectFifoSubview<memref<64xui8>>
# CHECK:       %6 = AIE.objectFifo.subview.access %5[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:       %7 = AIE.objectFifo.subview.access %5[1] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:       %8 = AIE.objectFifo.acquire @OF_3to4(Produce, 1) : !AIE.objectFifoSubview<memref<64xui8>>
# CHECK:       %9 = AIE.objectFifo.subview.access %8[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:       func.call @filter2dLine(%6, %7, %7, %9, %c64_i32, %alloc) : (memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>) -> ()
# CHECK:       AIE.objectFifo.release @OF_2to3(Consume, 2)
# CHECK:       AIE.objectFifo.release @OF_3to4(Produce, 1)
# CHECK:       AIE.end
# CHECK:     } {link_with = "filter2d.cc.o"}
# CHECK:     %core_0_4 = AIE.core(%tile_0_4) {
# CHECK:       %c64_i32 = arith.constant 64 : i32
# CHECK:       %c10_i16 = arith.constant 10 : i16
# CHECK:       %c255_i16 = arith.constant 255 : i16
# CHECK:       %c0_i8 = arith.constant 0 : i8
# CHECK:       %c0 = arith.constant 0 : index
# CHECK:       %c36 = arith.constant 36 : index
# CHECK:       %c1 = arith.constant 1 : index
# CHECK:       scf.for %arg0 = %c0 to %c36 step %c1 {
# CHECK:         %0 = AIE.objectFifo.acquire @OF_3to4(Consume, 1) : !AIE.objectFifoSubview<memref<64xui8>>
# CHECK:         %1 = AIE.objectFifo.subview.access %0[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         %2 = AIE.objectFifo.acquire @OF_4to5(Produce, 1) : !AIE.objectFifoSubview<memref<64xui8>>
# CHECK:         %3 = AIE.objectFifo.subview.access %2[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         func.call @thresholdLine(%1, %3, %c64_i32, %c10_i16, %c255_i16, %c0_i8) : (memref<64xui8>, memref<64xui8>, i32, i16, i16, i8) -> ()
# CHECK:         AIE.objectFifo.release @OF_3to4(Consume, 1)
# CHECK:         AIE.objectFifo.release @OF_4to5(Produce, 1)
# CHECK:       }
# CHECK:       AIE.end
# CHECK:     } {link_with = "threshold.cc.o"}
# CHECK:     %core_0_5 = AIE.core(%tile_0_5) {
# CHECK:       %c256_i32 = arith.constant 256 : i32
# CHECK:       %c0_i8 = arith.constant 0 : i8
# CHECK:       %c16384_i16 = arith.constant 16384 : i16
# CHECK:       %c64_i32 = arith.constant 64 : i32
# CHECK:       %c0 = arith.constant 0 : index
# CHECK:       %c36 = arith.constant 36 : index
# CHECK:       %c1 = arith.constant 1 : index
# CHECK:       scf.for %arg0 = %c0 to %c36 step %c1 {
# CHECK:         %0 = AIE.objectFifo.acquire @OF_4to5(Consume, 1) : !AIE.objectFifoSubview<memref<64xui8>>
# CHECK:         %1 = AIE.objectFifo.subview.access %0[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         %2 = AIE.objectFifo.acquire @OF_5to5(Produce, 1) : !AIE.objectFifoSubview<memref<256xui8>>
# CHECK:         %3 = AIE.objectFifo.subview.access %2[0] : !AIE.objectFifoSubview<memref<256xui8>> -> memref<256xui8>
# CHECK:         func.call @gray2rgbaLine(%1, %3, %c64_i32) : (memref<64xui8>, memref<256xui8>, i32) -> ()
# CHECK:         AIE.objectFifo.release @OF_4to5(Consume, 1)
# CHECK:         AIE.objectFifo.release @OF_5to5(Produce, 1)
# CHECK:         %4 = AIE.objectFifo.acquire @OF_5to5(Consume, 1) : !AIE.objectFifoSubview<memref<256xui8>>
# CHECK:         %5 = AIE.objectFifo.subview.access %4[0] : !AIE.objectFifoSubview<memref<256xui8>> -> memref<256xui8>
# CHECK:         %6 = AIE.objectFifo.acquire @inOF_L2L1(Consume, 1) : !AIE.objectFifoSubview<memref<256xui8>>
# CHECK:         %7 = AIE.objectFifo.subview.access %6[0] : !AIE.objectFifoSubview<memref<256xui8>> -> memref<256xui8>
# CHECK:         %8 = AIE.objectFifo.acquire @outOF_L1L2(Produce, 1) : !AIE.objectFifoSubview<memref<256xui8>>
# CHECK:         %9 = AIE.objectFifo.subview.access %8[0] : !AIE.objectFifoSubview<memref<256xui8>> -> memref<256xui8>
# CHECK:         func.call @addWeightedLine(%5, %7, %9, %c256_i32, %c16384_i16, %c16384_i16, %c0_i8) : (memref<256xui8>, memref<256xui8>, memref<256xui8>, i32, i16, i16, i8) -> ()
# CHECK:         AIE.objectFifo.release @OF_5to5(Consume, 1)
# CHECK:         AIE.objectFifo.release @inOF_L2L1(Consume, 1)
# CHECK:         AIE.objectFifo.release @outOF_L1L2(Produce, 1)
# CHECK:       }
# CHECK:       AIE.end
# CHECK:     } {link_with = "combined_gray2rgba_addWeighted.a"}
# CHECK:     func.func @sequence(%arg0: memref<2304xi32>, %arg1: memref<2304xi32>, %arg2: memref<2304xi32>) {
# CHECK:       %c0_i32 = arith.constant 0 : i32
# CHECK:       %c1_i32 = arith.constant 1 : i32
# CHECK:       %c36_i32 = arith.constant 36 : i32
# CHECK:       %c64_i32 = arith.constant 64 : i32
# CHECK:       AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg2[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c1_i32, %c1_i32, %c36_i32, %c64_i32] [%c0_i32, %c0_i32, %c64_i32]) {id = 0 : i32, metadata = @outOF_L2L3} : (i32, i32, memref<2304xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
# CHECK:       AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c1_i32, %c1_i32, %c36_i32, %c64_i32] [%c0_i32, %c0_i32, %c64_i32]) {id = 1 : i32, metadata = @inOF_L3L2} : (i32, i32, memref<2304xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
# CHECK:       AIEX.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
# CHECK:       return
# CHECK:     }
# CHECK:   }
# CHECK: }
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
