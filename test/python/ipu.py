# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

# RUN: %python %s | FileCheck %s

import aie.extras.types as T
from aie.dialects.aie import (
    AIEDevice,
    Call,
    ObjectFifoPort,
    ObjectFifoType,
    acquire,
    core,
    device,
    external_func,
    objectFifo,
    objectFifo_link,
    objectFifo_release,
    tile,
)
from aie.dialects.aiex import ipu_sync, ipu_dma_memcpy_nd
from aie.dialects.extras import memref, arith
from aie.dialects.func import FuncOp
from aie.dialects.scf import for_
from aie.dialects.scf import yield_
from aie.ir import Context, Location, Module, InsertionPoint, TypeAttr
from aie.passmanager import PassManager

range_ = for_


def construct_and_print_module(f):
    with Context() as ctx, Location.unknown():
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


@construct_and_print_module
def my_vector_scalar():
    N = 4096
    n = 1024
    N_div_n = N // n

    buffer_depth = 2

    @device(AIEDevice.ipu)
    def device_body():
        scale_int32 = external_func(
            "scale_int32", inputs=[T.memref(n, T.i32()), T.memref(n, T.i32())]
        )

        S = tile(0, 0)
        M = tile(0, 2)

        objectFifo(
            "in",
            S,
            [M],
            buffer_depth,
            TypeAttr.get(ObjectFifoType.get(T.memref(n, T.i32()))),
            [],
            [],
        )
        objectFifo(
            "out",
            M,
            [S],
            buffer_depth,
            TypeAttr.get(ObjectFifoType.get(T.memref(n, T.i32()))),
            [],
            [],
        )

        @core(M, "scale.o")
        def core_body():
            # Effective while(1)
            for _ in range_(0xFFFFFFFF):
                # Number of sub-vector "tile" iterations
                for _ in range_(N_div_n):
                    elem_out = acquire(
                        ObjectFifoPort.Produce, "out", 1, T.memref(n, T.i32())
                    ).acquired_elem()
                    elem_in = acquire(
                        ObjectFifoPort.Consume, "in", 1, T.memref(n, T.i32())
                    ).acquired_elem()
                    Call(scale_int32, [elem_in, elem_out])
                    objectFifo_release(ObjectFifoPort.Consume, "in", 1)
                    objectFifo_release(ObjectFifoPort.Produce, "out", 1)
                    yield_([])
                yield_([])

        @FuncOp.from_py_func(
            T.memref(N, T.i32()), T.memref(N, T.i32()), T.memref(N, T.i32())
        )
        def sequence(A, B, C):
            ipu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, lengths=[1, 1, 1, N])
            ipu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, lengths=[1, 1, 1, N])
            ipu_sync(column=0, row=0, direction=0, channel=0)


# CHECK-LABEL: my_matmul
# CHECK: module {
# CHECK:   AIE.device(ipu) {
# CHECK:     func.func private @zero_scalar_i16(memref<64x64xi16>)
# CHECK:     func.func private @zero_i16(memref<64x64xi16>)
# CHECK:     func.func private @matmul_scalar_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
# CHECK:     func.func private @matmul_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
# CHECK:     %tile_0_0 = AIE.tile(0, 0)
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


@construct_and_print_module
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
    def device_body():
        zero_scalar = external_func("zero_scalar_i16", inputs=[T.memref(m, n, T.i16())])
        zero = external_func("zero_i16", inputs=[T.memref(m, n, T.i16())])
        matmul_scalar = external_func(
            "matmul_scalar_i16_i16",
            inputs=[
                T.memref(m, k, T.i16()),
                T.memref(k, n, T.i16()),
                T.memref(m, n, T.i16()),
            ],
        )
        matmul = external_func(
            "matmul_i16_i16",
            inputs=[
                T.memref(m, k, T.i16()),
                T.memref(k, n, T.i16()),
                T.memref(m, n, T.i16()),
            ],
        )

        S = tile(0, 0)
        M = tile(0, 2)

        objectFifo(
            "inA",
            S,
            [M],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(m, k, T.i16()))),
            [],
            [],
        )
        objectFifo(
            "inB",
            S,
            [M],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(k, n, T.i16()))),
            [],
            [],
        )
        objectFifo(
            "outC",
            M,
            [S],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(m, n, T.i16()))),
            [],
            [],
        )

        @core(M, "mm.o")
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(tiles):
                    elem_out = acquire(
                        ObjectFifoPort.Produce, "outC", 1, T.memref(m, n, T.i16())
                    ).acquired_elem()
                    if vectorized:
                        Call(zero, [elem_out])
                    else:
                        Call(zero_scalar, [elem_out])

                    for _ in range_(K_div_k):
                        elem_in_a = acquire(
                            ObjectFifoPort.Consume, "inA", 1, T.memref(m, k, T.i16())
                        ).acquired_elem()
                        elem_in_b = acquire(
                            ObjectFifoPort.Consume, "inB", 1, T.memref(k, n, T.i16())
                        ).acquired_elem()
                        if vectorized:
                            Call(matmul, [elem_in_a, elem_in_b, elem_out])
                        else:
                            Call(matmul_scalar, [elem_in_a, elem_in_b, elem_out])
                        objectFifo_release(ObjectFifoPort.Consume, "inA", 1)
                        objectFifo_release(ObjectFifoPort.Consume, "inB", 1)
                        yield_([])

                    objectFifo_release(ObjectFifoPort.Produce, "outC", 1)
                    yield_([])
                yield_([])

        @FuncOp.from_py_func(
            T.memref(A_sz_in_i32s, T.i32()),
            T.memref(B_sz_in_i32s, T.i32()),
            T.memref(C_sz_in_i32s, T.i32()),
        )
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
                ipu_dma_memcpy_nd(
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
                    ipu_dma_memcpy_nd(
                        metadata="inA",
                        bd_id=2 * tile_row + 1,
                        mem=A,
                        offsets=[0, 0, 0, A_row_offset_in_i32s],
                        lengths=[N_div_n, K_div_k, m, k_in_i32s],
                        strides=[0, k_in_i32s, K_in_i32s],
                    )
                    ipu_dma_memcpy_nd(
                        metadata="inB",
                        bd_id=2 * tile_row + 2,
                        mem=B,
                        lengths=[N_div_n, K_div_k, k, n_in_i32s],
                        strides=[n_in_i32s, k_x_N_in_i32s, N_in_i32s],
                    )

                ipu_sync(column=0, row=0, direction=0, channel=0)


# CHECK-LABEL: edge_detect
# CHECK: module {
# CHECK:   AIE.device(ipu) {
# CHECK:     func.func private @rgba2gray_line(memref<256xui8>, memref<64xui8>, i32)
# CHECK:     func.func private @filter2d_line(memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>)
# CHECK:     func.func private @threshold_line(memref<64xui8>, memref<64xui8>, i32, i16, i16, i8)
# CHECK:     func.func private @gray2rgba_line(memref<64xui8>, memref<256xui8>, i32)
# CHECK:     func.func private @add_weighted_line(memref<256xui8>, memref<256xui8>, memref<256xui8>, i32, i16, i16, i8)
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
# CHECK:         func.call @rgba2gray_line(%1, %3, %c64_i32) : (memref<256xui8>, memref<64xui8>, i32) -> ()
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
# CHECK:       func.call @filter2d_line(%1, %1, %2, %4, %c64_i32, %alloc) : (memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>) -> ()
# CHECK:       AIE.objectFifo.release @OF_3to4(Produce, 1)
# CHECK:       scf.for %arg0 = %c1 to %c35 step %c1 {
# CHECK:         %10 = AIE.objectFifo.acquire @OF_2to3(Consume, 3) : !AIE.objectFifoSubview<memref<64xui8>>
# CHECK:         %11 = AIE.objectFifo.subview.access %10[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         %12 = AIE.objectFifo.subview.access %10[1] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         %13 = AIE.objectFifo.subview.access %10[2] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         %14 = AIE.objectFifo.acquire @OF_3to4(Produce, 1) : !AIE.objectFifoSubview<memref<64xui8>>
# CHECK:         %15 = AIE.objectFifo.subview.access %14[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         func.call @filter2d_line(%11, %12, %13, %15, %c64_i32, %alloc) : (memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>) -> ()
# CHECK:         AIE.objectFifo.release @OF_2to3(Consume, 1)
# CHECK:         AIE.objectFifo.release @OF_3to4(Produce, 1)
# CHECK:       }
# CHECK:       %5 = AIE.objectFifo.acquire @OF_2to3(Consume, 2) : !AIE.objectFifoSubview<memref<64xui8>>
# CHECK:       %6 = AIE.objectFifo.subview.access %5[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:       %7 = AIE.objectFifo.subview.access %5[1] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:       %8 = AIE.objectFifo.acquire @OF_3to4(Produce, 1) : !AIE.objectFifoSubview<memref<64xui8>>
# CHECK:       %9 = AIE.objectFifo.subview.access %8[0] : !AIE.objectFifoSubview<memref<64xui8>> -> memref<64xui8>
# CHECK:       func.call @filter2d_line(%6, %7, %7, %9, %c64_i32, %alloc) : (memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>) -> ()
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
# CHECK:         func.call @threshold_line(%1, %3, %c64_i32, %c10_i16, %c255_i16, %c0_i8) : (memref<64xui8>, memref<64xui8>, i32, i16, i16, i8) -> ()
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
# CHECK:         func.call @gray2rgba_line(%1, %3, %c64_i32) : (memref<64xui8>, memref<256xui8>, i32) -> ()
# CHECK:         AIE.objectFifo.release @OF_4to5(Consume, 1)
# CHECK:         AIE.objectFifo.release @OF_5to5(Produce, 1)
# CHECK:         %4 = AIE.objectFifo.acquire @OF_5to5(Consume, 1) : !AIE.objectFifoSubview<memref<256xui8>>
# CHECK:         %5 = AIE.objectFifo.subview.access %4[0] : !AIE.objectFifoSubview<memref<256xui8>> -> memref<256xui8>
# CHECK:         %6 = AIE.objectFifo.acquire @inOF_L2L1(Consume, 1) : !AIE.objectFifoSubview<memref<256xui8>>
# CHECK:         %7 = AIE.objectFifo.subview.access %6[0] : !AIE.objectFifoSubview<memref<256xui8>> -> memref<256xui8>
# CHECK:         %8 = AIE.objectFifo.acquire @outOF_L1L2(Produce, 1) : !AIE.objectFifoSubview<memref<256xui8>>
# CHECK:         %9 = AIE.objectFifo.subview.access %8[0] : !AIE.objectFifoSubview<memref<256xui8>> -> memref<256xui8>
# CHECK:         func.call @add_weighted_line(%5, %7, %9, %c256_i32, %c16384_i16, %c16384_i16, %c0_i8) : (memref<256xui8>, memref<256xui8>, memref<256xui8>, i32, i16, i16, i8) -> ()
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


@construct_and_print_module
def edge_detect():
    @device(AIEDevice.ipu)
    def device_body():
        rgba2gray_line = external_func(
            "rgba2gray_line",
            inputs=[T.memref(256, T.ui8()), T.memref(64, T.ui8()), T.i32()],
        )
        filter2d_line = external_func(
            "filter2d_line",
            inputs=[
                T.memref(64, T.ui8()),
                T.memref(64, T.ui8()),
                T.memref(64, T.ui8()),
                T.memref(64, T.ui8()),
                T.i32(),
                T.memref(3, 3, T.i16()),
            ],
        )
        threshold_line = external_func(
            "threshold_line",
            inputs=[
                T.memref(64, T.ui8()),
                T.memref(64, T.ui8()),
                T.i32(),
                T.i16(),
                T.i16(),
                T.i8(),
            ],
        )
        gray2rgba_line = external_func(
            "gray2rgba_line",
            inputs=[T.memref(64, T.ui8()), T.memref(256, T.ui8()), T.i32()],
        )
        add_weighted_line = external_func(
            "add_weighted_line",
            inputs=[
                T.memref(256, T.ui8()),
                T.memref(256, T.ui8()),
                T.memref(256, T.ui8()),
                T.i32(),
                T.i16(),
                T.i16(),
                T.i8(),
            ],
        )

        S = tile(0, 0)
        M = tile(0, 1)
        T2 = tile(0, 2)
        T3 = tile(0, 3)
        T4 = tile(0, 4)
        T5 = tile(0, 5)

        objectFifo(
            "inOF_L3L2",
            S,
            [M],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(256, T.ui8()))),
            [],
            [],
        )
        objectFifo(
            "inOF_L2L1",
            M,
            [T2, T5],
            [2, 2, 7],
            TypeAttr.get(ObjectFifoType.get(T.memref(256, T.ui8()))),
            [],
            [],
        )
        objectFifo_link(["inOF_L3L2"], ["inOF_L2L1"])

        objectFifo(
            "outOF_L2L3",
            M,
            [S],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(256, T.ui8()))),
            [],
            [],
        )
        objectFifo(
            "outOF_L1L2",
            T5,
            [M],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(256, T.ui8()))),
            [],
            [],
        )
        objectFifo_link(["outOF_L1L2"], ["outOF_L2L3"])

        objectFifo(
            "OF_2to3",
            T2,
            [T3],
            4,
            TypeAttr.get(ObjectFifoType.get(T.memref(64, T.ui8()))),
            [],
            [],
        )
        objectFifo(
            "OF_3to4",
            T3,
            [T4],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(64, T.ui8()))),
            [],
            [],
        )
        objectFifo(
            "OF_4to5",
            T4,
            [T5],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(64, T.ui8()))),
            [],
            [],
        )
        objectFifo(
            "OF_5to5",
            T5,
            [T5],
            1,
            TypeAttr.get(ObjectFifoType.get(T.memref(256, T.ui8()))),
            [],
            [],
        )

        @core(T2, "rgba2gray.cc.o")
        def core_body():
            for _ in range_(36):
                elem_in = acquire(
                    ObjectFifoPort.Consume, "inOF_L2L1", 1, T.memref(256, T.ui8())
                ).acquired_elem()
                elem_out = acquire(
                    ObjectFifoPort.Produce, "OF_2to3", 1, T.memref(64, T.ui8())
                ).acquired_elem()

                Call(rgba2gray_line, [elem_in, elem_out, arith.constant(64)])

                objectFifo_release(ObjectFifoPort.Consume, "inOF_L2L1", 1)
                objectFifo_release(ObjectFifoPort.Produce, "OF_2to3", 1)
                yield_([])

        @core(T3, "filter2d.cc.o")
        def core_body():
            kernel = memref.alloc([3, 3], T.i16())
            v0 = arith.constant(0, T.i16())
            v1 = arith.constant(4096, T.i16())
            v_minus4 = arith.constant(-16384, T.i16())
            memref.store(v0, kernel, [0, 0])
            memref.store(v1, kernel, [0, 1])
            memref.store(v0, kernel, [0, 2])
            memref.store(v1, kernel, [1, 0])
            memref.store(v_minus4, kernel, [1, 1])
            memref.store(v1, kernel, [1, 2])
            memref.store(v0, kernel, [2, 0])
            memref.store(v1, kernel, [2, 1])
            memref.store(v0, kernel, [2, 2])

            # Preamble : Top Border
            elems_in_pre = acquire(
                ObjectFifoPort.Consume, "OF_2to3", 2, T.memref(64, T.ui8())
            ).acquired_elem()
            elem_pre_out = acquire(
                ObjectFifoPort.Produce, "OF_3to4", 1, T.memref(64, T.ui8())
            ).acquired_elem()
            Call(
                filter2d_line,
                [
                    elems_in_pre[0],
                    elems_in_pre[0],
                    elems_in_pre[1],
                    elem_pre_out,
                    arith.constant(64),
                    kernel,
                ],
            )
            objectFifo_release(ObjectFifoPort.Produce, "OF_3to4", 1)

            # Steady State : Middle
            for _ in range_(1, 35):
                elems_in = acquire(
                    ObjectFifoPort.Consume, "OF_2to3", 3, T.memref(64, T.ui8())
                ).acquired_elem()
                elem_out = acquire(
                    ObjectFifoPort.Produce, "OF_3to4", 1, T.memref(64, T.ui8())
                ).acquired_elem()
                Call(
                    filter2d_line,
                    [
                        elems_in[0],
                        elems_in[1],
                        elems_in[2],
                        elem_out,
                        arith.constant(64),
                        kernel,
                    ],
                )
                objectFifo_release(ObjectFifoPort.Consume, "OF_2to3", 1)
                objectFifo_release(ObjectFifoPort.Produce, "OF_3to4", 1)
                yield_([])

            # Postamble : Bottom Border
            elems_in_post = acquire(
                ObjectFifoPort.Consume, "OF_2to3", 2, T.memref(64, T.ui8())
            ).acquired_elem()
            elem_post_out = acquire(
                ObjectFifoPort.Produce, "OF_3to4", 1, T.memref(64, T.ui8())
            ).acquired_elem()
            Call(
                filter2d_line,
                [
                    elems_in_post[0],
                    elems_in_post[1],
                    elems_in_post[1],
                    elem_post_out,
                    arith.constant(64),
                    kernel,
                ],
            )
            objectFifo_release(ObjectFifoPort.Consume, "OF_2to3", 2)
            objectFifo_release(ObjectFifoPort.Produce, "OF_3to4", 1)

        @core(T4, "threshold.cc.o")
        def core_body():
            v_thr = arith.constant(10, T.i16())
            v_max = arith.constant(255, T.i16())
            v_typ = arith.constant(0, T.i8())

            for _ in range_(36):
                elem_in = acquire(
                    ObjectFifoPort.Consume, "OF_3to4", 1, T.memref(64, T.ui8())
                ).acquired_elem()
                elem_out = acquire(
                    ObjectFifoPort.Produce, "OF_4to5", 1, T.memref(64, T.ui8())
                ).acquired_elem()

                Call(
                    threshold_line,
                    [elem_in, elem_out, arith.constant(64), v_thr, v_max, v_typ],
                )

                objectFifo_release(ObjectFifoPort.Consume, "OF_3to4", 1)
                objectFifo_release(ObjectFifoPort.Produce, "OF_4to5", 1)
                yield_([])

        @core(T5, "combined_gray2rgba_addWeighted.a")
        def core_body():
            for _ in range_(36):
                elem_in = acquire(
                    ObjectFifoPort.Consume, "OF_4to5", 1, T.memref(64, T.ui8())
                ).acquired_elem()
                elem_out = acquire(
                    ObjectFifoPort.Produce, "OF_5to5", 1, T.memref(256, T.ui8())
                ).acquired_elem()

                Call(gray2rgba_line, [elem_in, elem_out, arith.constant(64)])

                objectFifo_release(ObjectFifoPort.Consume, "OF_4to5", 1)
                objectFifo_release(ObjectFifoPort.Produce, "OF_5to5", 1)

                elem_in1 = acquire(
                    ObjectFifoPort.Consume, "OF_5to5", 1, T.memref(256, T.ui8())
                ).acquired_elem()
                elem_in2 = acquire(
                    ObjectFifoPort.Consume, "inOF_L2L1", 1, T.memref(256, T.ui8())
                ).acquired_elem()
                elem_out2 = acquire(
                    ObjectFifoPort.Produce, "outOF_L1L2", 1, T.memref(256, T.ui8())
                ).acquired_elem()

                alpha = arith.constant(16384, T.i16())
                beta = arith.constant(16384, T.i16())
                gamma = arith.constant(0, T.i8())

                Call(
                    add_weighted_line,
                    [
                        elem_in1,
                        elem_in2,
                        elem_out2,
                        arith.constant(256),
                        alpha,
                        beta,
                        gamma,
                    ],
                )

                objectFifo_release(ObjectFifoPort.Consume, "OF_5to5", 1)
                objectFifo_release(ObjectFifoPort.Consume, "inOF_L2L1", 1)
                objectFifo_release(ObjectFifoPort.Produce, "outOF_L1L2", 1)
                yield_([])

        @FuncOp.from_py_func(
            T.memref(2304, T.i32()), T.memref(2304, T.i32()), T.memref(2304, T.i32())
        )
        def sequence(I, B, O):
            ipu_dma_memcpy_nd(
                metadata="outOF_L2L3",
                bd_id=0,
                mem=O,
                lengths=[1, 1, 36, 64],
                strides=[0, 0, 64],
            )
            ipu_dma_memcpy_nd(
                metadata="inOF_L3L2",
                bd_id=1,
                mem=I,
                lengths=[1, 1, 36, 64],
                strides=[0, 0, 64],
            )
            ipu_sync(column=0, row=0, direction=0, channel=0)


# CHECK-LABEL: my_add_one_objFifo
# module {
#   AIE.device(ipu) {
#     %t00 = AIE.tile(0, 0)
#     %t01 = AIE.tile(0, 1)
#     %t02 = AIE.tile(0, 2)
#
#     AIE.objectFifo @objFifo_in0(%t00, {%t01}, 2 : i32) : !AIE.objectFifo<memref<16xi32>>
#     AIE.objectFifo @objFifo_in1(%t01, {%t02}, 2 : i32) : !AIE.objectFifo<memref<8xi32>>
#     AIE.objectFifo.link [@objFifo_in0] -> [@objFifo_in1] ()
#     AIE.objectFifo @objFifo_out0(%t01, {%t00}, 2 : i32) : !AIE.objectFifo<memref<16xi32>>
#     AIE.objectFifo @objFifo_out1(%t02, {%t01}, 2 : i32) : !AIE.objectFifo<memref<8xi32>>
#     AIE.objectFifo.link [@objFifo_out1] -> [@objFifo_out0] ()
#
#     AIE.core(%t02) {
#       %c8 = arith.constant 8 : index
#       %c0 = arith.constant 0 : index
#       %c1 = arith.constant 1 : index
#       %c1_32 = arith.constant 1 : i32
#
#       scf.for %steps = %c0 to %c8 step %c1 {
#         %subview0 = AIE.objectFifo.acquire @objFifo_in1(Consume, 1) : !AIE.objectFifoSubview<memref<8xi32>>
#         %elem0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<8xi32>> -> memref<8xi32>
#         %subview1 = AIE.objectFifo.acquire @objFifo_out1(Produce, 1) : !AIE.objectFifoSubview<memref<8xi32>>
#         %elem1 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<8xi32>> -> memref<8xi32>
#         scf.for %arg3 = %c0 to %c8 step %c1 {
#             %0 = memref.load %elem0[%arg3] : memref<8xi32>
#             %1 = arith.addi %0, %c1_32 : i32
#             memref.store %1, %elem1[%arg3] : memref<8xi32>
#         }
#         AIE.objectFifo.release @objFifo_in1(Consume, 1)
#         AIE.objectFifo.release @objFifo_out1(Produce, 1)
#       }
#       AIE.end
#     }
#     func.func @sequence(%in : memref<64xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
#       %c0 = arith.constant 0 : i32
#       %c1 = arith.constant 1 : i32
#       %c64 = arith.constant 64 : i32
#       AIEX.ipu.dma_memcpy_nd (%c0, %c0, %out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0]) { metadata = @objFifo_out0, id = 1 : i32 } : (i32, i32, memref<64xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
#       AIEX.ipu.dma_memcpy_nd (%c0, %c0, %in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0]) { metadata = @objFifo_in0, id = 0 : i32 } : (i32, i32, memref<64xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
#       AIEX.ipu.sync { column = 0 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
#       return
#     }
#   }
# }
@construct_and_print_module
def my_add_one_objFifo():
    @device(AIEDevice.ipu)
    def device_body():
        shim_tile = tile(0, 0)
        mem_tile = tile(0, 1)
        compute_tile2 = tile(0, 2)

        objectFifo(
            "in0",
            shim_tile,
            [mem_tile],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(16, T.i32()))),
            [],
            [],
        )
        objectFifo(
            "in1",
            mem_tile,
            [compute_tile2],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(8, T.i32()))),
            [],
            [],
        )
        objectFifo_link(["in0"], ["in1"])
        objectFifo(
            "out0",
            mem_tile,
            [shim_tile],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(8, T.i32()))),
            [],
            [],
        )
        objectFifo(
            "out1",
            compute_tile2,
            [mem_tile],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(16, T.i32()))),
            [],
            [],
        )
        objectFifo_link(["out1"], ["out0"])

        @core(compute_tile2)
        def core_body():
            # Effective while(1)
            for _ in range_(8):
                elem_in = acquire(
                    ObjectFifoPort.Consume, "in1", 1, T.memref(8, T.i32())
                ).acquired_elem()
                elem_out = acquire(
                    ObjectFifoPort.Produce, "out1", 1, T.memref(8, T.i32())
                ).acquired_elem()
                for i in range_(8):
                    v0 = memref.load(elem_in, [i])
                    v1 = arith.addi(v0, arith.constant(1, T.i32()))
                    memref.store(v1, elem_out, [i])
                    yield_([])
                objectFifo_release(ObjectFifoPort.Consume, "in1", 1)
                objectFifo_release(ObjectFifoPort.Produce, "out1", 1)
                yield_([])

        @FuncOp.from_py_func(
            T.memref(64, T.i32()), T.memref(32, T.i32()), T.memref(64, T.i32())
        )
        def sequence(inTensor, notUsed, outTensor):
            ipu_dma_memcpy_nd(
                metadata="out0", bd_id=0, mem=outTensor, lengths=[1, 1, 1, 64]
            )
            ipu_dma_memcpy_nd(
                metadata="in0", bd_id=1, mem=inTensor, lengths=[1, 1, 1, 64]
            )
            ipu_sync(column=0, row=0, direction=0, channel=0)
