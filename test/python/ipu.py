# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

# RUN: %python %s | FileCheck %s

from aie.extras.dialects.ext import memref, arith, func
from aie.extras.runtime.passes import run_pipeline, Pipeline
from aie.extras.util import find_ops, bb


import aie.extras.types as T
from aie.dialects import aie
from aie.dialects.aie import (
    AIEDevice,
    Call,
    CoreOp,
    DMAChannelDir,
    LockAction,
    ObjectFifoPort,
    ObjectFifoType,
    WireBundle,
    acquire,
    buffer,
    core,
    device,
    dma_bd,
    dma_start,
    end as end_,
    external_func,
    flow,
    generate_bcf,
    generate_cdo,
    generate_xaie,
    ipu_instgen,
    lock,
    mem,
    memtile_dma,
    next_bd,
    objectfifo,
    objectfifo_link,
    objectfifo_release,
    tile,
    translate_mlir_to_llvmir,
    use_lock,
)
from aie.dialects.aiex import ipu_sync, ipu_dma_memcpy_nd, ipu_dma_memcpy_nd_
from aie.dialects.func import FuncOp
from aie.dialects.scf import for_
from aie.dialects.scf import yield_
from aie.ir import TypeAttr
from util import construct_and_print_module

range_ = for_

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


# CHECK-LABEL: my_vector_scalar
# CHECK: module {
# CHECK:   aie.device(ipu) {
# CHECK:     func.func private @scale_int32(memref<1024xi32>, memref<1024xi32>)
# CHECK:     %tile_0_0 = aie.tile(0, 0)
# CHECK:     %tile_0_2 = aie.tile(0, 2)
# CHECK:     aie.objectfifo @in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
# CHECK:     aie.objectfifo @out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
# CHECK:     %core_0_2 = aie.core(%tile_0_2) {
# CHECK:       %c4 = arith.constant 4 : index
# CHECK:       %c0 = arith.constant 0 : index
# CHECK:       %c4294967295 = arith.constant 4294967295 : index
# CHECK:       %c1 = arith.constant 1 : index
# CHECK:       scf.for %arg0 = %c0 to %c4294967295 step %c1 {
# CHECK:         scf.for %arg1 = %c0 to %c4 step %c1 {
# CHECK:           %0 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<1024xi32>>
# CHECK:           %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
# CHECK:           %2 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
# CHECK:           %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
# CHECK:           func.call @scale_int32(%3, %1) : (memref<1024xi32>, memref<1024xi32>) -> ()
# CHECK:           aie.objectfifo.release @in(Consume, 1)
# CHECK:           aie.objectfifo.release @out(Produce, 1)
# CHECK:         }
# CHECK:       }
# CHECK:       aie.end
# CHECK:     } {link_with = "scale.o"}
# CHECK:     func.func @sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
# CHECK:       aiex.ipu.dma_memcpy_nd(0, 0, %arg2 : memref<4096xi32>) {id = 0 : i32, lengths = array<i32: 1, 1, 1, 4096>, metadata = @out, offsets = array<i32: 0, 0, 0, 0>, strides = array<i32: 0, 0, 0>}
# CHECK:       aiex.ipu.dma_memcpy_nd(0, 0, %arg0 : memref<4096xi32>) {id = 1 : i32, lengths = array<i32: 1, 1, 1, 4096>, metadata = @in, offsets = array<i32: 0, 0, 0, 0>, strides = array<i32: 0, 0, 0>}
# CHECK:       aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
# CHECK:       return
# CHECK:     }
# CHECK:   }
# CHECK: }


@construct_and_print_module
def my_vector_scalar(module):
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

        objectfifo(
            "in",
            S,
            [M],
            buffer_depth,
            TypeAttr.get(ObjectFifoType.get(T.memref(n, T.i32()))),
            [],
            [],
        )
        objectfifo(
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
                    objectfifo_release(ObjectFifoPort.Consume, "in", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "out", 1)
                    yield_([])
                yield_([])

        @FuncOp.from_py_func(
            T.memref(N, T.i32()), T.memref(N, T.i32()), T.memref(N, T.i32())
        )
        def sequence(A, B, C):
            ipu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, lengths=[1, 1, 1, N])
            ipu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, lengths=[1, 1, 1, N])
            ipu_sync(column=0, row=0, direction=0, channel=0)

    print(run_pipeline(module, "builtin.module(canonicalize)"))


# CHECK-LABEL: my_matmul
# CHECK: module {
# CHECK:   aie.device(ipu) {
# CHECK:     func.func private @zero_scalar_i16(memref<64x64xi16>)
# CHECK:     func.func private @zero_i16(memref<64x64xi16>)
# CHECK:     func.func private @matmul_scalar_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
# CHECK:     func.func private @matmul_i16_i16(memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>)
# CHECK:     %tile_0_0 = aie.tile(0, 0)
# CHECK:     %tile_0_2 = aie.tile(0, 2)
# CHECK:     aie.objectfifo @inA(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x32xi16>>
# CHECK:     aie.objectfifo @inB(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xi16>>
# CHECK:     aie.objectfifo @outC(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64x64xi16>>
# CHECK:     %core_0_2 = aie.core(%tile_0_2) {
# CHECK:       %c4 = arith.constant 4 : index
# CHECK:       %c0 = arith.constant 0 : index
# CHECK:       %c4294967295 = arith.constant 4294967295 : index
# CHECK:       %c1 = arith.constant 1 : index
# CHECK:       scf.for %arg0 = %c0 to %c4294967295 step %c1 {
# CHECK:         scf.for %arg1 = %c0 to %c4 step %c1 {
# CHECK:           %0 = aie.objectfifo.acquire @outC(Produce, 1) : !aie.objectfifosubview<memref<64x64xi16>>
# CHECK:           %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xi16>> -> memref<64x64xi16>
# CHECK:           func.call @zero_i16(%1) : (memref<64x64xi16>) -> ()
# CHECK:           scf.for %arg2 = %c0 to %c4 step %c1 {
# CHECK:             %2 = aie.objectfifo.acquire @inA(Consume, 1) : !aie.objectfifosubview<memref<64x32xi16>>
# CHECK:             %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x32xi16>> -> memref<64x32xi16>
# CHECK:             %4 = aie.objectfifo.acquire @inB(Consume, 1) : !aie.objectfifosubview<memref<32x64xi16>>
# CHECK:             %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xi16>> -> memref<32x64xi16>
# CHECK:             func.call @matmul_i16_i16(%3, %5, %1) : (memref<64x32xi16>, memref<32x64xi16>, memref<64x64xi16>) -> ()
# CHECK:             aie.objectfifo.release @inA(Consume, 1)
# CHECK:             aie.objectfifo.release @inB(Consume, 1)
# CHECK:           }
# CHECK:           aie.objectfifo.release @outC(Produce, 1)
# CHECK:         }
# CHECK:       }
# CHECK:       aie.end
# CHECK:     } {link_with = "mm.o"}
# CHECK:     func.func @sequence(%arg0: memref<8192xi32>, %arg1: memref<8192xi32>, %arg2: memref<8192xi32>) {
# CHECK:       aiex.ipu.dma_memcpy_nd(0, 0, %arg2 : memref<8192xi32>) {id = 0 : i32, lengths = array<i32: 2, 2, 64, 32>, metadata = @outC, offsets = array<i32: 0, 0, 0, 0>, strides = array<i32: 4096, 32, 64>}
# CHECK:       aiex.ipu.dma_memcpy_nd(0, 0, %arg0 : memref<8192xi32>) {id = 1 : i32, lengths = array<i32: 2, 4, 64, 16>, metadata = @inA, offsets = array<i32: 0, 0, 0, 0>, strides = array<i32: 0, 16, 64>}
# CHECK:       aiex.ipu.dma_memcpy_nd(0, 0, %arg1 : memref<8192xi32>) {id = 2 : i32, lengths = array<i32: 2, 4, 32, 32>, metadata = @inB, offsets = array<i32: 0, 0, 0, 0>, strides = array<i32: 32, 2048, 64>}
# CHECK:       aiex.ipu.dma_memcpy_nd(0, 0, %arg0 : memref<8192xi32>) {id = 3 : i32, lengths = array<i32: 2, 4, 64, 16>, metadata = @inA, offsets = array<i32: 0, 0, 0, 4096>, strides = array<i32: 0, 16, 64>}
# CHECK:       aiex.ipu.dma_memcpy_nd(0, 0, %arg1 : memref<8192xi32>) {id = 4 : i32, lengths = array<i32: 2, 4, 32, 32>, metadata = @inB, offsets = array<i32: 0, 0, 0, 0>, strides = array<i32: 32, 2048, 64>}
# CHECK:       aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
# CHECK:       return
# CHECK:     }
# CHECK:   }
# CHECK: }


@construct_and_print_module
def my_matmul(module):
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

        objectfifo(
            "inA",
            S,
            [M],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(m, k, T.i16()))),
            [],
            [],
        )
        objectfifo(
            "inB",
            S,
            [M],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(k, n, T.i16()))),
            [],
            [],
        )
        objectfifo(
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
                        objectfifo_release(ObjectFifoPort.Consume, "inA", 1)
                        objectfifo_release(ObjectFifoPort.Consume, "inB", 1)
                        yield_([])

                    objectfifo_release(ObjectFifoPort.Produce, "outC", 1)
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

    print(run_pipeline(module, "builtin.module(canonicalize)"))


# CHECK-LABEL: edge_detect
# CHECK: module {
# CHECK:   aie.device(ipu) {
# CHECK:     func.func private @rgba2gray_line(memref<256xui8>, memref<64xui8>, i32)
# CHECK:     func.func private @filter2d_line(memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>)
# CHECK:     func.func private @threshold_line(memref<64xui8>, memref<64xui8>, i32, i16, i16, i8)
# CHECK:     func.func private @gray2rgba_line(memref<64xui8>, memref<256xui8>, i32)
# CHECK:     func.func private @add_weighted_line(memref<256xui8>, memref<256xui8>, memref<256xui8>, i32, i16, i16, i8)
# CHECK:     %tile_0_0 = aie.tile(0, 0)
# CHECK:     %tile_0_1 = aie.tile(0, 1)
# CHECK:     %tile_0_2 = aie.tile(0, 2)
# CHECK:     %tile_0_3 = aie.tile(0, 3)
# CHECK:     %tile_0_4 = aie.tile(0, 4)
# CHECK:     %tile_0_5 = aie.tile(0, 5)
# CHECK:     aie.objectfifo @inOF_L3L2(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<256xui8>>
# CHECK:     aie.objectfifo @inOF_L2L1(%tile_0_1, {%tile_0_2, %tile_0_5}, [2 : i32, 2 : i32, 7 : i32]) : !aie.objectfifo<memref<256xui8>>
# CHECK:     aie.objectfifo.link [@inOF_L3L2] -> [@inOF_L2L1]()
# CHECK:     aie.objectfifo @outOF_L2L3(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<256xui8>>
# CHECK:     aie.objectfifo @outOF_L1L2(%tile_0_5, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<256xui8>>
# CHECK:     aie.objectfifo.link [@outOF_L1L2] -> [@outOF_L2L3]()
# CHECK:     aie.objectfifo @OF_2to3(%tile_0_2, {%tile_0_3}, 4 : i32) : !aie.objectfifo<memref<64xui8>>
# CHECK:     aie.objectfifo @OF_3to4(%tile_0_3, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<64xui8>>
# CHECK:     aie.objectfifo @OF_4to5(%tile_0_4, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<64xui8>>
# CHECK:     aie.objectfifo @OF_5to5(%tile_0_5, {%tile_0_5}, 1 : i32) : !aie.objectfifo<memref<256xui8>>
# CHECK:     %core_0_2 = aie.core(%tile_0_2) {
# CHECK:       %c64_i32 = arith.constant 64 : i32
# CHECK:       %c0 = arith.constant 0 : index
# CHECK:       %c36 = arith.constant 36 : index
# CHECK:       %c1 = arith.constant 1 : index
# CHECK:       scf.for %arg0 = %c0 to %c36 step %c1 {
# CHECK:         %0 = aie.objectfifo.acquire @inOF_L2L1(Consume, 1) : !aie.objectfifosubview<memref<256xui8>>
# CHECK:         %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<256xui8>> -> memref<256xui8>
# CHECK:         %2 = aie.objectfifo.acquire @OF_2to3(Produce, 1) : !aie.objectfifosubview<memref<64xui8>>
# CHECK:         %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         func.call @rgba2gray_line(%1, %3, %c64_i32) : (memref<256xui8>, memref<64xui8>, i32) -> ()
# CHECK:         aie.objectfifo.release @inOF_L2L1(Consume, 1)
# CHECK:         aie.objectfifo.release @OF_2to3(Produce, 1)
# CHECK:       }
# CHECK:       aie.end
# CHECK:     } {link_with = "rgba2gray.cc.o"}
# CHECK:     %core_0_3 = aie.core(%tile_0_3) {
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
# CHECK:       %0 = aie.objectfifo.acquire @OF_2to3(Consume, 2) : !aie.objectfifosubview<memref<64xui8>>
# CHECK:       %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
# CHECK:       %2 = aie.objectfifo.subview.access %0[1] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
# CHECK:       %3 = aie.objectfifo.acquire @OF_3to4(Produce, 1) : !aie.objectfifosubview<memref<64xui8>>
# CHECK:       %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
# CHECK:       func.call @filter2d_line(%1, %1, %2, %4, %c64_i32, %alloc) : (memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>) -> ()
# CHECK:       aie.objectfifo.release @OF_3to4(Produce, 1)
# CHECK:       scf.for %arg0 = %c1 to %c35 step %c1 {
# CHECK:         %10 = aie.objectfifo.acquire @OF_2to3(Consume, 3) : !aie.objectfifosubview<memref<64xui8>>
# CHECK:         %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         %12 = aie.objectfifo.subview.access %10[1] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         %13 = aie.objectfifo.subview.access %10[2] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         %14 = aie.objectfifo.acquire @OF_3to4(Produce, 1) : !aie.objectfifosubview<memref<64xui8>>
# CHECK:         %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         func.call @filter2d_line(%11, %12, %13, %15, %c64_i32, %alloc) : (memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>) -> ()
# CHECK:         aie.objectfifo.release @OF_2to3(Consume, 1)
# CHECK:         aie.objectfifo.release @OF_3to4(Produce, 1)
# CHECK:       }
# CHECK:       %5 = aie.objectfifo.acquire @OF_2to3(Consume, 2) : !aie.objectfifosubview<memref<64xui8>>
# CHECK:       %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
# CHECK:       %7 = aie.objectfifo.subview.access %5[1] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
# CHECK:       %8 = aie.objectfifo.acquire @OF_3to4(Produce, 1) : !aie.objectfifosubview<memref<64xui8>>
# CHECK:       %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
# CHECK:       func.call @filter2d_line(%6, %7, %7, %9, %c64_i32, %alloc) : (memref<64xui8>, memref<64xui8>, memref<64xui8>, memref<64xui8>, i32, memref<3x3xi16>) -> ()
# CHECK:       aie.objectfifo.release @OF_2to3(Consume, 2)
# CHECK:       aie.objectfifo.release @OF_3to4(Produce, 1)
# CHECK:       aie.end
# CHECK:     } {link_with = "filter2d.cc.o"}
# CHECK:     %core_0_4 = aie.core(%tile_0_4) {
# CHECK:       %c64_i32 = arith.constant 64 : i32
# CHECK:       %c10_i16 = arith.constant 10 : i16
# CHECK:       %c255_i16 = arith.constant 255 : i16
# CHECK:       %c0_i8 = arith.constant 0 : i8
# CHECK:       %c0 = arith.constant 0 : index
# CHECK:       %c36 = arith.constant 36 : index
# CHECK:       %c1 = arith.constant 1 : index
# CHECK:       scf.for %arg0 = %c0 to %c36 step %c1 {
# CHECK:         %0 = aie.objectfifo.acquire @OF_3to4(Consume, 1) : !aie.objectfifosubview<memref<64xui8>>
# CHECK:         %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         %2 = aie.objectfifo.acquire @OF_4to5(Produce, 1) : !aie.objectfifosubview<memref<64xui8>>
# CHECK:         %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         func.call @threshold_line(%1, %3, %c64_i32, %c10_i16, %c255_i16, %c0_i8) : (memref<64xui8>, memref<64xui8>, i32, i16, i16, i8) -> ()
# CHECK:         aie.objectfifo.release @OF_3to4(Consume, 1)
# CHECK:         aie.objectfifo.release @OF_4to5(Produce, 1)
# CHECK:       }
# CHECK:       aie.end
# CHECK:     } {link_with = "threshold.cc.o"}
# CHECK:     %core_0_5 = aie.core(%tile_0_5) {
# CHECK:       %c256_i32 = arith.constant 256 : i32
# CHECK:       %c0_i8 = arith.constant 0 : i8
# CHECK:       %c16384_i16 = arith.constant 16384 : i16
# CHECK:       %c64_i32 = arith.constant 64 : i32
# CHECK:       %c0 = arith.constant 0 : index
# CHECK:       %c36 = arith.constant 36 : index
# CHECK:       %c1 = arith.constant 1 : index
# CHECK:       scf.for %arg0 = %c0 to %c36 step %c1 {
# CHECK:         %0 = aie.objectfifo.acquire @OF_4to5(Consume, 1) : !aie.objectfifosubview<memref<64xui8>>
# CHECK:         %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64xui8>> -> memref<64xui8>
# CHECK:         %2 = aie.objectfifo.acquire @OF_5to5(Produce, 1) : !aie.objectfifosubview<memref<256xui8>>
# CHECK:         %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<256xui8>> -> memref<256xui8>
# CHECK:         func.call @gray2rgba_line(%1, %3, %c64_i32) : (memref<64xui8>, memref<256xui8>, i32) -> ()
# CHECK:         aie.objectfifo.release @OF_4to5(Consume, 1)
# CHECK:         aie.objectfifo.release @OF_5to5(Produce, 1)
# CHECK:         %4 = aie.objectfifo.acquire @OF_5to5(Consume, 1) : !aie.objectfifosubview<memref<256xui8>>
# CHECK:         %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<256xui8>> -> memref<256xui8>
# CHECK:         %6 = aie.objectfifo.acquire @inOF_L2L1(Consume, 1) : !aie.objectfifosubview<memref<256xui8>>
# CHECK:         %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<256xui8>> -> memref<256xui8>
# CHECK:         %8 = aie.objectfifo.acquire @outOF_L1L2(Produce, 1) : !aie.objectfifosubview<memref<256xui8>>
# CHECK:         %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<256xui8>> -> memref<256xui8>
# CHECK:         func.call @add_weighted_line(%5, %7, %9, %c256_i32, %c16384_i16, %c16384_i16, %c0_i8) : (memref<256xui8>, memref<256xui8>, memref<256xui8>, i32, i16, i16, i8) -> ()
# CHECK:         aie.objectfifo.release @OF_5to5(Consume, 1)
# CHECK:         aie.objectfifo.release @inOF_L2L1(Consume, 1)
# CHECK:         aie.objectfifo.release @outOF_L1L2(Produce, 1)
# CHECK:       }
# CHECK:       aie.end
# CHECK:     } {link_with = "combined_gray2rgba_addWeighted.a"}
# CHECK:     func.func @sequence(%arg0: memref<2304xi32>, %arg1: memref<2304xi32>, %arg2: memref<2304xi32>) {
# CHECK:       aiex.ipu.dma_memcpy_nd(0, 0, %arg2 : memref<2304xi32>) {id = 0 : i32, lengths = array<i32: 1, 1, 36, 64>, metadata = @outOF_L2L3, offsets = array<i32: 0, 0, 0, 0>, strides = array<i32: 0, 0, 64>}
# CHECK:       aiex.ipu.dma_memcpy_nd(0, 0, %arg0 : memref<2304xi32>) {id = 1 : i32, lengths = array<i32: 1, 1, 36, 64>, metadata = @inOF_L3L2, offsets = array<i32: 0, 0, 0, 0>, strides = array<i32: 0, 0, 64>}
# CHECK:       aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
# CHECK:       return
# CHECK:     }
# CHECK:   }
# CHECK: }


@construct_and_print_module
def edge_detect(module):
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

        objectfifo(
            "inOF_L3L2",
            S,
            [M],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(256, T.ui8()))),
            [],
            [],
        )
        objectfifo(
            "inOF_L2L1",
            M,
            [T2, T5],
            [2, 2, 7],
            TypeAttr.get(ObjectFifoType.get(T.memref(256, T.ui8()))),
            [],
            [],
        )
        objectfifo_link(["inOF_L3L2"], ["inOF_L2L1"])

        objectfifo(
            "outOF_L2L3",
            M,
            [S],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(256, T.ui8()))),
            [],
            [],
        )
        objectfifo(
            "outOF_L1L2",
            T5,
            [M],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(256, T.ui8()))),
            [],
            [],
        )
        objectfifo_link(["outOF_L1L2"], ["outOF_L2L3"])

        objectfifo(
            "OF_2to3",
            T2,
            [T3],
            4,
            TypeAttr.get(ObjectFifoType.get(T.memref(64, T.ui8()))),
            [],
            [],
        )
        objectfifo(
            "OF_3to4",
            T3,
            [T4],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(64, T.ui8()))),
            [],
            [],
        )
        objectfifo(
            "OF_4to5",
            T4,
            [T5],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(64, T.ui8()))),
            [],
            [],
        )
        objectfifo(
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

                objectfifo_release(ObjectFifoPort.Consume, "inOF_L2L1", 1)
                objectfifo_release(ObjectFifoPort.Produce, "OF_2to3", 1)
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
            objectfifo_release(ObjectFifoPort.Produce, "OF_3to4", 1)

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
                objectfifo_release(ObjectFifoPort.Consume, "OF_2to3", 1)
                objectfifo_release(ObjectFifoPort.Produce, "OF_3to4", 1)
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
            objectfifo_release(ObjectFifoPort.Consume, "OF_2to3", 2)
            objectfifo_release(ObjectFifoPort.Produce, "OF_3to4", 1)

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

                objectfifo_release(ObjectFifoPort.Consume, "OF_3to4", 1)
                objectfifo_release(ObjectFifoPort.Produce, "OF_4to5", 1)
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

                objectfifo_release(ObjectFifoPort.Consume, "OF_4to5", 1)
                objectfifo_release(ObjectFifoPort.Produce, "OF_5to5", 1)

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

                objectfifo_release(ObjectFifoPort.Consume, "OF_5to5", 1)
                objectfifo_release(ObjectFifoPort.Consume, "inOF_L2L1", 1)
                objectfifo_release(ObjectFifoPort.Produce, "outOF_L1L2", 1)
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

    print(run_pipeline(module, "builtin.module(canonicalize)"))


# CHECK-LABEL: my_add_one_objFifo
# module {
#   aie.device(ipu) {
#     %t00 = aie.tile(0, 0)
#     %t01 = aie.tile(0, 1)
#     %t02 = aie.tile(0, 2)
#
#     aie.objectfifo @objFifo_in0(%t00, {%t01}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
#     aie.objectfifo @objFifo_in1(%t01, {%t02}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
#     aie.objectfifo.link [@objFifo_in0] -> [@objFifo_in1] ()
#     aie.objectfifo @objFifo_out0(%t01, {%t00}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
#     aie.objectfifo @objFifo_out1(%t02, {%t01}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
#     aie.objectfifo.link [@objFifo_out1] -> [@objFifo_out0] ()
#
#     aie.core(%t02) {
#       %c8 = arith.constant 8 : index
#       %c0 = arith.constant 0 : index
#       %c1 = arith.constant 1 : index
#       %c1_32 = arith.constant 1 : i32
#
#       scf.for %steps = %c0 to %c8 step %c1 {
#         %subview0 = aie.objectfifo.acquire @objFifo_in1(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
#         %elem0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
#         %subview1 = aie.objectfifo.acquire @objFifo_out1(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
#         %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
#         scf.for %arg3 = %c0 to %c8 step %c1 {
#             %0 = memref.load %elem0[%arg3] : memref<8xi32>
#             %1 = arith.addi %0, %c1_32 : i32
#             memref.store %1, %elem1[%arg3] : memref<8xi32>
#         }
#         aie.objectfifo.release @objFifo_in1(Consume, 1)
#         aie.objectfifo.release @objFifo_out1(Produce, 1)
#       }
#       aie.end
#     }
#     func.func @sequence(%in : memref<64xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
#       aiex.ipu.dma_memcpy_nd(0, 0, %arg2 : memref<64xi32>) {id = 0 : i32, lengths = array<i32: 1, 1, 1, 64>, metadata = @out0, offsets = array<i32: 0, 0, 0, 0>, strides = array<i32: 0, 0, 0>}
#       aiex.ipu.dma_memcpy_nd(0, 0, %arg0 : memref<64xi32>) {id = 1 : i32, lengths = array<i32: 1, 1, 1, 64>, metadata = @in0, offsets = array<i32: 0, 0, 0, 0>, strides = array<i32: 0, 0, 0>}
#       aiex.ipu.sync { column = 0 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
#       return
#     }
#   }
# }
@construct_and_print_module
def my_add_one_objFifo(module):
    @device(AIEDevice.ipu)
    def device_body():
        shim_tile = tile(0, 0)
        mem_tile = tile(0, 1)
        compute_tile2 = tile(0, 2)

        objectfifo(
            "in0",
            shim_tile,
            [mem_tile],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(16, T.i32()))),
            [],
            [],
        )
        objectfifo(
            "in1",
            mem_tile,
            [compute_tile2],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(8, T.i32()))),
            [],
            [],
        )
        objectfifo_link(["in0"], ["in1"])
        objectfifo(
            "out0",
            mem_tile,
            [shim_tile],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(8, T.i32()))),
            [],
            [],
        )
        objectfifo(
            "out1",
            compute_tile2,
            [mem_tile],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(16, T.i32()))),
            [],
            [],
        )
        objectfifo_link(["out1"], ["out0"])

        @core(compute_tile2)
        def core_body():
            # Effective while(1)
            for _ in range_(8):
                elem_in = acquire(
                    ObjectFifoPort.Consume, "in1", 1, T.memref(8, T.i32())
                ).acquired_elem()
                elem_out = acquire(
                    ObjectFifoPort.Produce, "out1", 1, T.memref(16, T.i32())
                ).acquired_elem()
                for i in range_(8):
                    v0 = memref.load(elem_in, [i])
                    v1 = arith.addi(v0, arith.constant(1, T.i32()))
                    memref.store(v1, elem_out, [i])
                    yield_([])
                objectfifo_release(ObjectFifoPort.Consume, "in1", 1)
                objectfifo_release(ObjectFifoPort.Produce, "out1", 1)
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

    print(run_pipeline(module, "builtin.module(canonicalize)"))


@construct_and_print_module
def my_passthrough(module):
    N = 4096
    ofifo_mem_ref_ty = TypeAttr.get(ObjectFifoType.get(T.memref(1024, T.i32())))
    tensor_ty = T.memref(N, T.i32())

    @device(AIEDevice.ipu)
    def device_body():
        # Tile declarations
        shim_tile = tile(0, 0)
        compute_tile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        objectfifo("in", shim_tile, [compute_tile2], 2, ofifo_mem_ref_ty, [], [])
        objectfifo("out", compute_tile2, [shim_tile], 2, ofifo_mem_ref_ty, [], [])
        objectfifo_link(["in"], ["out"])

        @core(compute_tile2)
        def core_body():
            tmp = memref.alloc([1], T.i32())
            v0 = arith.constant(0, T.i32())
            memref.store(v0, tmp, [0])

        @func.func(emit=True)
        def sequence(A: tensor_ty, B: tensor_ty, C: tensor_ty):
            ipu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, lengths=[1, 1, 1, N])
            ipu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, lengths=[1, 1, 1, N])
            ipu_sync(column=0, row=0, direction=0, channel=0)

    pass_pipeline = ",".join(
        [
            "lower-affine",
            "aie-canonicalize-device",
            "aie.device(" + "aie-assign-lock-ids",
            "aie-register-objectFifos",
            "aie-objectFifo-stateful-transform",
            "aie-lower-broadcast-packet",
            "aie-create-packet-flows",
            "aie-lower-multicast",
            "aie-assign-buffer-addresses)",
            "convert-scf-to-cf",
        ]
    )
    input_with_addresses = run_pipeline(module, "builtin.module(" + pass_pipeline + ")")
    # print(module)
    cores = [
        (c.tile.owner.opview.col.value, c.tile.owner.opview.row.value, None)
        for c in find_ops(
            input_with_addresses.operation,
            lambda o: isinstance(o.operation.opview, CoreOp),
        )
    ]

    generated_ipu_insts = run_pipeline(
        input_with_addresses, "builtin.module(aie.device(aie-dma-to-ipu))"
    )
    ipu_insts = ipu_instgen(generated_ipu_insts.operation)
    # CHECK: 00000011
    # CHECK: 01000405
    # CHECK: 01000100
    # CHECK: 0B590100
    # CHECK: 000055FF
    # CHECK: 00000001
    # CHECK: 00000010
    # CHECK: 314E5A5F
    # CHECK: 635F5F31
    # CHECK: 676E696C
    # CHECK: 39354E5F
    # CHECK: 6E693131
    # CHECK: 5F727473
    # CHECK: 64726F77
    # CHECK: 00004573
    # CHECK: 07BD9630
    # CHECK: 000055FF
    # CHECK: 06000120
    # CHECK: 00000000
    # CHECK: 00001000
    # CHECK: 00000000
    # CHECK: 00000000
    # CHECK: 00000000
    # CHECK: 80000000
    # CHECK: 00000000
    # CHECK: 00000000
    # CHECK: 02000000
    # CHECK: 02000000
    # CHECK: 0001D204
    # CHECK: 80000000
    # CHECK: 06000101
    # CHECK: 00000000
    # CHECK: 00001000
    # CHECK: 00000000
    # CHECK: 00000000
    # CHECK: 00000000
    # CHECK: 80000000
    # CHECK: 00000000
    # CHECK: 00000000
    # CHECK: 02000000
    # CHECK: 02000000
    # CHECK: 0001D214
    # CHECK: 00000001
    # CHECK: 03000000
    # CHECK: 00010100
    print(ipu_insts)

    pass_pipeline = ",".join(
        [
            "aie.device(aie-localize-locks",
            "aie-normalize-address-spaces)",
            "aie-standard-lowering{ tilecol=%d tilerow=%d }" % cores[0][0:2],
            "aiex-standard-lowering",
        ]
    )
    input_opt_with_addresses = run_pipeline(
        input_with_addresses, "builtin.module(" + pass_pipeline + ")"
    )
    pass_pipeline = ",".join(
        [
            "canonicalize",
            "cse",
            "convert-vector-to-llvm",
            "expand-strided-metadata",
            "lower-affine",
            "convert-math-to-llvm",
            "convert-arith-to-llvm",
            "finalize-memref-to-llvm",
            "convert-func-to-llvm{use-bare-ptr-memref-call-conv}",
            "convert-cf-to-llvm",
            "canonicalize",
            "cse",
        ]
    )
    llvmlir = run_pipeline(
        input_opt_with_addresses, "builtin.module(" + pass_pipeline + ")"
    )
    llvmir = translate_mlir_to_llvmir(llvmlir.operation)
    # CHECK: ; ModuleID = 'LLVMDialectModule'
    # CHECK: source_filename = "LLVMDialectModule"
    # CHECK: target triple = "aie2"
    # CHECK: @in_cons_buff_1 = external global [1024 x i32]
    # CHECK: @in_cons_buff_0 = external global [1024 x i32]
    # CHECK: @out_cons = external global [1024 x i32]
    # CHECK: @out = external global [1024 x i32]
    # CHECK: @in_cons = external global [1024 x i32]
    # CHECK: @in = external global [1024 x i32]
    # CHECK: declare void @debug_i32(i32)
    # CHECK: declare void @llvm.aie2.put.ms(i32, i32)
    # CHECK: declare { i32, i32 } @llvm.aie2.get.ss()
    # CHECK: declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)
    # CHECK: declare <16 x i32> @llvm.aie2.scd.read.vec(i32)
    # CHECK: declare void @llvm.aie2.acquire(i32, i32)
    # CHECK: declare void @llvm.aie2.release(i32, i32)
    # CHECK: define void @sequence(ptr %0, ptr %1, ptr %2) {
    # CHECK:   ret void
    # CHECK: }
    # CHECK: define void @core_0_2() {
    # CHECK:   ret void
    # CHECK: }
    # CHECK: !llvm.module.flags = !{!0}
    # CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
    print(llvmir)

    pass_pipeline = ",".join(
        [
            "aie-create-pathfinder-flows",
            "aie-lower-broadcast-packet",
            "aie-create-packet-flows",
            "aie-lower-multicast",
        ]
    )
    input_physical = run_pipeline(
        input_with_addresses, "builtin.module(aie.device(" + pass_pipeline + "))"
    )

    aie_inc = generate_xaie(input_physical.operation)
    # CHECK: #ifndef MLIR_AIE_QUIET
    # CHECK: #define __mlir_aie_verbose(x) x
    # CHECK: #else
    # CHECK: #define __mlir_aie_verbose(x)
    # CHECK: #endif
    #
    # CHECK: // The following is a wrapper for the common "if(call() != 0) return 1" pattern.
    # CHECK: // Use this only in functions that return int. If the call this wrapper is used
    # CHECK: // on does not succeed, the expanded code will exit out of the function
    # CHECK: // containing this macro with an error code.
    # CHECK: #define __mlir_aie_try(x) do { \
    # CHECK:   AieRC ret = (x); \
    # CHECK:   if(ret != XAIE_OK) { \
    # CHECK:     return x; \
    # CHECK:   } \
    # CHECK: } while(0)
    #
    # CHECK: static XAie_DmaDimDesc *__mlir_aie_alloc_dim_desc(size_t ndims) {
    # CHECK:   XAie_DmaDimDesc *ret = NULL;
    # CHECK:   ret = (XAie_DmaDimDesc *)calloc(sizeof(XAie_DmaDimDesc), ndims);
    # CHECK:   if(NULL == ret) {
    # CHECK:     __mlir_aie_verbose(fprintf(stderr, "Allocating DmaDimDesc failed.\n"));
    # CHECK:   }
    # CHECK:   return ret;
    # CHECK: }
    #
    # CHECK: aie_libxaie_ctx_t* mlir_aie_init_libxaie() {
    # CHECK:   aie_libxaie_ctx_t *ctx = new aie_libxaie_ctx_t;
    # CHECK:   if (!ctx)
    # CHECK:     return 0;
    # CHECK:   ctx->AieConfigPtr.AieGen = XAIE_DEV_GEN_AIEML;
    # CHECK:   ctx->AieConfigPtr.BaseAddr = 0x20000000000;
    # CHECK:   ctx->AieConfigPtr.ColShift = 25;
    # CHECK:   ctx->AieConfigPtr.RowShift = 20;
    # CHECK:   ctx->AieConfigPtr.NumRows = 6;
    # CHECK:   ctx->AieConfigPtr.NumCols = 5;
    # CHECK:   ctx->AieConfigPtr.ShimRowNum = 0;
    # CHECK:   ctx->AieConfigPtr.MemTileRowStart = 1;
    # CHECK:   ctx->AieConfigPtr.MemTileNumRows = 1;
    # CHECK:   //  ctx->AieConfigPtr.ReservedRowStart = XAIE_RES_TILE_ROW_START;
    # CHECK:   //  ctx->AieConfigPtr.ReservedNumRows  = XAIE_RES_TILE_NUM_ROWS;
    # CHECK:   ctx->AieConfigPtr.AieTileRowStart = 2;
    # CHECK:   ctx->AieConfigPtr.AieTileNumRows = 4;
    # CHECK:   ctx->AieConfigPtr.PartProp = {0};
    # CHECK:   ctx->DevInst = {0};
    # CHECK:   return ctx;
    # CHECK: }
    #
    # CHECK: int mlir_aie_configure_cores(aie_libxaie_ctx_t* ctx) {
    # CHECK: __mlir_aie_try(XAie_CoreReset(&(ctx->DevInst), XAie_TileLoc(0,2)));
    # CHECK: __mlir_aie_try(XAie_CoreDisable(&(ctx->DevInst), XAie_TileLoc(0,2)));
    # CHECK: for (int l = 0; l < 16; ++l)
    # CHECK:   __mlir_aie_try(XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(0,2), XAie_LockInit(l, 0x0), 0));
    # CHECK: {
    # CHECK: AieRC RC = XAie_LoadElf(&(ctx->DevInst), XAie_TileLoc(0,2), (const char*)"core_0_2.elf",0);
    # CHECK: if (RC != XAIE_OK)
    # CHECK:     __mlir_aie_verbose(fprintf(stderr, "Failed to load elf for Core[%d,%d], ret is %d\n", 0, 2, RC));
    # CHECK: assert(RC == XAIE_OK);
    # CHECK: }
    # CHECK: return XAIE_OK;
    # CHECK: } // mlir_aie_configure_cores
    #
    # CHECK: int mlir_aie_start_cores(aie_libxaie_ctx_t* ctx) {
    # CHECK: __mlir_aie_try(XAie_CoreUnreset(&(ctx->DevInst), XAie_TileLoc(0,2)));
    # CHECK: __mlir_aie_try(XAie_CoreEnable(&(ctx->DevInst), XAie_TileLoc(0,2)));
    # CHECK: return XAIE_OK;
    # CHECK: } // mlir_aie_start_cores
    #
    # CHECK: int mlir_aie_configure_dmas(aie_libxaie_ctx_t* ctx) {
    # CHECK: XAie_DmaDesc dma_tile02_bd0;
    # CHECK: __mlir_aie_try(XAie_DmaDescInit(&(ctx->DevInst), &(dma_tile02_bd0), XAie_TileLoc(0,2)));
    # CHECK: __mlir_aie_try(XAie_DmaSetLock(&(dma_tile02_bd0), XAie_LockInit(0,-1),XAie_LockInit(1,1)));
    # CHECK: __mlir_aie_try(XAie_DmaSetAddrLen(&(dma_tile02_bd0), /* addrA */ 0x400,  /* len */ 1024 * 4));
    # CHECK: __mlir_aie_try(XAie_DmaSetNextBd(&(dma_tile02_bd0),  /* nextbd */ 1,  /* enableNextBd */ 1));
    # CHECK: __mlir_aie_try(XAie_DmaEnableBd(&(dma_tile02_bd0)));
    # CHECK: __mlir_aie_try(XAie_DmaWriteBd(&(ctx->DevInst), &(dma_tile02_bd0), XAie_TileLoc(0,2),  /* bd */ 0));
    # CHECK: XAie_DmaDesc dma_tile02_bd1;
    # CHECK: __mlir_aie_try(XAie_DmaDescInit(&(ctx->DevInst), &(dma_tile02_bd1), XAie_TileLoc(0,2)));
    # CHECK: __mlir_aie_try(XAie_DmaSetLock(&(dma_tile02_bd1), XAie_LockInit(0,-1),XAie_LockInit(1,1)));
    # CHECK: __mlir_aie_try(XAie_DmaSetAddrLen(&(dma_tile02_bd1), /* addrA */ 0x1400,  /* len */ 1024 * 4));
    # CHECK: __mlir_aie_try(XAie_DmaSetNextBd(&(dma_tile02_bd1),  /* nextbd */ 0,  /* enableNextBd */ 1));
    # CHECK: __mlir_aie_try(XAie_DmaEnableBd(&(dma_tile02_bd1)));
    # CHECK: __mlir_aie_try(XAie_DmaWriteBd(&(ctx->DevInst), &(dma_tile02_bd1), XAie_TileLoc(0,2),  /* bd */ 1));
    # CHECK: XAie_DmaDesc dma_tile02_bd2;
    # CHECK: __mlir_aie_try(XAie_DmaDescInit(&(ctx->DevInst), &(dma_tile02_bd2), XAie_TileLoc(0,2)));
    # CHECK: __mlir_aie_try(XAie_DmaSetLock(&(dma_tile02_bd2), XAie_LockInit(1,-1),XAie_LockInit(0,1)));
    # CHECK: __mlir_aie_try(XAie_DmaSetAddrLen(&(dma_tile02_bd2), /* addrA */ 0x400,  /* len */ 1024 * 4));
    # CHECK: __mlir_aie_try(XAie_DmaSetNextBd(&(dma_tile02_bd2),  /* nextbd */ 3,  /* enableNextBd */ 1));
    # CHECK: __mlir_aie_try(XAie_DmaEnableBd(&(dma_tile02_bd2)));
    # CHECK: __mlir_aie_try(XAie_DmaWriteBd(&(ctx->DevInst), &(dma_tile02_bd2), XAie_TileLoc(0,2),  /* bd */ 2));
    # CHECK: XAie_DmaDesc dma_tile02_bd3;
    # CHECK: __mlir_aie_try(XAie_DmaDescInit(&(ctx->DevInst), &(dma_tile02_bd3), XAie_TileLoc(0,2)));
    # CHECK: __mlir_aie_try(XAie_DmaSetLock(&(dma_tile02_bd3), XAie_LockInit(1,-1),XAie_LockInit(0,1)));
    # CHECK: __mlir_aie_try(XAie_DmaSetAddrLen(&(dma_tile02_bd3), /* addrA */ 0x1400,  /* len */ 1024 * 4));
    # CHECK: __mlir_aie_try(XAie_DmaSetNextBd(&(dma_tile02_bd3),  /* nextbd */ 2,  /* enableNextBd */ 1));
    # CHECK: __mlir_aie_try(XAie_DmaEnableBd(&(dma_tile02_bd3)));
    # CHECK: __mlir_aie_try(XAie_DmaWriteBd(&(ctx->DevInst), &(dma_tile02_bd3), XAie_TileLoc(0,2),  /* bd */ 3));
    # CHECK: __mlir_aie_try(XAie_DmaChannelPushBdToQueue(&(ctx->DevInst), XAie_TileLoc(0,2), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */0));
    # CHECK: __mlir_aie_try(XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(0,2), /* ChNum */ 0, /* dmaDir */ DMA_S2MM));
    # CHECK: __mlir_aie_try(XAie_DmaChannelPushBdToQueue(&(ctx->DevInst), XAie_TileLoc(0,2), /* ChNum */0, /* dmaDir */ DMA_MM2S, /* BdNum */2));
    # CHECK: __mlir_aie_try(XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(0,2), /* ChNum */ 0, /* dmaDir */ DMA_MM2S));
    # CHECK: return XAIE_OK;
    # CHECK: } // mlir_aie_configure_dmas
    #
    # CHECK: int mlir_aie_initialize_locks(aie_libxaie_ctx_t* ctx) {
    # CHECK: __mlir_aie_try(XAie_LockSetValue(&(ctx->DevInst), XAie_TileLoc(0,0), XAie_LockInit(2, 0)));
    # CHECK: __mlir_aie_try(XAie_LockSetValue(&(ctx->DevInst), XAie_TileLoc(0,0), XAie_LockInit(3, 0)));
    # CHECK: __mlir_aie_try(XAie_LockSetValue(&(ctx->DevInst), XAie_TileLoc(0,2), XAie_LockInit(0, 2)));
    # CHECK: __mlir_aie_try(XAie_LockSetValue(&(ctx->DevInst), XAie_TileLoc(0,2), XAie_LockInit(1, 0)));
    # CHECK: __mlir_aie_try(XAie_LockSetValue(&(ctx->DevInst), XAie_TileLoc(0,0), XAie_LockInit(0, 0)));
    # CHECK: __mlir_aie_try(XAie_LockSetValue(&(ctx->DevInst), XAie_TileLoc(0,0), XAie_LockInit(1, 0)));
    # CHECK: return XAIE_OK;
    # CHECK: } // mlir_aie_initialize_locks
    # CHECK: int mlir_aie_configure_switchboxes(aie_libxaie_ctx_t* ctx) {
    # CHECK:   int x, y;
    # CHECK: // Core Stream Switch column 0 row 0
    # CHECK: x = 0;
    # CHECK: y = 0;
    # CHECK: __mlir_aie_try(XAie_StrmConnCctEnable(&(ctx->DevInst), XAie_TileLoc(x,y), SOUTH, 3, NORTH, 0));
    # CHECK: __mlir_aie_try(XAie_StrmConnCctEnable(&(ctx->DevInst), XAie_TileLoc(x,y), NORTH, 0, SOUTH, 2));
    # CHECK: // Core Stream Switch column 0 row 2
    # CHECK: x = 0;
    # CHECK: y = 2;
    # CHECK: __mlir_aie_try(XAie_StrmConnCctEnable(&(ctx->DevInst), XAie_TileLoc(x,y), SOUTH, 0, DMA, 0));
    # CHECK: __mlir_aie_try(XAie_StrmConnCctEnable(&(ctx->DevInst), XAie_TileLoc(x,y), DMA, 0, SOUTH, 0));
    # CHECK: // Core Stream Switch column 0 row 1
    # CHECK: x = 0;
    # CHECK: y = 1;
    # CHECK: __mlir_aie_try(XAie_StrmConnCctEnable(&(ctx->DevInst), XAie_TileLoc(x,y), SOUTH, 0, NORTH, 0));
    # CHECK: __mlir_aie_try(XAie_StrmConnCctEnable(&(ctx->DevInst), XAie_TileLoc(x,y), NORTH, 0, SOUTH, 0));
    # CHECK: // ShimMux column 0 row 0
    # CHECK: // NOTE ShimMux always connects from the south as directions are defined relative to the tile stream switch
    # CHECK: x = 0;
    # CHECK: y = 0;
    # CHECK: __mlir_aie_try(XAie_EnableShimDmaToAieStrmPort(&(ctx->DevInst), XAie_TileLoc(x,y), 3));
    # CHECK: __mlir_aie_try(XAie_EnableAieToShimDmaStrmPort(&(ctx->DevInst), XAie_TileLoc(x,y), 2));
    # CHECK: return XAIE_OK;
    # CHECK: } // mlir_aie_configure_switchboxes
    #
    # CHECK: const int in_cons_buff_0_offset = 1024;
    # CHECK: int32_t mlir_aie_read_buffer_in_cons_buff_0(aie_libxaie_ctx_t* ctx, int index) {
    # CHECK: u32 value; auto rc = XAie_DataMemRdWord(&(ctx->DevInst), XAie_TileLoc(0,2), in_cons_buff_0_offset + (index*4), &value);
    # CHECK:   return value;
    # CHECK: }
    # CHECK: int mlir_aie_write_buffer_in_cons_buff_0(aie_libxaie_ctx_t* ctx, int index, int32_t value) {
    # CHECK:   int32_t int_value = value;
    # CHECK: AieRC rc =    XAie_DataMemWrWord(&(ctx->DevInst), XAie_TileLoc(0,2), in_cons_buff_0_offset + (index*4), int_value);
    # CHECK: return rc;
    # CHECK: }
    # CHECK: const int in_cons_buff_1_offset = 5120;
    # CHECK: int32_t mlir_aie_read_buffer_in_cons_buff_1(aie_libxaie_ctx_t* ctx, int index) {
    # CHECK: u32 value; auto rc = XAie_DataMemRdWord(&(ctx->DevInst), XAie_TileLoc(0,2), in_cons_buff_1_offset + (index*4), &value);
    # CHECK:   return value;
    # CHECK: }
    # CHECK: int mlir_aie_write_buffer_in_cons_buff_1(aie_libxaie_ctx_t* ctx, int index, int32_t value) {
    # CHECK:   int32_t int_value = value;
    # CHECK: AieRC rc =    XAie_DataMemWrWord(&(ctx->DevInst), XAie_TileLoc(0,2), in_cons_buff_1_offset + (index*4), int_value);
    # CHECK: return rc;
    # CHECK: }
    # CHECK: int mlir_aie_acquire_out_cons_prod_lock(aie_libxaie_ctx_t* ctx, int value, int timeout) {
    # CHECK:   const int id = 2;
    # CHECK:   return XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(0,0), XAie_LockInit(id,value), timeout);
    # CHECK: }
    # CHECK: int mlir_aie_release_out_cons_prod_lock(aie_libxaie_ctx_t* ctx, int value, int timeout) {
    # CHECK:   const int id = 2;
    # CHECK:   return XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(0,0), XAie_LockInit(id,value), timeout);
    # CHECK: }
    # CHECK: int mlir_aie_acquire_out_cons_cons_lock(aie_libxaie_ctx_t* ctx, int value, int timeout) {
    # CHECK:   const int id = 3;
    # CHECK:   return XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(0,0), XAie_LockInit(id,value), timeout);
    # CHECK: }
    # CHECK: int mlir_aie_release_out_cons_cons_lock(aie_libxaie_ctx_t* ctx, int value, int timeout) {
    # CHECK:   const int id = 3;
    # CHECK:   return XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(0,0), XAie_LockInit(id,value), timeout);
    # CHECK: }
    # CHECK: int mlir_aie_acquire_in_cons_prod_lock(aie_libxaie_ctx_t* ctx, int value, int timeout) {
    # CHECK:   const int id = 0;
    # CHECK:   return XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(0,2), XAie_LockInit(id,value), timeout);
    # CHECK: }
    # CHECK: int mlir_aie_release_in_cons_prod_lock(aie_libxaie_ctx_t* ctx, int value, int timeout) {
    # CHECK:   const int id = 0;
    # CHECK:   return XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(0,2), XAie_LockInit(id,value), timeout);
    # CHECK: }
    # CHECK: int mlir_aie_acquire_in_cons_cons_lock(aie_libxaie_ctx_t* ctx, int value, int timeout) {
    # CHECK:   const int id = 1;
    # CHECK:   return XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(0,2), XAie_LockInit(id,value), timeout);
    # CHECK: }
    # CHECK: int mlir_aie_release_in_cons_cons_lock(aie_libxaie_ctx_t* ctx, int value, int timeout) {
    # CHECK:   const int id = 1;
    # CHECK:   return XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(0,2), XAie_LockInit(id,value), timeout);
    # CHECK: }
    # CHECK: int mlir_aie_acquire_in_prod_lock(aie_libxaie_ctx_t* ctx, int value, int timeout) {
    # CHECK:   const int id = 0;
    # CHECK:   return XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(0,0), XAie_LockInit(id,value), timeout);
    # CHECK: }
    # CHECK: int mlir_aie_release_in_prod_lock(aie_libxaie_ctx_t* ctx, int value, int timeout) {
    # CHECK:   const int id = 0;
    # CHECK:   return XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(0,0), XAie_LockInit(id,value), timeout);
    # CHECK: }
    # CHECK: int mlir_aie_acquire_in_cons_lock(aie_libxaie_ctx_t* ctx, int value, int timeout) {
    # CHECK:   const int id = 1;
    # CHECK:   return XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(0,0), XAie_LockInit(id,value), timeout);
    # CHECK: }
    # CHECK: int mlir_aie_release_in_cons_lock(aie_libxaie_ctx_t* ctx, int value, int timeout) {
    # CHECK:   const int id = 1;
    # CHECK:   return XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(0,0), XAie_LockInit(id,value), timeout);
    # CHECK: }
    print(aie_inc)

    aie_control = generate_cdo(input_physical.operation)
    # CHECK: /********************************************* Disclaimer *********************************************/
    # CHECK: /* This file is generated by aie-translate. */
    # CHECK: /* Changes to this file may cause incorrect behavior. */
    #
    # CHECK: /************************** Constants/Macros *****************************/
    # CHECK: #define HW_GEN                   XAIE_DEV_GEN_AIEML
    # CHECK: #define XAIE_NUM_ROWS            6
    # CHECK: #define XAIE_NUM_COLS            5
    # CHECK: #define XAIE_BASE_ADDR           0x40000000
    # CHECK: #define XAIE_COL_SHIFT           25
    # CHECK: #define XAIE_ROW_SHIFT           20
    # CHECK: #define XAIE_SHIM_ROW            0
    # CHECK: #define XAIE_MEM_TILE_ROW_START  1
    # CHECK: #define XAIE_MEM_TILE_NUM_ROWS   1
    # CHECK: #define XAIE_AIE_TILE_ROW_START  2
    # CHECK: #define XAIE_AIE_TILE_NUM_ROWS   4
    # CHECK: #define FOR_WRITE                0
    # CHECK: #define FOR_READ                 1
    # CHECK: #define XAIE_PARTITION_BASE_ADDR 0x0
    #
    # CHECK: /***************************** Includes *********************************/
    # CHECK: //#include <fstream>
    # CHECK: extern "C"
    # CHECK: {
    # CHECK:   #include <xaiengine.h>
    # CHECK: }
    # CHECK: //#include "adf/adf_api/AIEControlConfig.h"
    #
    # CHECK: #define __mlir_aie_try(x) x
    # CHECK: static XAie_DmaDimDesc *__mlir_aie_alloc_dim_desc(size_t ndims) {
    # CHECK:   XAie_DmaDimDesc *ret = NULL;
    # CHECK:   ret = (XAie_DmaDimDesc *)calloc(sizeof(XAie_DmaDimDesc), ndims);
    # CHECK:   if(NULL == ret) {
    # CHECK:     fprintf(stderr, "Allocating DmaDimDesc failed.\n");
    # CHECK:   }
    # CHECK:   return ret;
    # CHECK: }
    # CHECK: XAie_InstDeclare(DevInst, &ConfigPtr);   // Declare global device instance
    #
    # CHECK: bool ppgraph_load_elf(const std::string& work_path, std::vector<std::string>& elfInfoPath)
    # CHECK: {
    # CHECK: std::string work_dir = (work_path.empty() ?  "Work" : work_path);
    # CHECK: {
    # CHECK: if (XAie_LoadElf(&DevInst, XAie_TileLoc(0,2), (work_dir + "/core_0_2.elf").c_str(), XAIE_ENABLE) != XAIE_OK)
    # CHECK: {
    # CHECK:     std::cerr << "ERROR: Failed to load elf for core(%d,%d)" << std::endl;
    # CHECK:     return false;
    # CHECK: }
    # CHECK: }
    # CHECK:     return true;
    # CHECK: } // ppgraph_load_elf
    #
    # CHECK: void ppgraph_core_enable()
    # CHECK: {
    # CHECK: XAie_CoreEnable(&DevInst, XAie_TileLoc(0,2));
    # CHECK:     return;
    # CHECK: } // ppgraph_core_enable
    #
    # CHECK: void enableErrorHandling()
    # CHECK: {
    # CHECK:     XAie_ErrorHandlingInit(&DevInst);
    # CHECK: } // enableErrorHandling
    #
    # CHECK: void ppgraph_init(const std::string& work_path)
    # CHECK: {
    # CHECK: XAie_CoreReset(&DevInst, XAie_TileLoc(0,2));
    # CHECK: XAie_CoreUnreset(&DevInst, XAie_TileLoc(0,2));
    # CHECK: for (int l=0; l<16; l++)
    # CHECK:   XAie_LockSetValue(&DevInst, XAie_TileLoc(0,2), XAie_LockInit(l, 0));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,0), XAie_LockInit(2, 0));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,0), XAie_LockInit(3, 0));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,2), XAie_LockInit(0, 2));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,2), XAie_LockInit(1, 0));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,0), XAie_LockInit(0, 0));
    # CHECK: XAie_LockSetValue(&DevInst, XAie_TileLoc(0,0), XAie_LockInit(1, 0));
    # CHECK: XAie_DmaDesc dma_tile02_bd0;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile02_bd0), XAie_TileLoc(0,2));
    # CHECK: XAie_DmaSetLock(&(dma_tile02_bd0), XAie_LockInit(0,-1),XAie_LockInit(1,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile02_bd0), /* addrA */ 0x400,  /* len */ 1024 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile02_bd0),  /* nextbd */ 1,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile02_bd0));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile02_bd0), XAie_TileLoc(0,2),  /* bd */ 0);
    # CHECK: XAie_DmaDesc dma_tile02_bd1;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile02_bd1), XAie_TileLoc(0,2));
    # CHECK: XAie_DmaSetLock(&(dma_tile02_bd1), XAie_LockInit(0,-1),XAie_LockInit(1,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile02_bd1), /* addrA */ 0x1400,  /* len */ 1024 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile02_bd1),  /* nextbd */ 0,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile02_bd1));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile02_bd1), XAie_TileLoc(0,2),  /* bd */ 1);
    # CHECK: XAie_DmaDesc dma_tile02_bd2;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile02_bd2), XAie_TileLoc(0,2));
    # CHECK: XAie_DmaSetLock(&(dma_tile02_bd2), XAie_LockInit(1,-1),XAie_LockInit(0,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile02_bd2), /* addrA */ 0x400,  /* len */ 1024 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile02_bd2),  /* nextbd */ 3,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile02_bd2));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile02_bd2), XAie_TileLoc(0,2),  /* bd */ 2);
    # CHECK: XAie_DmaDesc dma_tile02_bd3;
    # CHECK: XAie_DmaDescInit(&DevInst, &(dma_tile02_bd3), XAie_TileLoc(0,2));
    # CHECK: XAie_DmaSetLock(&(dma_tile02_bd3), XAie_LockInit(1,-1),XAie_LockInit(0,1));
    # CHECK: XAie_DmaSetAddrLen(&(dma_tile02_bd3), /* addrA */ 0x1400,  /* len */ 1024 * 4);
    # CHECK: XAie_DmaSetNextBd(&(dma_tile02_bd3),  /* nextbd */ 2,  /* enableNextBd */ 1);
    # CHECK: XAie_DmaEnableBd(&(dma_tile02_bd3));
    # CHECK: XAie_DmaWriteBd(&DevInst, &(dma_tile02_bd3), XAie_TileLoc(0,2),  /* bd */ 3);
    # CHECK: XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(0,2), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */0);
    # CHECK: XAie_DmaChannelEnable(&DevInst, XAie_TileLoc(0,2), /* ChNum */ 0, /* dmaDir */ DMA_S2MM);
    # CHECK: XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(0,2), /* ChNum */0, /* dmaDir */ DMA_MM2S, /* BdNum */2);
    # CHECK: XAie_DmaChannelEnable(&DevInst, XAie_TileLoc(0,2), /* ChNum */ 0, /* dmaDir */ DMA_MM2S);
    # CHECK:   int x, y;
    # CHECK: // Core Stream Switch column 0 row 0
    # CHECK: x = 0;
    # CHECK: y = 0;
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), CTRL, 0, SOUTH, 0);
    # CHECK: {
    # CHECK:   //configure DMA_<S2MM/MM2S>_<N>_Ctrl register
    # CHECK:   XAie_DmaChannelDesc DmaChannelDescInst;
    # CHECK:   XAie_DmaChannelDescInit(&DevInst, &DmaChannelDescInst, XAie_TileLoc(x,y));
    # CHECK:   XAie_DmaChannelSetControllerId(&DmaChannelDescInst, 0);
    # CHECK:   XAie_DmaWriteChannel(&DevInst, &DmaChannelDescInst, XAie_TileLoc(x,y), 0, DMA_S2MM);
    # CHECK: }
    #
    # CHECK: {
    # CHECK:   //configure DMA_<S2MM/MM2S>_<N>_Ctrl register
    # CHECK:   XAie_DmaChannelDesc DmaChannelDescInst;
    # CHECK:   XAie_DmaChannelDescInit(&DevInst, &DmaChannelDescInst, XAie_TileLoc(x,y));
    # CHECK:   XAie_DmaChannelSetControllerId(&DmaChannelDescInst, 0);
    # CHECK:   XAie_DmaWriteChannel(&DevInst, &DmaChannelDescInst, XAie_TileLoc(x,y), 1, DMA_S2MM);
    # CHECK: }
    #
    # CHECK: XAie_AieToPlIntfEnable (&DevInst, XAie_TileLoc(x, y), 0, PLIF_WIDTH_32);
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), SOUTH, 3, NORTH, 0);
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), NORTH, 0, SOUTH, 2);
    # CHECK: // Core Stream Switch column 0 row 2
    # CHECK: x = 0;
    # CHECK: y = 2;
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), SOUTH, 0, DMA, 0);
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), DMA, 0, SOUTH, 0);
    # CHECK: // Core Stream Switch column 0 row 1
    # CHECK: x = 0;
    # CHECK: y = 1;
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), SOUTH, 0, NORTH, 0);
    # CHECK: XAie_StrmConnCctEnable(&DevInst, XAie_TileLoc(x,y), NORTH, 0, SOUTH, 0);
    # CHECK: // ShimMux column 0 row 0
    # CHECK: // NOTE ShimMux always connects from the south as directions are defined relative to the tile stream switch
    # CHECK: x = 0;
    # CHECK: y = 0;
    # CHECK: XAie_EnableShimDmaToAieStrmPort(&DevInst, XAie_TileLoc(x,y), 3);
    # CHECK: XAie_EnableAieToShimDmaStrmPort(&DevInst, XAie_TileLoc(x,y), 2);
    # CHECK: } // ppgraph_init
    #
    #
    #
    # CHECK:   class InitializeAIEControl
    # CHECK:   {
    # CHECK:   public:
    # CHECK:     InitializeAIEControl()
    # CHECK:     {
    # CHECK:       XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR, XAIE_COL_SHIFT,
    # CHECK:                        XAIE_ROW_SHIFT, XAIE_NUM_COLS, XAIE_NUM_ROWS,
    # CHECK:                        XAIE_SHIM_ROW, XAIE_MEM_TILE_ROW_START,
    # CHECK:                        XAIE_MEM_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START,
    # CHECK:                        XAIE_AIE_TILE_NUM_ROWS);
    #
    # CHECK:       XAie_SetupPartitionConfig(&DevInst, XAIE_PARTITION_BASE_ADDR, 1, 1);
    #
    # CHECK:       XAie_CfgInitialize(&(DevInst), &ConfigPtr);
    #
    # CHECK: #if defined(__AIESIM__)
    # CHECK: #if defined(__CDO__)
    # CHECK:       XAie_SetIOBackend(&(DevInst), XAIE_IO_BACKEND_CDO); // Set aiengine driver library to run for CDO Mode
    # CHECK:       XAie_UpdateNpiAddr(&(DevInst), 0x0);
    # CHECK: #else
    # CHECK:       //AIE driver currently error out XAie_UpdateNpiAddr for AIESIM
    # CHECK: #endif
    # CHECK: #else
    # CHECK:       XAie_UpdateNpiAddr(&(DevInst), 0x0);
    # CHECK: #endif
    #
    # CHECK: #if defined(__AIESIM__) && !defined(__CDO__)
    # CHECK:       XAie_TurnEccOff(&DevInst);
    # CHECK: #endif
    #
    # CHECK: #if defined(__AIESIM__) && !defined(__CDO__)
    # CHECK:       extern unsigned ess_debug;
    # CHECK: #else
    # CHECK:       unsigned ess_debug = false;
    # CHECK: #endif
    #
    # CHECK: #ifdef __EXCLUDE_PL_CONTROL__
    # CHECK:       bool exclude_pl_control = true;
    # CHECK: #else
    # CHECK:       bool exclude_pl_control = false;
    # CHECK: #endif
    #
    # CHECK: #ifdef __CDO__
    # CHECK:       int trace_config_stream_option = 2;
    # CHECK: #else
    # CHECK:       int trace_config_stream_option = 0;
    # CHECK: #endif
    # CHECK:     }
    # CHECK:   } initAIEControl;
    print(aie_control)

    core_0_2 = generate_bcf(input_with_addresses.operation, 0, 2)
    # CHECK: _entry_point _main_init
    # CHECK: _symbol core_0_2 _after _main_init
    # CHECK: _symbol      _main_init 0
    # CHECK: _reserved DMb      0x00000 0x40000 //Don't put data in code memory
    # CHECK: _reserved DMb 0x40000 0x10000  // No tile with memory exists to the south.
    # CHECK: _reserved DMb 0x50000 0x10000  // No tile with memory exists to the west.
    # CHECK: _reserved DMb 0x60000 0x10000  // Don't allocate variables outside of local memory.
    # CHECK: _symbol in_cons_buff_0 0x70400 0x1000
    # CHECK: _extern in_cons_buff_0
    # CHECK: _reserved DMb 0x70400 0x1000
    # CHECK: _symbol in_cons_buff_1 0x71400 0x1000
    # CHECK: _extern in_cons_buff_1
    # CHECK: _reserved DMb 0x71400 0x1000
    # CHECK: _stack    DM_stack 0x70000  0x400 //stack for core
    # CHECK: _reserved DMb 0x80000 0x80000 // And everything else the core can't see
    # CHECK: _resolve _main core_0_2
    print(core_0_2)


# CHECK-LABEL: add_one_using_dma
# CHECK:  aie.device(ipu) {
# CHECK:    memref.global "public" @objFifo_in0 : memref<16xi32>
# CHECK:    memref.global "public" @objFifo_in0_cons : memref<16xi32>
# CHECK:    memref.global "public" @objFifo_in1 : memref<8xi32>
# CHECK:    memref.global "public" @objFifo_in1_cons : memref<8xi32>
# CHECK:    memref.global "public" @objFifo_out0 : memref<16xi32>
# CHECK:    memref.global "public" @objFifo_out0_cons : memref<16xi32>
# CHECK:    memref.global "public" @objFifo_out1 : memref<8xi32>
# CHECK:    memref.global "public" @objFifo_out1_cons : memref<8xi32>
# CHECK:    %tile_0_0 = aie.tile(0, 0)
# CHECK:    %tile_0_1 = aie.tile(0, 1)
# CHECK:    %tile_0_2 = aie.tile(0, 2)
# CHECK:    %objFifo_in0_cons_buff_0 = aie.buffer(%tile_0_1) {sym_name = "objFifo_in0_cons_buff_0"} : memref<16xi32>
# CHECK:    %objFifo_in0_cons_buff_1 = aie.buffer(%tile_0_1) {sym_name = "objFifo_in0_cons_buff_1"} : memref<16xi32>
# CHECK:    %objFifo_out0_buff_0 = aie.buffer(%tile_0_1) {sym_name = "objFifo_out0_buff_0"} : memref<16xi32>
# CHECK:    %objFifo_out0_buff_1 = aie.buffer(%tile_0_1) {sym_name = "objFifo_out0_buff_1"} : memref<16xi32>
# CHECK:    %objFifo_in1_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_in1_cons_buff_0"} : memref<8xi32>
# CHECK:    %objFifo_in1_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "objFifo_in1_cons_buff_1"} : memref<8xi32>
# CHECK:    %objFifo_out1_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_out1_buff_0"} : memref<8xi32>
# CHECK:    %objFifo_out1_buff_1 = aie.buffer(%tile_0_2) {sym_name = "objFifo_out1_buff_1"} : memref<8xi32>
# CHECK:    %objFifo_in0_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "objFifo_in0_prod_lock"}
# CHECK:    %objFifo_in0_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_lock"}
# CHECK:    %objFifo_out0_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "objFifo_out0_cons_prod_lock"}
# CHECK:    %objFifo_out0_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "objFifo_out0_cons_cons_lock"}
# CHECK:    %objFifo_in0_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 2 : i32, sym_name = "objFifo_in0_cons_prod_lock"}
# CHECK:    %objFifo_in0_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_cons_lock"}
# CHECK:    %objFifo_out0_prod_lock = aie.lock(%tile_0_1, 2) {init = 2 : i32, sym_name = "objFifo_out0_prod_lock"}
# CHECK:    %objFifo_out0_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "objFifo_out0_cons_lock"}
# CHECK:    %objFifo_in1_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "objFifo_in1_cons_prod_lock"}
# CHECK:    %objFifo_in1_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "objFifo_in1_cons_cons_lock"}
# CHECK:    %objFifo_out1_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "objFifo_out1_prod_lock"}
# CHECK:    %objFifo_out1_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "objFifo_out1_cons_lock"}
# CHECK:    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
# CHECK:    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
# CHECK:    aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)
# CHECK:    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
# CHECK:    %core_0_2 = aie.core(%tile_0_2) {
# CHECK:      %c1 = arith.constant 1 : index
# CHECK:      %c1_i32 = arith.constant 1 : i32
# CHECK:      %c0 = arith.constant 0 : index
# CHECK:      %c8 = arith.constant 8 : index
# CHECK:      %c2 = arith.constant 2 : index
# CHECK:      scf.for %arg0 = %c0 to %c8 step %c2 {
# CHECK:        aie.use_lock(%objFifo_in1_cons_cons_lock, AcquireGreaterEqual, 1)
# CHECK:        aie.use_lock(%objFifo_out1_prod_lock, AcquireGreaterEqual, 1)
# CHECK:        scf.for %arg1 = %c0 to %c8 step %c1 {
# CHECK:          %0 = memref.load %objFifo_in1_cons_buff_0[%arg1] : memref<8xi32>
# CHECK:          %1 = arith.addi %0, %c1_i32 : i32
# CHECK:          memref.store %1, %objFifo_out1_buff_0[%arg1] : memref<8xi32>
# CHECK:        }
# CHECK:        aie.use_lock(%objFifo_in1_cons_prod_lock, Release, 1)
# CHECK:        aie.use_lock(%objFifo_out1_cons_lock, Release, 1)
# CHECK:        aie.use_lock(%objFifo_in1_cons_cons_lock, AcquireGreaterEqual, 1)
# CHECK:        aie.use_lock(%objFifo_out1_prod_lock, AcquireGreaterEqual, 1)
# CHECK:        scf.for %arg1 = %c0 to %c8 step %c1 {
# CHECK:          %0 = memref.load %objFifo_in1_cons_buff_1[%arg1] : memref<8xi32>
# CHECK:          %1 = arith.addi %0, %c1_i32 : i32
# CHECK:          memref.store %1, %objFifo_out1_buff_1[%arg1] : memref<8xi32>
# CHECK:        }
# CHECK:        aie.use_lock(%objFifo_in1_cons_prod_lock, Release, 1)
# CHECK:        aie.use_lock(%objFifo_out1_cons_lock, Release, 1)
# CHECK:      }
# CHECK:      aie.end
# CHECK:    }
# CHECK:    aie.shim_dma_allocation @objFifo_in0(MM2S, 0, 0)
# CHECK:    func.func @bobsyouruncle(%arg0: memref<64xi32>, %arg1: memref<32xi32>, %arg2: memref<64xi32>) {
# CHECK:      aiex.ipu.dma_memcpy_nd(0, 0, %arg0 : memref<64xi32>) {id = 0 : i32, lengths = array<i32: 1, 1, 1, 64>, metadata = @objFifo_in0, offsets = array<i32: 0, 0, 0, 0>, strides = array<i32: 0, 0, 0>}
# CHECK:      aiex.ipu.dma_memcpy_nd(0, 0, %arg2 : memref<64xi32>) {id = 1 : i32, lengths = array<i32: 1, 1, 1, 64>, metadata = @objFifo_out0, offsets = array<i32: 0, 0, 0, 0>, strides = array<i32: 0, 0, 0>}
# CHECK:      aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
# CHECK:      return
# CHECK:    }
# CHECK:    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
# CHECK:      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
# CHECK:    ^bb1:  // 2 preds: ^bb0, ^bb2
# CHECK:      aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
# CHECK:      aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<16xi32>, 0, 16)
# CHECK:      aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
# CHECK:      aie.next_bd ^bb2
# CHECK:    ^bb2:  // pred: ^bb1
# CHECK:      aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
# CHECK:      aie.dma_bd(%objFifo_in0_cons_buff_1 : memref<16xi32>, 0, 16)
# CHECK:      aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
# CHECK:      aie.next_bd ^bb1
# CHECK:    ^bb3:  // pred: ^bb0
# CHECK:      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
# CHECK:    ^bb4:  // 2 preds: ^bb3, ^bb5
# CHECK:      aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
# CHECK:      aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<16xi32>, 0, 16)
# CHECK:      aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
# CHECK:      aie.next_bd ^bb5
# CHECK:    ^bb5:  // pred: ^bb4
# CHECK:      aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
# CHECK:      aie.dma_bd(%objFifo_in0_cons_buff_1 : memref<16xi32>, 0, 16)
# CHECK:      aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
# CHECK:      aie.next_bd ^bb4
# CHECK:    ^bb6:  // pred: ^bb3
# CHECK:      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
# CHECK:    ^bb7:  // 2 preds: ^bb6, ^bb8
# CHECK:      aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
# CHECK:      aie.dma_bd(%objFifo_out0_buff_0 : memref<16xi32>, 0, 16)
# CHECK:      aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
# CHECK:      aie.next_bd ^bb8
# CHECK:    ^bb8:  // pred: ^bb7
# CHECK:      aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
# CHECK:      aie.dma_bd(%objFifo_out0_buff_1 : memref<16xi32>, 0, 16)
# CHECK:      aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
# CHECK:      aie.next_bd ^bb7
# CHECK:    ^bb9:  // pred: ^bb6
# CHECK:      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb12)
# CHECK:    ^bb10:  // 2 preds: ^bb9, ^bb11
# CHECK:      aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
# CHECK:      aie.dma_bd(%objFifo_out0_buff_0 : memref<16xi32>, 0, 16)
# CHECK:      aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
# CHECK:      aie.next_bd ^bb11
# CHECK:    ^bb11:  // pred: ^bb10
# CHECK:      aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
# CHECK:      aie.dma_bd(%objFifo_out0_buff_1 : memref<16xi32>, 0, 16)
# CHECK:      aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
# CHECK:      aie.next_bd ^bb10
# CHECK:    ^bb12:  // pred: ^bb9
# CHECK:      aie.end
# CHECK:    }
# CHECK:    aie.shim_dma_allocation @objFifo_out0(S2MM, 0, 0)
# CHECK:    %mem_0_2 = aie.mem(%tile_0_2) {
# CHECK:      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
# CHECK:    ^bb1:  // 2 preds: ^bb0, ^bb2
# CHECK:      aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
# CHECK:      aie.dma_bd(%objFifo_in1_cons_buff_0 : memref<8xi32>, 0, 8)
# CHECK:      aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
# CHECK:      aie.next_bd ^bb2
# CHECK:    ^bb2:  // pred: ^bb1
# CHECK:      aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
# CHECK:      aie.dma_bd(%objFifo_in1_cons_buff_1 : memref<8xi32>, 0, 8)
# CHECK:      aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
# CHECK:      aie.next_bd ^bb1
# CHECK:    ^bb3:  // pred: ^bb0
# CHECK:      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
# CHECK:    ^bb4:  // 2 preds: ^bb3, ^bb5
# CHECK:      aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
# CHECK:      aie.dma_bd(%objFifo_out1_buff_0 : memref<8xi32>, 0, 8)
# CHECK:      aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
# CHECK:      aie.next_bd ^bb5
# CHECK:    ^bb5:  // pred: ^bb4
# CHECK:      aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
# CHECK:      aie.dma_bd(%objFifo_out1_buff_1 : memref<8xi32>, 0, 8)
# CHECK:      aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
# CHECK:      aie.next_bd ^bb4
# CHECK:    ^bb6:  // pred: ^bb3
# CHECK:      aie.end
# CHECK:    }
# CHECK:  }
@construct_and_print_module
def add_one_using_dma(module):
    @device(AIEDevice.ipu)
    def ipu():
        memref.global_("objFifo_in0", T.memref(16, T.i32()), sym_visibility="public")
        memref.global_(
            "objFifo_in0_cons", T.memref(16, T.i32()), sym_visibility="public"
        )
        memref.global_("objFifo_in1", T.memref(8, T.i32()), sym_visibility="public")
        memref.global_(
            "objFifo_in1_cons", T.memref(8, T.i32()), sym_visibility="public"
        )
        memref.global_("objFifo_out0", T.memref(16, T.i32()), sym_visibility="public")
        memref.global_(
            "objFifo_out0_cons", T.memref(16, T.i32()), sym_visibility="public"
        )
        memref.global_("objFifo_out1", T.memref(8, T.i32()), sym_visibility="public")
        memref.global_(
            "objFifo_out1_cons", T.memref(8, T.i32()), sym_visibility="public"
        )

        tile_0_0 = tile(0, 0)
        tile_0_1 = tile(0, 1)
        tile_0_2 = tile(0, 2)

        objFifo_in0_cons_buff_0 = aie.buffer(
            T.memref(16, T.i32()), tile_0_1, sym_name="objFifo_in0_cons_buff_0"
        )
        objFifo_in0_cons_buff_1 = aie.buffer(
            T.memref(16, T.i32()), tile_0_1, sym_name="objFifo_in0_cons_buff_1"
        )
        objFifo_out0_buff_0 = aie.buffer(
            T.memref(16, T.i32()), tile_0_1, sym_name="objFifo_out0_buff_0"
        )
        objFifo_out0_buff_1 = aie.buffer(
            T.memref(16, T.i32()), tile_0_1, sym_name="objFifo_out0_buff_1"
        )

        objFifo_in1_cons_buff_0 = aie.buffer(
            T.memref(8, T.i32()), tile_0_2, sym_name="objFifo_in1_cons_buff_0"
        )
        objFifo_in1_cons_buff_1 = aie.buffer(
            T.memref(8, T.i32()), tile_0_2, sym_name="objFifo_in1_cons_buff_1"
        )
        objFifo_out1_buff_0 = aie.buffer(
            T.memref(8, T.i32()), tile_0_2, sym_name="objFifo_out1_buff_0"
        )
        objFifo_out1_buff_1 = aie.buffer(
            T.memref(8, T.i32()), tile_0_2, sym_name="objFifo_out1_buff_1"
        )

        objFifo_in0_prod_lock = aie.lock(
            tile_0_0, lock_id=0, init=0, sym_name="objFifo_in0_prod_lock"
        )
        objFifo_in0_cons_lock = aie.lock(
            tile_0_0, lock_id=1, init=0, sym_name="objFifo_in0_cons_lock"
        )
        objFifo_out0_cons_prod_lock = aie.lock(
            tile_0_0, lock_id=2, init=0, sym_name="objFifo_out0_cons_prod_lock"
        )
        objFifo_out0_cons_cons_lock = aie.lock(
            tile_0_0, lock_id=3, init=0, sym_name="objFifo_out0_cons_cons_lock"
        )

        objFifo_in0_cons_prod_lock = aie.lock(
            tile_0_1, lock_id=0, init=2, sym_name="objFifo_in0_cons_prod_lock"
        )
        objFifo_in0_cons_cons_lock = aie.lock(
            tile_0_1, lock_id=1, init=0, sym_name="objFifo_in0_cons_cons_lock"
        )
        objFifo_out0_prod_lock = aie.lock(
            tile_0_1, lock_id=2, init=2, sym_name="objFifo_out0_prod_lock"
        )
        objFifo_out0_cons_lock = aie.lock(
            tile_0_1, lock_id=3, init=0, sym_name="objFifo_out0_cons_lock"
        )

        objFifo_in1_cons_prod_lock = aie.lock(
            tile_0_2, lock_id=0, init=2, sym_name="objFifo_in1_cons_prod_lock"
        )
        objFifo_in1_cons_cons_lock = aie.lock(
            tile_0_2, lock_id=1, init=0, sym_name="objFifo_in1_cons_cons_lock"
        )
        objFifo_out1_prod_lock = aie.lock(
            tile_0_2, lock_id=2, init=2, sym_name="objFifo_out1_prod_lock"
        )
        objFifo_out1_cons_lock = aie.lock(
            tile_0_2, lock_id=3, init=0, sym_name="objFifo_out1_cons_lock"
        )

        aie.flow(tile_0_0, DMA, 0, tile_0_1, DMA, 0)
        aie.flow(tile_0_1, DMA, 0, tile_0_2, DMA, 0)
        aie.flow(tile_0_1, DMA, 1, tile_0_0, DMA, 0)
        aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 1)

        @aie.core(tile_0_2)
        def core():
            c1_i32 = arith.constant(1)
            for i in range_(0, 8, 2):
                # TODO(max): fix the ordering in the asm to match the ordering in the `ins`
                aie.use_lock(objFifo_in1_cons_cons_lock, 1, AcquireGreaterEqual)
                aie.use_lock(objFifo_out1_prod_lock, 1, AcquireGreaterEqual)

                for arg1 in range_(0, 8, 1):
                    v0 = memref.load(objFifo_in1_cons_buff_0, [arg1])
                    v1 = arith.addi(v0, c1_i32)
                    memref.store(v1, objFifo_out1_buff_0, [arg1])
                    yield_([])

                aie.use_lock(objFifo_in1_cons_prod_lock, 1, Release)
                aie.use_lock(objFifo_out1_cons_lock, 1, Release)

                aie.use_lock(objFifo_in1_cons_cons_lock, 1, AcquireGreaterEqual)
                aie.use_lock(objFifo_out1_prod_lock, 1, AcquireGreaterEqual)

                for arg1 in range_(0, 8, 1):
                    v0 = memref.load(objFifo_in1_cons_buff_1, [arg1])
                    v1 = arith.addi(v0, c1_i32)
                    memref.store(v1, objFifo_out1_buff_1, [arg1])
                    yield_([])

                aie.use_lock(objFifo_in1_cons_prod_lock, 1, Release)
                aie.use_lock(objFifo_out1_cons_lock, 1, Release)

                yield_([])

        aie.shim_dma_allocation("objFifo_in0", MM2S, 0, 0)

        @func.func(emit=True)
        def bobsyouruncle(
            arg0: T.memref(64, T.i32()),
            arg1: T.memref(32, T.i32()),
            arg2: T.memref(64, T.i32()),
        ):
            ipu_dma_memcpy_nd_(
                0,
                0,
                arg0,
                [0, 0, 0, 0],
                [1, 1, 1, 64],
                [0, 0, 0],
                metadata="objFifo_in0",
                id=0,
            )

            ipu_dma_memcpy_nd_(
                0,
                0,
                arg2,
                [0, 0, 0, 0],
                [1, 1, 1, 64],
                [0, 0, 0],
                metadata="objFifo_out0",
                id=1,
            )
            ipu_sync(channel=0, column=0, column_num=1, direction=0, row=0, row_num=1)

        @memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            bb1, bb3 = aie.dma_start(S2MM, 0)
            with bb(bb1):  # 2 preds: bb0, bb2
                aie.use_lock(objFifo_in0_cons_prod_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_in0_cons_buff_0, 0, 16)
                aie.use_lock(objFifo_in0_cons_cons_lock, 1, Release)
                bb2 = aie.next_bd()
            with bb(bb2):  # pred: bb1
                aie.use_lock(objFifo_in0_cons_prod_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_in0_cons_buff_1, 0, 16)
                aie.use_lock(objFifo_in0_cons_cons_lock, 1, Release)
                aie.next_bd(bb1)
            with bb(bb3):  # pred: bb0
                bb4, bb6 = aie.dma_start(MM2S, 0)
            with bb(bb4):  # 2 preds: bb3, bb5
                aie.use_lock(objFifo_in0_cons_cons_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_in0_cons_buff_0, 0, 16)
                aie.use_lock(objFifo_in0_cons_prod_lock, 1, Release)
                bb5 = aie.next_bd()
            with bb(bb5):  # pred: bb4
                aie.use_lock(objFifo_in0_cons_cons_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_in0_cons_buff_1, 0, 16)
                aie.use_lock(objFifo_in0_cons_prod_lock, 1, Release)
                aie.next_bd(bb4)
            with bb(bb6):  # pred: bb3
                bb7, bb9 = aie.dma_start(MM2S, 1)
            with bb(bb7):  # 2 preds: bb6, bb8
                aie.use_lock(objFifo_out0_cons_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_out0_buff_0, 0, 16)
                aie.use_lock(objFifo_out0_prod_lock, 1, Release)
                bb8 = aie.next_bd()
            with bb(bb8):  # pred: bb7
                aie.use_lock(objFifo_out0_cons_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_out0_buff_1, 0, 16)
                aie.use_lock(objFifo_out0_prod_lock, 1, Release)
                aie.next_bd(bb7)
            with bb(bb9):  # pred: bb6
                bb10, bb12 = aie.dma_start(S2MM, 1)
            with bb(bb10):  # 2 preds: bb9, bb11
                aie.use_lock(objFifo_out0_prod_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_out0_buff_0, 0, 16)
                aie.use_lock(objFifo_out0_cons_lock, 1, Release)
                bb11 = aie.next_bd()
            with bb(bb11):  # pred: bb10
                aie.use_lock(objFifo_out0_prod_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_out0_buff_1, 0, 16)
                aie.use_lock(objFifo_out0_cons_lock, 1, Release)
                aie.next_bd(bb10)
            with bb(bb12):  # pred: bb9
                aie.end()

        aie.shim_dma_allocation("objFifo_out0", S2MM, 0, 0)

        @mem(tile_0_2)
        def mem_0_2():
            bb1, bb3 = aie.dma_start(S2MM, 0)
            with bb(bb1):  # 2 preds: bb0, bb2
                aie.use_lock(objFifo_in1_cons_prod_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_in1_cons_buff_0, 0, 8)
                aie.use_lock(objFifo_in1_cons_cons_lock, 1, Release)
                bb2 = aie.next_bd()
            with bb(bb2):  # pred: bb1
                aie.use_lock(objFifo_in1_cons_prod_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_in1_cons_buff_1, 0, 8)
                aie.use_lock(objFifo_in1_cons_cons_lock, 1, Release)
                aie.next_bd(bb1)
            with bb(bb3):  # pred: bb0
                bb4, bb6 = aie.dma_start(MM2S, 0)
            with bb(bb4):  # 2 preds: bb3, bb5
                aie.use_lock(objFifo_out1_cons_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_out1_buff_0, 0, 8)
                aie.use_lock(objFifo_out1_prod_lock, 1, Release)
                bb5 = aie.next_bd()
            with bb(bb5):  # pred: bb4
                aie.use_lock(objFifo_out1_cons_lock, 1, AcquireGreaterEqual)
                aie.dma_bd(objFifo_out1_buff_1, 0, 8)
                aie.use_lock(objFifo_out1_prod_lock, 1, Release)
                aie.next_bd(bb4)
            with bb(bb6):  # pred: bb3
                aie.end()

    mod = run_pipeline(module, Pipeline().cse().canonicalize())

    print(mod)
