# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

# RUN: %python %s | FileCheck %s
import numpy as np
from aie.extras.dialects.ext import memref
import aie.extras.types as T
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    ObjectFifoPort,
    WireBundle,
    core,
    device,
    external_func,
    object_fifo,
    object_fifo_link,
    tile,
)
from aie.dialects.aiex import dma_wait, npu_dma_memcpy_nd, runtime_sequence
from aie.helpers.dialects.ext.scf import _for as range_
from util import construct_and_print_module

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


# CHECK-LABEL: my_vector_scalar
@construct_and_print_module
def my_vector_scalar(module):
    N = 4096
    n = 1024
    N_div_n = N // n

    buffer_depth = 2

    @device(AIEDevice.npu1_4col)
    def device_body():
        n_ty = np.ndarray[(n,), np.dtype[np.int32]]
        N_ty = np.ndarray[(N,), np.dtype[np.int32]]
        scale_int32 = external_func("scale_int32", inputs=[n_ty, n_ty])

        S = tile(0, 0)
        M = tile(0, 2)

        of_in = object_fifo("in", S, M, buffer_depth, n_ty)
        of_out = object_fifo("out", M, S, buffer_depth, n_ty)

        @core(M, "scale.o")
        def core_body():
            # Effective while(1)
            for _ in range_(0xFFFFFFFF):
                # Number of sub-vector "tile" iterations
                for _ in range_(N_div_n):
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    scale_int32(elem_in, elem_out)
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(N_ty, N_ty, N_ty)
        def sequence(A, B, C):
            npu_dma_memcpy_nd(metadata=of_in, bd_id=1, mem=A, sizes=[1, 1, 1, N])
            npu_dma_memcpy_nd(metadata=of_out, bd_id=0, mem=C, sizes=[1, 1, 1, N])
            dma_wait(of_out)

    assert module.operation.verify()


# CHECK-LABEL: my_matmul
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

    @device(AIEDevice.npu1_4col)
    def device_body():
        func_type = "" if vectorized else "scalar_"
        zero = external_func(
            f"zero_{func_type}i16", inputs=[np.ndarray[(m, n), np.dtype[np.int16]]]
        )
        matmul = external_func(
            f"matmul_{func_type}i16_i16",
            inputs=[
                np.ndarray[(m, k), np.dtype[np.int16]],
                np.ndarray[(k, n), np.dtype[np.int16]],
                np.ndarray[(m, n), np.dtype[np.int16]],
            ],
        )

        S = tile(0, 0)
        M = tile(0, 2)

        of_inA = object_fifo("inA", S, M, 2, np.ndarray[(m, k), np.dtype[np.int16]])
        of_inB = object_fifo("inB", S, M, 2, np.ndarray[(k, n), np.dtype[np.int16]])
        of_outC = object_fifo("outC", M, S, 2, np.ndarray[(m, n), np.dtype[np.int16]])

        @core(M, "mm.o")
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(tiles):
                    elem_out = of_outC.acquire(ObjectFifoPort.Produce, 1)
                    zero(elem_out)

                    for _ in range_(K_div_k):
                        elem_in_a = of_inA.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_b = of_inB.acquire(ObjectFifoPort.Consume, 1)
                        matmul(elem_in_a, elem_in_b, elem_out)
                        of_inA.release(ObjectFifoPort.Consume, 1)
                        of_inB.release(ObjectFifoPort.Consume, 1)
                    of_outC.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(A_sz_in_i32s,), np.dtype[np.int32]],
            np.ndarray[(B_sz_in_i32s,), np.dtype[np.int32]],
            np.ndarray[(C_sz_in_i32s,), np.dtype[np.int32]],
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
                for tile_row in range(num_tile_rows):
                    A_row_offset_in_i32s = (
                        ((tile_row_block * rows_per_block) + tile_row)
                        * m
                        * K
                        * word_size_in
                        // 4
                    )
                    npu_dma_memcpy_nd(
                        metadata=of_inA,
                        bd_id=2 * tile_row + 1,
                        mem=A,
                        offsets=[0, 0, 0, A_row_offset_in_i32s],
                        sizes=[N_div_n, K_div_k, m, k_in_i32s],
                        strides=[0, k_in_i32s, K_in_i32s, 1],
                    )
                    npu_dma_memcpy_nd(
                        metadata=of_inB,
                        bd_id=2 * tile_row + 2,
                        mem=B,
                        sizes=[N_div_n, K_div_k, k, n_in_i32s],
                        strides=[n_in_i32s, k_x_N_in_i32s, N_in_i32s, 1],
                    )
                npu_dma_memcpy_nd(
                    metadata=of_outC,
                    bd_id=0,
                    mem=C,
                    offsets=[0, 0, 0, C_row_offset_in_i32s],
                    sizes=[num_tile_rows, N_div_n, m, n_in_i32s_out],
                    strides=[m_x_N_in_i32s_out, n_in_i32s_out, N_in_i32s_out, 1],
                )
                dma_wait(of_outC)

    assert module.operation.verify()


# CHECK-LABEL: edge_detect
@construct_and_print_module
def edge_detect(module):
    @device(AIEDevice.npu1_4col)
    def device_body():
        vec64_ty = np.ndarray[(64,), np.dtype[np.uint8]]
        vec256_ty = np.ndarray[(256,), np.dtype[np.uint8]]
        rgba2gray_line = external_func(
            "rgba2gray_line", inputs=[vec256_ty, vec64_ty, np.int32]
        )
        filter2d_line = external_func(
            "filter2d_line",
            inputs=[
                vec64_ty,
                vec64_ty,
                vec64_ty,
                vec64_ty,
                np.int32,
                np.ndarray[(3, 3), np.dtype[np.int16]],
            ],
        )
        threshold_line = external_func(
            "threshold_line",
            inputs=[
                vec64_ty,
                vec64_ty,
                np.int32,
                np.int16,
                np.int16,
                np.int8,
            ],
        )
        gray2rgba_line = external_func(
            "gray2rgba_line", inputs=[vec64_ty, vec256_ty, np.int32]
        )
        add_weighted_line = external_func(
            "add_weighted_line",
            inputs=[
                vec256_ty,
                vec256_ty,
                vec256_ty,
                np.int32,
                np.int16,
                np.int16,
                np.int8,
            ],
        )

        S = tile(0, 0)
        M = tile(0, 1)
        T2 = tile(0, 2)
        T3 = tile(0, 3)
        T4 = tile(0, 4)
        T5 = tile(0, 5)

        inOF_L3L2 = object_fifo("inOF_L3L2", S, M, 2, vec256_ty)
        inOF_L2L1 = object_fifo("inOF_L2L1", M, [T2, T5], [2, 2, 7], vec256_ty)
        object_fifo_link(inOF_L3L2, inOF_L2L1)

        outOF_L2L3 = object_fifo("outOF_L2L3", M, S, 2, vec256_ty)
        outOF_L1L2 = object_fifo("outOF_L1L2", T5, M, 2, vec256_ty)
        object_fifo_link(outOF_L1L2, outOF_L2L3)

        OF_2to3 = object_fifo("OF_2to3", T2, T3, 4, vec64_ty)
        OF_3to4 = object_fifo("OF_3to4", T3, T4, 2, vec64_ty)
        OF_4to5 = object_fifo("OF_4to5", T4, T5, 2, vec64_ty)
        OF_5to5 = object_fifo("OF_5to5", T5, T5, 1, vec256_ty)

        @core(T2, "rgba2gray.cc.o")
        def core_body():
            for _ in range_(36):
                elem_in = inOF_L2L1.acquire(ObjectFifoPort.Consume, 1)
                elem_out = OF_2to3.acquire(ObjectFifoPort.Produce, 1)

                rgba2gray_line(elem_in, elem_out, 64)

                inOF_L2L1.release(ObjectFifoPort.Consume, 1)
                OF_2to3.release(ObjectFifoPort.Produce, 1)

        @core(T3, "filter2d.cc.o")
        def core_body():
            kernel = memref.alloc((3, 3), T.i16())
            v0 = 0
            v1 = 4096
            v_minus4 = -16384
            kernel[0, 0] = v0
            kernel[0, 1] = v1
            kernel[0, 2] = v0
            kernel[1, 0] = v1
            kernel[1, 1] = v_minus4
            kernel[1, 2] = v1
            kernel[2, 0] = v0
            kernel[2, 1] = v1
            kernel[2, 2] = v0

            # Preamble : Top Border
            elems_in_pre = OF_2to3.acquire(ObjectFifoPort.Consume, 2)
            elem_pre_out = OF_3to4.acquire(ObjectFifoPort.Produce, 1)
            filter2d_line(
                elems_in_pre[0],
                elems_in_pre[0],
                elems_in_pre[1],
                elem_pre_out,
                64,
                kernel,
            )
            OF_3to4.release(ObjectFifoPort.Produce, 1)

            # Steady State : Middle
            for _ in range_(1, 35):
                elems_in = OF_2to3.acquire(ObjectFifoPort.Consume, 3)
                elem_out = OF_3to4.acquire(ObjectFifoPort.Produce, 1)
                filter2d_line(
                    elems_in[0], elems_in[1], elems_in[2], elem_out, 64, kernel
                )
                OF_2to3.release(ObjectFifoPort.Consume, 1)
                OF_3to4.release(ObjectFifoPort.Produce, 1)

            # Postamble : Bottom Border
            elems_in_post = OF_2to3.acquire(ObjectFifoPort.Consume, 2)
            elem_post_out = OF_3to4.acquire(ObjectFifoPort.Produce, 1)
            filter2d_line(
                elems_in_post[0],
                elems_in_post[1],
                elems_in_post[1],
                elem_post_out,
                64,
                kernel,
            )
            OF_2to3.release(ObjectFifoPort.Consume, 2)
            OF_3to4.release(ObjectFifoPort.Produce, 1)

        @core(T4, "threshold.cc.o")
        def core_body():
            v_thr = 10
            v_max = 255
            v_typ = 0

            for _ in range_(36):
                elem_in = OF_3to4.acquire(ObjectFifoPort.Consume, 1)
                elem_out = OF_4to5.acquire(ObjectFifoPort.Produce, 1)
                threshold_line(elem_in, elem_out, 64, v_thr, v_max, v_typ)
                OF_3to4.release(ObjectFifoPort.Consume, 1)
                OF_4to5.release(ObjectFifoPort.Produce, 1)

        @core(T5, "combined_gray2rgba_addWeighted.a")
        def core_body():
            for _ in range_(36):
                elem_in = OF_4to5.acquire(ObjectFifoPort.Consume, 1)
                elem_out = OF_5to5.acquire(ObjectFifoPort.Produce, 1)
                gray2rgba_line(elem_in, elem_out, 64)
                OF_4to5.release(ObjectFifoPort.Consume, 1)
                OF_5to5.release(ObjectFifoPort.Produce, 1)

                elem_in1 = OF_5to5.acquire(ObjectFifoPort.Consume, 1)
                elem_in2 = inOF_L2L1.acquire(ObjectFifoPort.Consume, 1)
                elem_out2 = outOF_L1L2.acquire(ObjectFifoPort.Produce, 1)

                alpha = 16384
                beta = 16384
                gamma = 0

                add_weighted_line(
                    elem_in1, elem_in2, elem_out2, 256, alpha, beta, gamma
                )

                OF_5to5.release(ObjectFifoPort.Consume, 1)
                inOF_L2L1.release(ObjectFifoPort.Consume, 1)
                outOF_L1L2.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(2304,), np.dtype[np.int32]],
            np.ndarray[(2304,), np.dtype[np.int32]],
            np.ndarray[(2304,), np.dtype[np.int32]],
        )
        def sequence(I, B, O):
            npu_dma_memcpy_nd(
                metadata=inOF_L3L2,
                bd_id=1,
                mem=I,
                sizes=[1, 1, 36, 64],
                strides=[0, 0, 64, 1],
            )
            npu_dma_memcpy_nd(
                metadata=outOF_L2L3,
                bd_id=0,
                mem=O,
                sizes=[1, 1, 36, 64],
                strides=[0, 0, 64, 1],
            )
            dma_wait(outOF_L2L3)

    assert module.operation.verify()


# CHECK-LABEL: my_add_one_objFifo
@construct_and_print_module
def my_add_one_objFifo(module):
    @device(AIEDevice.npu1_4col)
    def device_body():
        shim_tile = tile(0, 0)
        mem_tile = tile(0, 1)
        compute_tile2 = tile(0, 2)

        tile16_ty = np.ndarray[(16,), np.dtype[np.int32]]
        tile8_ty = np.ndarray[(8,), np.dtype[np.int32]]

        of_in0 = object_fifo("in0", shim_tile, mem_tile, 2, tile16_ty)
        of_in1 = object_fifo("in1", mem_tile, compute_tile2, 2, tile16_ty)
        object_fifo_link(of_in0, of_in1)

        of_out0 = object_fifo("out0", mem_tile, shim_tile, 2, tile8_ty)
        of_out1 = object_fifo("out1", compute_tile2, mem_tile, 2, tile16_ty)
        object_fifo_link(of_out1, of_out0)

        @core(compute_tile2)
        def core_body():
            # Effective while(1)
            for _ in range_(8):
                elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                for i in range_(8):
                    elem_out[i] = elem_in[i] + 1
                of_in1.release(ObjectFifoPort.Consume, 1)
                of_out1.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(64,), np.dtype[np.int32]],
            np.ndarray[(32,), np.dtype[np.int32]],
            np.ndarray[(64,), np.dtype[np.int32]],
        )
        def sequence(inTensor, notUsed, outTensor):
            npu_dma_memcpy_nd(
                metadata=of_in0, bd_id=1, mem=inTensor, sizes=[1, 1, 1, 64]
            )
            npu_dma_memcpy_nd(
                metadata=of_out0, bd_id=0, mem=outTensor, sizes=[1, 1, 1, 64]
            )
            dma_wait(of_out0)

    assert module.operation.verify()
