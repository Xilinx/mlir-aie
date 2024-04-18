# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

# RUN: %python %s | FileCheck %s

from aie.extras.dialects.ext import memref, arith

import aie.extras.types as T
from aie.dialects.aie import (
    AIEDevice,
    call,
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
from aie.dialects.aiex import npu_sync, npu_dma_memcpy_nd
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
@construct_and_print_module
def my_vector_scalar(module):
    N = 4096
    n = 1024
    N_div_n = N // n

    buffer_depth = 2

    @device(AIEDevice.npu)
    def device_body():
        scale_int32 = external_func(
            "scale_int32", inputs=[T.memref(n, T.i32()), T.memref(n, T.i32())]
        )

        S = tile(0, 0)
        M = tile(0, 2)

        of_in = object_fifo("in", S, M, buffer_depth, T.memref(n, T.i32()))
        of_out = object_fifo("out", M, S, buffer_depth, T.memref(n, T.i32()))

        @core(M, "scale.o")
        def core_body():
            # Effective while(1)
            for _ in range_(0xFFFFFFFF):
                # Number of sub-vector "tile" iterations
                for _ in range_(N_div_n):
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    call(scale_int32, [elem_in, elem_out])
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                yield_([])

        @FuncOp.from_py_func(
            T.memref(N, T.i32()), T.memref(N, T.i32()), T.memref(N, T.i32())
        )
        def sequence(A, B, C):
            npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
            npu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, N])
            npu_sync(column=0, row=0, direction=0, channel=0)

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

    @device(AIEDevice.npu)
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

        of_inA = object_fifo("inA", S, M, 2, T.memref(m, k, T.i16()))
        of_inB = object_fifo("inB", S, M, 2, T.memref(k, n, T.i16()))
        of_outC = object_fifo("outC", M, S, 2, T.memref(m, n, T.i16()))

        @core(M, "mm.o")
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(tiles):
                    elem_out = of_outC.acquire(ObjectFifoPort.Produce, 1)
                    if vectorized:
                        call(zero, [elem_out])
                    else:
                        call(zero_scalar, [elem_out])

                    for _ in range_(K_div_k):
                        elem_in_a = of_inA.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_b = of_inB.acquire(ObjectFifoPort.Consume, 1)
                        if vectorized:
                            call(matmul, [elem_in_a, elem_in_b, elem_out])
                        else:
                            call(matmul_scalar, [elem_in_a, elem_in_b, elem_out])
                        of_inA.release(ObjectFifoPort.Consume, 1)
                        of_inB.release(ObjectFifoPort.Consume, 1)
                        yield_([])

                    of_outC.release(ObjectFifoPort.Produce, 1)
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
                npu_dma_memcpy_nd(
                    metadata="outC",
                    bd_id=0,
                    mem=C,
                    offsets=[0, 0, 0, C_row_offset_in_i32s],
                    sizes=[num_tile_rows, N_div_n, m, n_in_i32s_out],
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
                    npu_dma_memcpy_nd(
                        metadata="inA",
                        bd_id=2 * tile_row + 1,
                        mem=A,
                        offsets=[0, 0, 0, A_row_offset_in_i32s],
                        sizes=[N_div_n, K_div_k, m, k_in_i32s],
                        strides=[0, k_in_i32s, K_in_i32s],
                    )
                    npu_dma_memcpy_nd(
                        metadata="inB",
                        bd_id=2 * tile_row + 2,
                        mem=B,
                        sizes=[N_div_n, K_div_k, k, n_in_i32s],
                        strides=[n_in_i32s, k_x_N_in_i32s, N_in_i32s],
                    )

                npu_sync(column=0, row=0, direction=0, channel=0)

    assert module.operation.verify()


# CHECK-LABEL: edge_detect
@construct_and_print_module
def edge_detect(module):
    @device(AIEDevice.npu)
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

        inOF_L3L2 = object_fifo("inOF_L3L2", S, M, 2, T.memref(256, T.ui8()))
        inOF_L2L1 = object_fifo(
            "inOF_L2L1", M, [T2, T5], [2, 2, 7], T.memref(256, T.ui8())
        )
        object_fifo_link(inOF_L3L2, inOF_L2L1)

        outOF_L2L3 = object_fifo("outOF_L2L3", M, S, 2, T.memref(256, T.ui8()))
        outOF_L1L2 = object_fifo("outOF_L1L2", T5, M, 2, T.memref(256, T.ui8()))
        object_fifo_link(outOF_L1L2, outOF_L2L3)

        OF_2to3 = object_fifo("OF_2to3", T2, T3, 4, T.memref(64, T.ui8()))
        OF_3to4 = object_fifo("OF_3to4", T3, T4, 2, T.memref(64, T.ui8()))
        OF_4to5 = object_fifo("OF_4to5", T4, T5, 2, T.memref(64, T.ui8()))
        OF_5to5 = object_fifo("OF_5to5", T5, T5, 1, T.memref(256, T.ui8()))

        @core(T2, "rgba2gray.cc.o")
        def core_body():
            for _ in range_(36):
                elem_in = inOF_L2L1.acquire(ObjectFifoPort.Consume, 1)
                elem_out = OF_2to3.acquire(ObjectFifoPort.Produce, 1)

                call(rgba2gray_line, [elem_in, elem_out, arith.constant(64)])

                inOF_L2L1.release(ObjectFifoPort.Consume, 1)
                OF_2to3.release(ObjectFifoPort.Produce, 1)
                yield_([])

        @core(T3, "filter2d.cc.o")
        def core_body():
            kernel = memref.alloc(3, 3, T.i16())
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
            elems_in_pre = OF_2to3.acquire(ObjectFifoPort.Consume, 2)
            elem_pre_out = OF_3to4.acquire(ObjectFifoPort.Produce, 1)
            call(
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
            OF_3to4.release(ObjectFifoPort.Produce, 1)

            # Steady State : Middle
            for _ in range_(1, 35):
                elems_in = OF_2to3.acquire(ObjectFifoPort.Consume, 3)
                elem_out = OF_3to4.acquire(ObjectFifoPort.Produce, 1)
                call(
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
                OF_2to3.release(ObjectFifoPort.Consume, 1)
                OF_3to4.release(ObjectFifoPort.Produce, 1)
                yield_([])

            # Postamble : Bottom Border
            elems_in_post = OF_2to3.acquire(ObjectFifoPort.Consume, 2)
            elem_post_out = OF_3to4.acquire(ObjectFifoPort.Produce, 1)
            call(
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
            OF_2to3.release(ObjectFifoPort.Consume, 2)
            OF_3to4.release(ObjectFifoPort.Produce, 1)

        @core(T4, "threshold.cc.o")
        def core_body():
            v_thr = arith.constant(10, T.i16())
            v_max = arith.constant(255, T.i16())
            v_typ = arith.constant(0, T.i8())

            for _ in range_(36):
                elem_in = OF_3to4.acquire(ObjectFifoPort.Consume, 1)
                elem_out = OF_4to5.acquire(ObjectFifoPort.Produce, 1)

                call(
                    threshold_line,
                    [elem_in, elem_out, arith.constant(64), v_thr, v_max, v_typ],
                )

                OF_3to4.release(ObjectFifoPort.Consume, 1)
                OF_4to5.release(ObjectFifoPort.Produce, 1)
                yield_([])

        @core(T5, "combined_gray2rgba_addWeighted.a")
        def core_body():
            for _ in range_(36):
                elem_in = OF_4to5.acquire(ObjectFifoPort.Consume, 1)
                elem_out = OF_5to5.acquire(ObjectFifoPort.Produce, 1)

                call(gray2rgba_line, [elem_in, elem_out, arith.constant(64)])

                OF_4to5.release(ObjectFifoPort.Consume, 1)
                OF_5to5.release(ObjectFifoPort.Produce, 1)

                elem_in1 = OF_5to5.acquire(ObjectFifoPort.Consume, 1)
                elem_in2 = inOF_L2L1.acquire(ObjectFifoPort.Consume, 1)
                elem_out2 = outOF_L1L2.acquire(ObjectFifoPort.Produce, 1)

                alpha = arith.constant(16384, T.i16())
                beta = arith.constant(16384, T.i16())
                gamma = arith.constant(0, T.i8())

                call(
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

                OF_5to5.release(ObjectFifoPort.Consume, 1)
                inOF_L2L1.release(ObjectFifoPort.Consume, 1)
                outOF_L1L2.release(ObjectFifoPort.Produce, 1)
                yield_([])

        @FuncOp.from_py_func(
            T.memref(2304, T.i32()), T.memref(2304, T.i32()), T.memref(2304, T.i32())
        )
        def sequence(I, B, O):
            npu_dma_memcpy_nd(
                metadata="outOF_L2L3",
                bd_id=0,
                mem=O,
                sizes=[1, 1, 36, 64],
                strides=[0, 0, 64],
            )
            npu_dma_memcpy_nd(
                metadata="inOF_L3L2",
                bd_id=1,
                mem=I,
                sizes=[1, 1, 36, 64],
                strides=[0, 0, 64],
            )
            npu_sync(column=0, row=0, direction=0, channel=0)

    assert module.operation.verify()


# CHECK-LABEL: my_add_one_objFifo
@construct_and_print_module
def my_add_one_objFifo(module):
    @device(AIEDevice.npu)
    def device_body():
        shim_tile = tile(0, 0)
        mem_tile = tile(0, 1)
        compute_tile2 = tile(0, 2)

        of_in0 = object_fifo("in0", shim_tile, mem_tile, 2, T.memref(16, T.i32()))
        of_in1 = object_fifo("in1", mem_tile, compute_tile2, 2, T.memref(8, T.i32()))
        object_fifo_link(of_in0, of_in1)

        of_out0 = object_fifo("out0", mem_tile, shim_tile, 2, T.memref(8, T.i32()))
        of_out1 = object_fifo("out1", compute_tile2, mem_tile, 2, T.memref(16, T.i32()))
        object_fifo_link(of_out1, of_out0)

        @core(compute_tile2)
        def core_body():
            # Effective while(1)
            for _ in range_(8):
                elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                for i in range_(8):
                    v0 = memref.load(elem_in, [i])
                    v1 = arith.addi(v0, arith.constant(1, T.i32()))
                    memref.store(v1, elem_out, [i])
                    yield_([])
                of_in1.release(ObjectFifoPort.Consume, 1)
                of_out1.release(ObjectFifoPort.Produce, 1)
                yield_([])

        @FuncOp.from_py_func(
            T.memref(64, T.i32()), T.memref(32, T.i32()), T.memref(64, T.i32())
        )
        def sequence(inTensor, notUsed, outTensor):
            npu_dma_memcpy_nd(
                metadata="out0", bd_id=0, mem=outTensor, sizes=[1, 1, 1, 64]
            )
            npu_dma_memcpy_nd(
                metadata="in0", bd_id=1, mem=inTensor, sizes=[1, 1, 1, 64]
            )
            npu_sync(column=0, row=0, direction=0, channel=0)

    assert module.operation.verify()
