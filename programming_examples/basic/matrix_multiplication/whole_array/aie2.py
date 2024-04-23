#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys
import argparse

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-N", type=int, default=512)
    args = argparser.parse_args()
    my_matmul(args.M, args.K, args.N)


def my_matmul(M=512, K=512, N=512):
    m = 64
    k = 64
    n = 64
    r = 4
    s = 8
    t = 4
    word_size_in = 2
    word_size_out = 2

    n_rows = 4
    n_cols = 4
    n_cores = n_rows * n_cols

    A_sz_in_i32s = M * K * word_size_in // 4
    B_sz_in_i32s = K * N * word_size_in // 4
    C_sz_in_bytes = M * N * word_size_out
    C_sz_in_i32s = C_sz_in_bytes // 4

    M_div_m = M // m
    M_div_m_div_n_rows = M // (m * n_rows)
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n // n_cores
    N_div_n_div_n_cols = N_div_n // n_cols

    # Matrix A: MxK, submatrices a: mxk
    k_in_i32s = k * word_size_in // 4
    K_in_i32s = K * word_size_in // 4
    m_x_n_rows = m * n_rows

    # Matrix B: KxN, submatrices b: kxn
    n_in_i32s = n * word_size_in // 4
    N_in_i32s = N * word_size_in // 4
    k_x_N_in_i32s = k * N * word_size_in // 4
    n_x_n_cols_in_i32s = n_in_i32s * n_cols

    # Output Matrix C: MxN
    n_in_i32s_out = n * word_size_out // 4
    N_in_i32s_out = N * word_size_out // 4
    m_x_n_rows_x_N_in_i32s_out = m * n_rows * N_in_i32s_out
    n_x_n_cols_in_i32s_out = n_in_i32s_out * n_cols

    vectorized = True

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu)
        def device_body():
            memRef_inA_ty = T.memref(m * k, T.bf16())
            memRef_inB_ty = T.memref(k * n, T.bf16())
            memRef_outC_ty = T.memref(m * n * n_rows, T.bf16())
            memRef_A_ty = T.memref(m, k, T.bf16())
            memRef_B_ty = T.memref(k, n, T.bf16())
            memRef_C_ty = T.memref(m, n, T.bf16())

            # AIE Core Function declarations
            zero_scalar = external_func("zero_scalar_bf16", inputs=[memRef_C_ty])
            zero = external_func("zero_bf16", inputs=[memRef_C_ty])
            matmul_scalar = external_func(
                "matmul_scalar_bf16_bf16",
                inputs=[memRef_A_ty, memRef_B_ty, memRef_C_ty],
            )
            matmul = external_func(
                "matmul_bf16_bf16", inputs=[memRef_A_ty, memRef_B_ty, memRef_C_ty]
            )

            # Tile declarations
            _0_ShimTile = tile(0, 0)
            _0_MemTile = tile(0, 1)
            _0_ComputeTile2 = tile(0, 2)
            _0_ComputeTile3 = tile(0, 3)
            _0_ComputeTile4 = tile(0, 4)
            _0_ComputeTile5 = tile(0, 5)
            _0_cores = [
                _0_ComputeTile2,
                _0_ComputeTile3,
                _0_ComputeTile4,
                _0_ComputeTile5,
            ]
            _1_ShimTile = tile(1, 0)
            _1_MemTile = tile(1, 1)
            _1_ComputeTile2 = tile(1, 2)
            _1_ComputeTile3 = tile(1, 3)
            _1_ComputeTile4 = tile(1, 4)
            _1_ComputeTile5 = tile(1, 5)
            _1_cores = [
                _1_ComputeTile2,
                _1_ComputeTile3,
                _1_ComputeTile4,
                _1_ComputeTile5,
            ]
            _2_ShimTile = tile(2, 0)
            _2_MemTile = tile(2, 1)
            _2_ComputeTile2 = tile(2, 2)
            _2_ComputeTile3 = tile(2, 3)
            _2_ComputeTile4 = tile(2, 4)
            _2_ComputeTile5 = tile(2, 5)
            _2_cores = [
                _2_ComputeTile2,
                _2_ComputeTile3,
                _2_ComputeTile4,
                _2_ComputeTile5,
            ]
            _3_ShimTile = tile(3, 0)
            _3_MemTile = tile(3, 1)
            _3_ComputeTile2 = tile(3, 2)
            _3_ComputeTile3 = tile(3, 3)
            _3_ComputeTile4 = tile(3, 4)
            _3_ComputeTile5 = tile(3, 5)
            _3_cores = [
                _3_ComputeTile2,
                _3_ComputeTile3,
                _3_ComputeTile4,
                _3_ComputeTile5,
            ]
            shims = [_0_ShimTile, _1_ShimTile, _2_ShimTile, _3_ShimTile]
            mems = [_0_MemTile, _1_MemTile, _2_MemTile, _3_MemTile]
            cores = [
                [_0_ComputeTile2, _0_ComputeTile3, _0_ComputeTile4, _0_ComputeTile5],
                [_1_ComputeTile2, _1_ComputeTile3, _1_ComputeTile4, _1_ComputeTile5],
                [_2_ComputeTile2, _2_ComputeTile3, _2_ComputeTile4, _2_ComputeTile5],
                [_3_ComputeTile2, _3_ComputeTile3, _3_ComputeTile4, _3_ComputeTile5],
            ]
            t_cores = [
                [cores[j][i] for j in range(len(cores))] for i in range(len(cores[0]))
            ]
            inA_fifo_names = ["inA0", "inA1", "inA2", "inA3"]
            inA_fifos = {}
            inB_fifo_names = ["inB0", "inB1", "inB2", "inB3"]
            inB_fifos = {}
            memA_fifo_names = ["memA0", "memA1", "memA2", "memA3"]
            memA_fifos = {}
            memB_fifo_names = ["memB0", "memB1", "memB2", "memB3"]
            memB_fifos = {}
            _0_outC_fifo_names = ["memC00", "memC10", "memC20", "memC30"]
            _0_outC_fifos = {}
            _1_outC_fifo_names = ["memC01", "memC11", "memC21", "memC31"]
            _1_outC_fifos = {}
            _2_outC_fifo_names = ["memC02", "memC12", "memC22", "memC32"]
            _2_outC_fifos = {}
            _3_outC_fifo_names = ["memC03", "memC13", "memC23", "memC33"]
            _3_outC_fifos = {}
            memC_fifo_names = [
                _0_outC_fifo_names,
                _1_outC_fifo_names,
                _2_outC_fifo_names,
                _3_outC_fifo_names,
            ]
            memC_fifos = [_0_outC_fifos, _1_outC_fifos, _2_outC_fifos, _3_outC_fifos]
            outC_fifo_names = ["outC0", "outC1", "outC2", "outC3"]
            outC_fifos = {}

            # AIE-array data movement with object fifos
            # Input A
            for i in range(n_cols):
                inA_fifos[inA_fifo_names[i]] = object_fifo(
                    inA_fifo_names[i],
                    shims[i],
                    mems[i],
                    2,
                    memRef_inA_ty,
                )
                memA_fifos[memA_fifo_names[i]] = object_fifo(
                    memA_fifo_names[i],
                    mems[i],
                    t_cores[i][0:n_cols],
                    2,
                    memRef_A_ty,
                    [
                        (m // r, r * k),
                        (k // s, s),
                        (r, k),
                        (s, 1),
                    ],
                )
                object_fifo_link(inA_fifo_names[i], memA_fifo_names[i])

            # Input B
            for i in range(n_cols):
                inB_fifos[inB_fifo_names[i]] = object_fifo(
                    inB_fifo_names[i],
                    shims[i],
                    mems[i],
                    2,
                    memRef_inB_ty,
                )
                memB_fifos[memB_fifo_names[i]] = object_fifo(
                    memB_fifo_names[i],
                    mems[i],
                    cores[i][0:n_rows],
                    2,
                    memRef_B_ty,
                    [
                        (k // s, s * n),
                        (n // t, t),
                        (s, n),
                        (t, 1),
                    ],
                )
                object_fifo_link(inB_fifo_names[i], memB_fifo_names[i])

            # Output C
            for i in range(n_cols):
                for j in range(n_rows):
                    memC_fifos[i][memC_fifo_names[i][j]] = object_fifo(
                        memC_fifo_names[i][j],
                        cores[i][j],
                        mems[i],
                        2,
                        memRef_C_ty,
                    )
                outC_fifos[outC_fifo_names[i]] = object_fifo(
                    outC_fifo_names[i],
                    mems[i],
                    shims[i],
                    2,
                    memRef_outC_ty,
                    [
                        (m // r, r * n),
                        (r, t),
                        (n // t, r * t),
                        (t, 1),
                    ],
                )
                object_fifo_link(memC_fifo_names[i], outC_fifo_names[i])

            # Set up compute tiles
            for j in range(n_cols):
                for i in range(n_rows):
                    # Compute tile i
                    @core(cores[j][i], "mm.o")
                    def core_body():
                        for _ in for_(0xFFFFFFFF):
                            for _ in for_(tiles):
                                elem_out = memC_fifos[j][memC_fifo_names[j][i]].acquire(
                                    ObjectFifoPort.Produce,
                                    1,
                                )
                                call(zero, [elem_out])

                                for _ in for_(K_div_k):
                                    elem_in_a = memA_fifos[memA_fifo_names[i]].acquire(
                                        ObjectFifoPort.Consume,
                                        1,
                                    )
                                    elem_in_b = memB_fifos[memB_fifo_names[j]].acquire(
                                        ObjectFifoPort.Consume,
                                        1,
                                    )
                                    call(matmul, [elem_in_a, elem_in_b, elem_out])
                                    memA_fifos[memA_fifo_names[i]].release(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    memB_fifos[memB_fifo_names[j]].release(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    yield_([])

                                memC_fifos[j][memC_fifo_names[j][i]].release(
                                    ObjectFifoPort.Produce, 1
                                )
                                yield_([])
                            yield_([])

            # To/from AIE-array data movement

            @FuncOp.from_py_func(
                T.memref(A_sz_in_i32s, T.i32()),
                T.memref(B_sz_in_i32s, T.i32()),
                T.memref(C_sz_in_i32s, T.i32()),
            )
            def sequence(A, B, C):
                # only do 5 tile rows at a time before synchronizing, so we can reuse BDs
                rows_per_block = 5
                for tile_row_block in range(
                    (M_div_m_div_n_rows + rows_per_block - 1) // rows_per_block
                ):
                    num_tile_rows = min(
                        [
                            rows_per_block,
                            M_div_m_div_n_rows - tile_row_block * rows_per_block,
                        ]
                    )
                    C_row_offset = (
                        tile_row_block * rows_per_block * m * n_rows * N * word_size_out
                    )
                    for i in range(n_cols):
                        C_col_offset = i * n * word_size_out
                        C_offset_in_i32s = (C_col_offset + C_row_offset) // 4
                        npu_dma_memcpy_nd(
                            metadata=outC_fifo_names[i],
                            bd_id=0,
                            mem=C,
                            offsets=[0, 0, 0, C_offset_in_i32s],
                            sizes=[
                                num_tile_rows,
                                N_div_n_div_n_cols,
                                m_x_n_rows,
                                n_in_i32s_out,
                            ],
                            strides=[
                                m_x_n_rows_x_N_in_i32s_out,
                                n_x_n_cols_in_i32s_out,
                                N_in_i32s_out,
                            ],
                        )
                        for tile_row in range(num_tile_rows):
                            A_row_offset_in_i32s = (
                                ((tile_row_block * rows_per_block) + tile_row)
                                * n_rows
                                * m
                                * K
                                * word_size_in
                                // 4
                            )
                            A_col_offset_in_i32s = i * m * K * word_size_in // 4
                            B_col_offset_in_i32s = i * n * word_size_in // 4
                            npu_dma_memcpy_nd(
                                metadata=inA_fifo_names[i],
                                bd_id=2 * tile_row + 1,
                                mem=A,
                                offsets=[
                                    0,
                                    0,
                                    0,
                                    A_col_offset_in_i32s + A_row_offset_in_i32s,
                                ],
                                sizes=[N_div_n_div_n_cols, K_div_k, m, k_in_i32s],
                                strides=[0, k_in_i32s, K_in_i32s],
                            )
                            npu_dma_memcpy_nd(
                                metadata=inB_fifo_names[i],
                                bd_id=2 * tile_row + 2,
                                mem=B,
                                offsets=[0, 0, 0, B_col_offset_in_i32s],
                                sizes=[N_div_n_div_n_cols, K_div_k, k, n_in_i32s],
                                strides=[n_x_n_cols_in_i32s, k_x_N_in_i32s, N_in_i32s],
                            )
                    for i in range(n_cols):
                        npu_sync(column=i, row=0, direction=0, channel=0)

    # print(ctx.module.operation.verify())
    print(ctx.module)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
