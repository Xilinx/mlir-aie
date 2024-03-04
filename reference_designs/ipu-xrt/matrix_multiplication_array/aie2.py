#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *


def my_matmul():
    M = 512
    K = 512
    N = 512
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
    tiles = M_div_m * N_div_n // n_rows
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

        @device(AIEDevice.ipu)
        def device_body():
            memRef_inA_ty = T.memref(m * k * n_rows, T.bf16())
            memRef_inB_ty = T.memref(k * n * 1, T.bf16())
            memRef_outC_ty = T.memref(m * n * n_rows, T.bf16())
            memRef_A_ty = T.memref(m, k, T.bf16())
            memRef_B_ty = T.memref(k, n, T.bf16())
            memRef_C_ty = T.memref(m, n, T.bf16())

            ofifo_memRef_inA_ty = TypeAttr.get(ObjectFifoType.get(memRef_inA_ty))
            ofifo_memRef_inB_ty = TypeAttr.get(ObjectFifoType.get(memRef_inB_ty))
            ofifo_memRef_outC_ty = TypeAttr.get(ObjectFifoType.get(memRef_outC_ty))
            ofifo_memRef_A_ty = TypeAttr.get(ObjectFifoType.get(memRef_A_ty))
            ofifo_memRef_B_ty = TypeAttr.get(ObjectFifoType.get(memRef_B_ty))
            ofifo_memRef_C_ty = TypeAttr.get(ObjectFifoType.get(memRef_C_ty))

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
            inA_fifos = ["inA0", "inA1", "inA2", "inA3"]
            inB_fifos = ["inB0", "inB1", "inB2", "inB3"]
            memA_fifos = ["memA0", "memA1", "memA2", "memA3"]
            memB_fifos = ["memB0", "memB1", "memB2", "memB3"]
            _0_outC_fifos = ["memC00", "memC10", "memC20", "memC30"]
            _1_outC_fifos = ["memC01", "memC11", "memC21", "memC31"]
            _2_outC_fifos = ["memC02", "memC12", "memC22", "memC32"]
            _3_outC_fifos = ["memC03", "memC13", "memC23", "memC33"]
            memC_fifos = [_0_outC_fifos, _1_outC_fifos, _2_outC_fifos, _3_outC_fifos]
            outC_fifos = ["outC0", "outC1", "outC2", "outC3"]

            # AIE-array data movement with object fifos
            # Input A
            for i in range(n_cols):
                objectfifo(
                    inA_fifos[i], shims[i], [mems[i]], 2, ofifo_memRef_inA_ty, [], []
                )
                objectfifo(
                    memA_fifos[i],
                    mems[i],
                    t_cores[i][0:n_cols],
                    2,
                    ofifo_memRef_A_ty,
                    [
                        (m // r, r * k * word_size_in // 4),
                        (k // s, s * word_size_in // 4),
                        (r, k * word_size_in // 4),
                        (s * word_size_in // 4, 1),
                    ],
                    [],
                )
                objectfifo_link([inA_fifos[i]], [memA_fifos[i]])

            # Input B
            for i in range(n_cols):
                objectfifo(
                    inB_fifos[i], shims[i], [mems[i]], 2, ofifo_memRef_inB_ty, [], []
                )
                objectfifo(
                    memB_fifos[i],
                    mems[i],
                    cores[i][0:n_rows],
                    2,
                    ofifo_memRef_B_ty,
                    [
                        (k // s, s * n * word_size_in // 4),
                        (n // t, t * word_size_in // 4),
                        (s, n * word_size_in // 4),
                        (t * word_size_in // 4, 1),
                    ],
                    [],
                )
                objectfifo_link([inB_fifos[i]], [memB_fifos[i]])

            # Output C
            for i in range(n_cols):
                for j in range(n_rows):
                    objectfifo(
                        memC_fifos[i][j],
                        cores[i][j],
                        [mems[i]],
                        2,
                        ofifo_memRef_C_ty,
                        [],
                        [],
                    )
                objectfifo(
                    outC_fifos[i],
                    mems[i],
                    shims[i],
                    2,
                    ofifo_memRef_outC_ty,
                    [
                        (m // r, r * n * word_size_out // 4),
                        (r, t * word_size_out // 4),
                        (n // t, r * t * word_size_out // 4),
                        (t * word_size_out // 4, 1),
                    ],
                    [],
                )
                objectfifo_link(memC_fifos[i], [outC_fifos[i]])

            # Set up compute tiles
            for j in range(n_cols):
                for i in range(n_rows):
                    # Compute tile i
                    @core(cores[j][i], "mm.o")
                    def core_body():
                        for _ in for_(0xFFFFFFFF):
                            for _ in for_(tiles):
                                elem_out = acquire(
                                    ObjectFifoPort.Produce,
                                    memC_fifos[j][i],
                                    1,
                                    memRef_C_ty,
                                ).acquired_elem()
                                Call(zero, [elem_out])

                                for _ in for_(K_div_k):
                                    elem_in_a = acquire(
                                        ObjectFifoPort.Consume,
                                        memA_fifos[i],
                                        1,
                                        memRef_A_ty,
                                    ).acquired_elem()
                                    elem_in_b = acquire(
                                        ObjectFifoPort.Consume,
                                        memB_fifos[j],
                                        1,
                                        memRef_B_ty,
                                    ).acquired_elem()
                                    Call(matmul, [elem_in_a, elem_in_b, elem_out])
                                    objectfifo_release(
                                        ObjectFifoPort.Consume, memA_fifos[i], 1
                                    )
                                    objectfifo_release(
                                        ObjectFifoPort.Consume, memB_fifos[j], 1
                                    )
                                    yield_([])

                                objectfifo_release(
                                    ObjectFifoPort.Produce, memC_fifos[j][i], 1
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
                        ipu_dma_memcpy_nd(
                            metadata=outC_fifos[i],
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
                            ipu_dma_memcpy_nd(
                                metadata=inA_fifos[i],
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
                            ipu_dma_memcpy_nd(
                                metadata=inB_fifos[i],
                                bd_id=2 * tile_row + 2,
                                mem=B,
                                offsets=[0, 0, 0, B_col_offset_in_i32s],
                                sizes=[N_div_n_div_n_cols, K_div_k, k, n_in_i32s],
                                strides=[n_x_n_cols_in_i32s, k_x_N_in_i32s, N_in_i32s],
                            )
                    for i in range(n_cols):
                        ipu_sync(column=i, row=0, direction=0, channel=0)

    # print(ctx.module.operation.verify())
    print(ctx.module)


my_matmul()
