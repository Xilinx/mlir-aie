#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx


def my_matmul():
    M = 128
    K = 128
    N = 128
    m = 64
    k = 32
    n = 64
    r = 4
    s = 4
    t = 4
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

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():
            memref_a_ty = T.memref(m, k, T.i16())
            memref_b_ty = T.memref(k, n, T.i16())
            memref_c_ty = T.memref(m, n, T.i16())

            # AIE Core Function declarations
            zero_scalar = external_func("zero_scalar_i16", inputs=[memref_c_ty])
            zero = external_func("zero_i16", inputs=[memref_c_ty])
            matmul_scalar = external_func(
                "matmul_scalar_i16_i16",
                inputs=[memref_a_ty, memref_b_ty, memref_c_ty],
            )
            matmul = external_func(
                "matmul_i16_i16", inputs=[memref_a_ty, memref_b_ty, memref_c_ty]
            )

            # Tile declarations
            shim_tile = tile(0, 0)
            mem_tile = tile(0, 1)
            compute_tile2 = tile(0, 2)

            # AIE-array data movement with object fifos
            # Input A
            inA = object_fifo("inA", shim_tile, mem_tile, 2, memref_a_ty)
            memA = object_fifo(
                "memA",
                mem_tile,
                compute_tile2,
                2,
                memref_a_ty,
                [
                    (m // r, r * k * word_size_in // 4),
                    (k // s, s * word_size_in // 4),
                    (r, k * word_size_in // 4),
                    (s * word_size_in // 4, 1),
                ],
            )
            object_fifo_link(inA, memA)

            # Input B
            inB = object_fifo("inB", shim_tile, mem_tile, 2, memref_b_ty)
            memB = object_fifo(
                "memB",
                mem_tile,
                compute_tile2,
                2,
                memref_b_ty,
                [
                    (k // s, s * n * word_size_in // 4),
                    (n // t, t * word_size_in // 4),
                    (s, n * word_size_in // 4),
                    (t * word_size_in // 4, 1),
                ],
            )
            object_fifo_link(inB, memB)

            # Output C
            memC = object_fifo("memC", compute_tile2, mem_tile, 2, memref_c_ty)
            outC = object_fifo(
                "outC",
                mem_tile,
                shim_tile,
                2,
                memref_c_ty,
                [
                    (m // r, r * n * word_size_out // 4),
                    (r, t * word_size_out // 4),
                    (n // t, r * t * word_size_out // 4),
                    (t * word_size_out // 4, 1),
                ],
            )
            object_fifo_link(memC, outC)

            # Set up compute tiles

            # Compute tile 2
            @core(compute_tile2, "mm.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    for _ in for_(tiles):
                        elem_out = memC.acquire(1)
                        if vectorized:
                            call(zero, [elem_out])
                        else:
                            call(zero_scalar, [elem_out])

                        for _ in for_(K_div_k):
                            elem_in_a = memA.acquire(1)
                            elem_in_b = memB.acquire(1)
                            if vectorized:
                                call(matmul, [elem_in_a, elem_in_b, elem_out])
                            else:
                                call(matmul_scalar, [elem_in_a, elem_in_b, elem_out])
                            memA.release(1)
                            memB.release(1)
                            yield_([])

                        memC.release(1)
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
                        ipu_dma_memcpy_nd(
                            metadata="inA",
                            bd_id=2 * tile_row + 1,
                            mem=A,
                            offsets=[0, 0, 0, A_row_offset_in_i32s],
                            sizes=[N_div_n, K_div_k, m, k_in_i32s],
                            strides=[0, k_in_i32s, K_in_i32s],
                        )
                        ipu_dma_memcpy_nd(
                            metadata="inB",
                            bd_id=2 * tile_row + 2,
                            mem=B,
                            sizes=[N_div_n, K_div_k, k, n_in_i32s],
                            strides=[n_in_i32s, k_x_N_in_i32s, N_in_i32s],
                        )

                    ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_matmul()
