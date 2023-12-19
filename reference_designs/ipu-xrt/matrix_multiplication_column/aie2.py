#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import aie
from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.extras import memref, arith
from aie.util import mlir_mod_ctx


def my_matmul():
    M = 256
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

    n_cores = 4

    A_sz_in_i32s = M * K * word_size_in // 4
    B_sz_in_i32s = K * N * word_size_in // 4
    C_sz_in_bytes = M * N * word_size_out
    C_sz_in_i32s = C_sz_in_bytes // 4

    M_div_m = M // m
    M_div_m_div_n_cores = M // (m * n_cores)
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n // n_cores

    # Matrix A: MxK, submatrices a: mxk
    k_in_i32s = k * word_size_in // 4
    K_in_i32s = K * word_size_in // 4
    m_x_n_cores = m * n_cores

    # Matrix B: KxN, submatrices b: kxn
    n_in_i32s = n * word_size_in // 4
    N_in_i32s = N * word_size_in // 4

    # Output Matrix C: MxN
    n_in_i32s_out = n * word_size_out // 4
    N_in_i32s_out = N * word_size_out // 4
    m_x_n_cores_x_N_in_i32s_out = m * n_cores * N * word_size_out // 4

    vectorized = True

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():
            memRef_inA_ty = T.memref(m * k * n_cores, T.i16())
            memRef_inB_ty = T.memref(k * n * 1, T.i16())
            memRef_outC_ty = T.memref(m * n * n_cores, T.i16())
            memRef_A_ty = T.memref(m, k, T.i16())
            memRef_B_ty = T.memref(k, n, T.i16())
            memRef_C_ty = T.memref(m, n, T.i16())

            ofifo_memRef_inA_ty = TypeAttr.get(ObjectFifoType.get(memRef_inA_ty))
            ofifo_memRef_inB_ty = TypeAttr.get(ObjectFifoType.get(memRef_inB_ty))
            ofifo_memRef_outC_ty = TypeAttr.get(ObjectFifoType.get(memRef_outC_ty))
            ofifo_memRef_A_ty = TypeAttr.get(ObjectFifoType.get(memRef_A_ty))
            ofifo_memRef_B_ty = TypeAttr.get(ObjectFifoType.get(memRef_B_ty))
            ofifo_memRef_C_ty = TypeAttr.get(ObjectFifoType.get(memRef_C_ty))

            # AIE Core Function declarations
            zero_scalar = external_func("zero_scalar_i16", inputs=[memRef_C_ty])
            zero = external_func("zero_i16", inputs=[memRef_C_ty])
            matmul_scalar = external_func(
                "matmul_scalar_i16_i16", inputs=[memRef_A_ty, memRef_B_ty, memRef_C_ty]
            )
            matmul = external_func(
                "matmul_i16_i16", inputs=[memRef_A_ty, memRef_B_ty, memRef_C_ty]
            )

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)
            ComputeTile3 = tile(0, 3)
            ComputeTile4 = tile(0, 4)
            ComputeTile5 = tile(0, 5)
    
            cores = [ComputeTile2, ComputeTile3, ComputeTile4, ComputeTile5]
            inA_fifos = ["memA0",  "memA1",  "memA2",  "memA3" ]
            inB_fifos = ["memB" ]
            outC_fifos = ["memC0", "memC1", "memC2", "memC3"]

            # AIE-array data movement with object fifos
            # Input A
            objectfifo("inA", ShimTile, [MemTile], 2, ofifo_memRef_inA_ty, [], [])
            for i in range(n_cores):
                objectfifo(
                    inA_fifos[i],
                    MemTile,
                    [cores[i]],
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
            objectfifo_link(["inA"], inA_fifos[0:n_cores])

            # Input B
            objectfifo("inB", ShimTile, [MemTile], 2, ofifo_memRef_inB_ty, [], [])
            objectfifo(
                inB_fifos[0],
                MemTile,
                cores[0:n_cores],
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
            objectfifo_link(["inB"], [inB_fifos[0]])

            # Output C
            for i in range(n_cores) :
                objectfifo(
                    outC_fifos[i], cores[i], [MemTile], 2, ofifo_memRef_C_ty, [], []
                )
            objectfifo(
                "outC",
                MemTile,
                [ShimTile],
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
            objectfifo_link(outC_fifos[0:n_cores], ["outC"])

            # Set up compute tiles
            for i in range(n_cores):
                # Compute tile i
                @core(cores[i], "mm.o")
                def core_body():
                    for _ in for_(0xFFFFFFFF):
                        for _ in for_(tiles):
                            elem_out = acquire(
                                ObjectFifoPort.Produce, outC_fifos[i], 1, memRef_C_ty
                            ).acquired_elem()
                            Call(zero, [elem_out])

                            for _ in for_(K_div_k):
                                elem_in_a = acquire(
                                    ObjectFifoPort.Consume, inA_fifos[i], 1, memRef_A_ty
                                ).acquired_elem()
                                elem_in_b = acquire(
                                    ObjectFifoPort.Consume, inB_fifos[0], 1, memRef_B_ty
                                ).acquired_elem()
                                Call(matmul, [elem_in_a, elem_in_b, elem_out])
                                objectfifo_release(
                                    ObjectFifoPort.Consume, inA_fifos[i], 1
                                )
                                objectfifo_release(
                                    ObjectFifoPort.Consume, inB_fifos[0], 1
                                )
                                yield_([])

                            objectfifo_release(ObjectFifoPort.Produce, outC_fifos[i], 1)
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
                    (M_div_m_div_n_cores + rows_per_block - 1) // rows_per_block
                ):
                    C_row_offset_in_i32s = (
                        tile_row_block
                        * rows_per_block
                        * m * n_cores
                        * N
                        * word_size_out
                        // 4
                    )
                    num_tile_rows = min(
                        [
                            rows_per_block,
                            M_div_m_div_n_cores - tile_row_block * rows_per_block,
                        ]
                    )
                    ipu_dma_memcpy_nd(
                        metadata="outC",
                        bd_id=0,
                        mem=C,
                        offsets=[0, 0, 0, C_row_offset_in_i32s],
                        lengths=[num_tile_rows, N_div_n, m_x_n_cores, n_in_i32s_out],
                        strides=[
                            m_x_n_cores_x_N_in_i32s_out,
                            n_in_i32s_out,
                            N_in_i32s_out,
                        ],
                    )
                    for tile_row in range(num_tile_rows):
                        A_row_offset_in_i32s = (
                            ((tile_row_block * rows_per_block) + tile_row)
                            * m
                            * n_cores
                            * K
                            * word_size_in
                            // 4
                        )
                        ipu_dma_memcpy_nd(
                            metadata="inA",
                            bd_id=2 * tile_row + 1,
                            mem=A,
                            offsets=[0, 0, 0, A_row_offset_in_i32s],
                            lengths=[N_div_n, K_div_k, m_x_n_cores, k_in_i32s],
                            strides=[0, k_in_i32s, K_in_i32s],
                        )
                        ipu_dma_memcpy_nd(
                            metadata="inB",
                            bd_id=2 * tile_row + 2,
                            mem=B,
                            lengths=[1, N_div_n, K, n_in_i32s],
                            strides=[0, n_in_i32s, N_in_i32s],
                        )

                    ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_matmul()
