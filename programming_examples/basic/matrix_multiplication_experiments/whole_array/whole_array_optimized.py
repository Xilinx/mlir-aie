#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
import argparse
from ml_dtypes import bfloat16
import numpy as np
import sys

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence

dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
}


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-N", type=int, default=512)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=32)
    argparser.add_argument("-mtk", type=int, default=512)
    argparser.add_argument("-ktn", type=int, default=512)
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4], default=4)
    argparser.add_argument("--n-aie-rows", type=int, choices=[1, 2, 4], default=4)
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16"
    )
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i16",
    )
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument(
        "--generate-taps",
        action="store_true",
        help="Generate TensorAccessPatterns, a Python object to represent each data transfer"
        "of the input/output matrices. These objects can be used for visualization.",
    )
    args = argparser.parse_args()
    with mlir_mod_ctx() as ctx:
        maybe_taps = my_matmul(
            args.M,
            args.K,
            args.N,
            args.m,
            args.k,
            args.n,
            args.mtk,
            args.ktn,
            args.n_aie_cols,
            args.n_aie_rows,
            args.dtype_in,
            args.dtype_out,
            args.b_col_maj,
            args.trace_size,
            args.generate_taps,
        )
        # print(ctx.module.operation.verify())
        print(ctx.module)

    if args.generate_taps:
        return maybe_taps


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(
    M,
    K,
    N,
    m,
    k,
    n,
    mtk,
    ktn,
    n_aie_cols,
    n_aie_rows,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    trace_size,
    generate_taps=False,
):
    
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    if dtype_in_str == "bf16":
        r = 4
        s = 8
        t = 4
    elif dtype_in_str == "i8":
        r = 4
        s = 8
        t = 8
    elif dtype_in_str == "i16":
        r = 4
        s = 4
        t = 4

    # We bring (m * n_aie_rows) tiles of matrix A (row-wise). Each tile size is m * mtk,
    # and is stored in a separate MemTile. Each tile is broadcasted accross rows.  
    assert (
        M % (m * n_aie_rows) == 0
    ), """A must be tileable into (m * n_aie_rows, k)-sized blocks"""

    # We bring (n * n_aie_cols) tiles of Matrix B (col-wise). Each tile size is either ktn * n, 
    # when bringing more contiguous data acrros 'K' and works only for col-maj order
    # Otherwise, just bring k * n tiles for row-maj order.
    assert (
        N % (n * n_aie_cols) == 0
    ), """B must be tileable into (k, n * n_aie_cols)-sized blocks"""

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0


    # K needs to be divisible by mtk
    assert K % mtk == 0

    # also mtk as a bigger block needs to be divisible by k
    assert mtk % k == 0


    # When using col-major we might need to bring bigger tiles
    if b_col_maj:        
        # K needs to be divisible by ktn
        assert K % ktn == 0

        # also ktn as a bigger block needs to be divisible by k
        assert ktn % k == 0
    else:
        # Just check if big K is divisible by small k
        assert K % k == 0
    
    
    # If you get errors during CDO generation due to running out of program
    # memory, it may be because too much code is generated due to ObjectFIFO
    # loop unrollings. Reducing the depth to 1 here will work around that at
    # a big performance cost.
    fifo_depth = 2

    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores


    # set whole array AIE device
    dev = AIEDevice.npu1_4col


    @device(dev)
    def device_body():

        # Load a bigger m * mtk block in MemTile
        # Special condition: (k = mtk) is also possible
        A_l2_ty = np.ndarray[(m * mtk,), np.dtype[dtype_in]] 

        # When B in col-maj load a bigger ktn * n block in MemTile
        # Special condition: (k = ktn) is also possible
        if b_col_maj:
            B_l2_ty = np.ndarray[(ktn * n,), np.dtype[dtype_in]]
        # When B in row-maj always load k * n
        # because loading more contiguous data is not possible
        else:
            B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
        
        C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
        A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

        # AIE Core Function declarations
        zero = external_func(f"zero_{dtype_out_str}", inputs=[C_l1_ty])
        matmul_vectorized_func_name = (
            f"matmul_{dtype_in_str}_{dtype_out_str}"
            if not b_col_maj
            else 
            f"matmul_{dtype_in_str}_{dtype_out_str}_b_col_maj"
        )
        matmul = external_func(
            matmul_vectorized_func_name,
            inputs=[A_l1_ty, B_l1_ty, C_l1_ty],
        )

        # Tile declarations as tile[row][col]
        tiles = [
            [tile(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)
        ]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]


        # AIE-array data movement with object fifos
        
        # Here the number of Object Fifos for matrix A is determined by 
        # the number of row "tiles", although we bring them using shim and mem tiles,
        # which are column-wise
        A_l3l2_fifos = [None] * n_aie_rows
        A_l2l1_fifos = [None] * n_aie_rows


        # For matrix B the number of Object Fifos is determined by 
        # the number of column "tiles", which in this case is equal to 
        # the number of AIE columns (n_aie_cols)
        B_l3l2_fifos = [None] * n_aie_cols
        B_l2l1_fifos = [None] * n_aie_cols

        C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3_fifos = [None] * n_aie_cols

        # Input A
        for row in range(n_aie_rows):

            # Shim to Mem Object Fifos
            A_l3l2_fifos[row] = object_fifo(
                f"A_L3L2_{row}",
                shim_tiles[row],
                mem_tiles[row],
                fifo_depth,
                A_l2_ty,
                None,
                # S2MM in MemTile to convert m * mtk block from row-maj
                # to tiled m*k blocks
                [
                   [
                       (m, k),
                       (mtk // k, m * k),
                       (k, 1),
                   ]
                ],
            )

            # Mem to Cores Object Fifos
            A_l2l1_fifos[row] = object_fifo(
                f"A_L2L1_{row}",
                mem_tiles[row],
                core_tiles[row][0:n_aie_cols],  # broadcast along one row, into n_aie_cols columns
                fifo_depth,
                A_l1_ty,
                # 4D MM2S expressing the (m * mtk) tile
                [
                    (mtk // k, m * k),
                    (k // s, s),
                    (m, k),
                    (s, 1),
                ],

                # S2MM in compute tiles, so each (m * k) tile 
                # have the correct data layout. 
                # We have broadcast into n_aie_cols compute tiles, so each needs it's own
                # S2MM transformation.
                [
                    [
                        (k // s, r * s),
                        (m // r, r * k),
                        (r * s, 1),
                    ] for _ in range(n_aie_cols)
                ],
            )
            
            # link Object Fifos
            object_fifo_link(A_l3l2_fifos[row], A_l2l1_fifos[row])

        

        # Input B
        for col in range(n_aie_cols):

            # Shim to Mem Object Fifos
            B_l3l2_fifos[col] = object_fifo(
                f"B_L3L2_{col}",
                shim_tiles[col],
                mem_tiles[col],
                fifo_depth,
                B_l2_ty,
                None,
                (
                    # S2MM in MemTile to convert ktn * n block from col-maj
                    # to tiled k*n blocks, if B in col-major
                    [
                        [
                            (n, k),
                            (ktn // k, k * n),
                            (k, 1),
                        ]
                    ]
                    if b_col_maj
                    else None
                ),      
            )

            # Mem to Cores Object Fifos
            B_l2l1_fifos[col] = object_fifo(
                f"B_L2L1_{col}",
                mem_tiles[col],
                [
                    core_tiles[j][col] for j in range(n_aie_rows)
                ],  # broadcast along one column
                fifo_depth,
                B_l1_ty,
                (
                    # 4D MM2S expressing the (ktn * n) tile
                    # when B in col-maj
                    [
                        (ktn // k, k * n),
                        (k // s, s),
                        (n, k),
                        (s, 1),
                    ]
                    if b_col_maj

                    # if B in row-maj
                    else [
                        (k // s, s * n),
                        (n // t, t),
                        (s, n),
                        (t, 1),
                    ]
                ),
                (
                    # S2MM in compute tiles, so eack (k * n) tile 
                    # have the correct data layout, if B in col-maj.
                    # We have broadcast into n_aie_cols compute tiles, so each needs it's own
                    # S2MM transformation.
                    [
                        [
                            (k // s, s * t),
                            (n // t, k * t),
                            (s * t, 1),
                        ] for _ in range(n_aie_rows)
                    ]
                    if b_col_maj
                    else None
                ),

            )
            object_fifo_link(B_l3l2_fifos[col], B_l2l1_fifos[col])


        # Output C
        for col in range(n_aie_cols):
            for row in range(n_aie_rows):
                C_l1l2_fifos[row][col] = object_fifo(
                    f"C_L1L2_{col}_{row}",
                    core_tiles[row][col],
                    mem_tiles[col],
                    fifo_depth,
                    C_l1_ty,
                )
            C_l2l3_fifos[col] = object_fifo(
                f"C_L2L3_{col}",
                mem_tiles[col],
                shim_tiles[col],
                fifo_depth,
                C_l2_ty,
                [
                    (m // r, r * n),
                    (r, t),
                    (n // t, r * t),
                    (t, 1),
                ],
            )
            if n_aie_rows > 1:
                of_offsets = [m * n * i for i in range(n_aie_rows)]
            else:
                of_offsets = []
            object_fifo_link(
                [C_l1l2_fifos[j][col] for j in range(n_aie_rows)],
                C_l2l3_fifos[col],
                of_offsets,
                [],
            )  # join along one column

        # Set up compute tiles
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):

                @core(core_tiles[row][col], f"mm_{m}x{k}x{n}.o")
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        loop = (
                            range_(n_tiles_per_core)
                            if n_tiles_per_core > 1
                            else range(1)
                        )  # Workaround for issue #1547
                        for _ in loop:
                            elem_out = C_l1l2_fifos[row][col].acquire(
                                ObjectFifoPort.Produce, 1
                            )
                            zero(elem_out)

                            for _ in range_(K // k):
                                elem_in_a = A_l2l1_fifos[row].acquire(
                                    ObjectFifoPort.Consume, 1
                                )
                                elem_in_b = B_l2l1_fifos[col].acquire(
                                    ObjectFifoPort.Consume, 1
                                )
                                matmul(elem_in_a, elem_in_b, elem_out)
                                A_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                                B_l2l1_fifos[col].release(ObjectFifoPort.Consume, 1)

                            C_l1l2_fifos[row][col].release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):
            # We are limited in the number of BDs. After synchronizing, we can reuse BDs.
            # We only transfer 4 rows of tiles at once before starting a new transfer block.
            tb_max_n_rows = (
                4  # tb = transfer block; block of transfers before sync call
            )
            for tb in range(ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
                for pingpong in [0, 1]:
                    M // m // n_aie_rows // tb_max_n_rows
                    row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                    bd_id_base = 8 * pingpong
                    tb_n_rows = min(
                        [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                    )
                    if tb_n_rows <= 0:
                        # for small input sizes, we may not even need a "pong" iteration
                        break
                    for col in range(n_aie_cols):

                        # C Output Transfer:
                        # The smallest transfer unit is a (m*n_aie_rows)-x-(n)-sized sub-tile of the matrix.
                        # Transfer one such tile for every (n_aie_cols)-th column, evenly spaced,
                        # then repeat that (tb_n_rows) times for the next contiguous blocks of rows.
                        # Each shim will start at a different column offset, transferring interleaved
                        # columns. For example, shim 0 may transfer the blocks marked 0 below, and shim 1
                        # may transfer the blocks marked 1.
                        #
                        #             N
                        #      ----------------
                        #     |0011    0011    |
                        #     |0011    0011    |
                        #     |0011    0011    |
                        # M   |0011    0011    |
                        #     |                |
                        #     |                |
                        #     |                |
                        #     |                |
                        #      ----------------
                        C_row_offset = row_base * m * n_aie_rows * N
                        C_col_offset = col * n
                        C_offset = C_col_offset + C_row_offset
                        C_sizes = [tb_n_rows, N // n // n_aie_cols, m * n_aie_rows, n]
                        C_strides = [m * n_aie_rows * N, n * n_aie_cols, N, 1]
                        npu_dma_memcpy_nd(
                            metadata=C_l2l3_fifos[col],
                            bd_id=bd_id_base,
                            mem=C,
                            offsets=[0, 0, 0, C_offset],
                            sizes=C_sizes,
                            strides=C_strides,
                        )
                        

                        for tile_row in range(tb_n_rows):

                            # A input transfer:
                            #
                            # The smallest transfer unit is a (m*mtk)-sized sub-tile of the input matrix.
                            # Transfer one such tile for every column, contiguously.
                            # Repeat this transfer with identical tiles a total of (N//n//n_aie_cols) times.
                            # Each shim transfers the tiles for separate rows. For example, shim 0 may transfer the
                            # tiles marked 0 below, and shim 1 may transfer the tiles marked 1.
                            #             K
                            #      ----------------
                            #     |0000000000000000|    (repeated N//n//n_aie_cols times)
                            #     |0000000000000000|
                            #     |1111111111111111|
                            # M   |1111111111111111|
                            #     |                |
                            #     |                |
                            #     |                |
                            #     |                |
                            #      ----------------
                            A_block_offset = (
                                (row_base + tile_row) * n_aie_rows * m * K
                            )  # base address for this transfer block for all BDs
                            A_row_offset = (
                                col * m * K
                            )  # base address for the shim in this column
                            A_offset = A_block_offset + A_row_offset


                            # sizes and strides in mtk representation
                            A_sizes = [N // n // n_aie_cols, K // mtk, m, mtk]
                            A_strides = [0, mtk, K, 1]


                            npu_dma_memcpy_nd(
                                metadata=A_l3l2_fifos[col],
                                bd_id=bd_id_base + 2 * tile_row + 1,
                                mem=A,
                                offsets=[0, 0, 0, A_offset],
                                sizes=A_sizes,
                                strides=A_strides,
                            )
                            
                            # B input transfer:
                            # Transfer the first a (n)-wide block of columns of B,
                            # Then transfer the (n_aie_columns)-th such block, and so on.
                            # Each shim will start at a different column offset.
                            # For example, shim 0 may transfer the tiles marked 0 below,
                            # and shim 1 may transfer the tiles marked 1.
                            #
                            #             N
                            #      ----------------
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            # K   |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #      ----------------

                            B_col_offset = col * n if not b_col_maj else col * n * K

                            if b_col_maj:
                                # ktn representation in col-maj
                                B_sizes = [N // n // n_aie_cols, K // ktn, n, ktn]
                                B_strides = [n * n_aie_cols * K, ktn, K, 1]
                            else:
                                B_sizes = [N // n // n_aie_cols, K // k, k, n]
                                B_strides = [n * n_aie_cols, k * N, N, 1]

                            npu_dma_memcpy_nd(
                                metadata=B_l3l2_fifos[col],
                                bd_id=bd_id_base + 2 * tile_row + 2,
                                mem=B,
                                offsets=[0, 0, 0, B_col_offset],
                                sizes=B_sizes,
                                strides=B_strides,
                            )
                            
                    if tb > 0 or (tb == 0 and pingpong > 0):
                        dma_wait(*C_l2l3_fifos)
            dma_wait(*C_l2l3_fifos)

if __name__ == "__main__":
    main()
