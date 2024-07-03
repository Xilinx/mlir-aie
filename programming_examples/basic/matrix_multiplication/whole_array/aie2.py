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
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=64)
    args = argparser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n)
        # print(ctx.module.operation.verify())
        print(ctx.module)


def my_matmul(M, K, N, m, k, n):
    r = 4
    s = 8
    t = 4

    n_rows = 4
    n_cols = 4
    n_cores = n_rows * n_cols

    # Input matrix A:
    # Conceptually, we divide input A into (m * n_rows, k)-sized blocks. These
    # blocks are _broadcast_ across AIE core columns, then _distributed_ across
    # rows, s.t. each of the n_rows compute cores in a column receives a
    # contiguous (m, k)-sized block of A.
    assert (
        M % (m * n_rows) == 0
    ), """A must be tileable into (m * n_rows, k)-sized blocks"""

    # Both A and B are tiled in the K dimension into size k.
    assert K % k == 0

    # Input matrix B:
    # Conceptually, we do the same as with A, but instead of broadcasting
    # across columns we broadcast across rows and distribute across columns.
    assert (
        N % (n * n_cols) == 0
    ), """B must be tileable into (k, n * n_cols)-sized blocks"""

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    # If you get errors during CDO generation due to running out of program
    # memory, it may be because too much code is generated due to ObjectFIFO
    # loop unrollings. Reducing the depth to 1 here will work around that at
    # a big performance cost.
    fifo_depth = 2

    n_tiles = (M // m) * (N // n) // n_cores

    @device(AIEDevice.npu1_4col)
    def device_body():
        a_l2_memref_ty = T.memref(m * k, T.bf16())
        b_l2_memref_ty = T.memref(k * n, T.bf16())
        c_l2_memref_ty = T.memref(m * n * n_rows, T.bf16())
        a_l1_memref_ty = T.memref(m, k, T.bf16())
        b_l1_memref_ty = T.memref(k, n, T.bf16())
        c_l1_memref_ty = T.memref(m, n, T.bf16())

        # AIE Core Function declarations
        zero_scalar = external_func("zero_scalar_bf16", inputs=[c_l1_memref_ty])
        zero = external_func("zero_bf16", inputs=[c_l1_memref_ty])
        matmul_scalar = external_func(
            "matmul_scalar_bf16_bf16",
            inputs=[a_l1_memref_ty, b_l1_memref_ty, c_l1_memref_ty],
        )
        matmul = external_func(
            "matmul_bf16_bf16", inputs=[a_l1_memref_ty, b_l1_memref_ty, c_l1_memref_ty]
        )

        # Tile declarations as tile[row][col]
        tiles = [[tile(col, row) for col in range(0, 4)] for row in range(0, 6)]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        # AIE-array data movement with object fifos
        a_l3l2_fifos = [None] * n_cols
        a_l2l1_fifos = [None] * n_rows

        b_l3l2_fifos = [None] * n_cols
        b_l2l1_fifos = [None] * n_cols

        c_l1l2_fifos = [[None] * n_cols for _ in range(n_rows)]
        c_l2l3_fifos = [None] * n_cols

        # Input A
        for col in range(n_cols):
            a_l3l2_fifos[col] = object_fifo(
                f"A_L3L2_{col}",
                shim_tiles[col],
                mem_tiles[col],
                fifo_depth,
                a_l2_memref_ty,
            )
            a_l2l1_fifos[col] = object_fifo(  # TODO i --> j (n rows)
                f"A_L2L1_{col}",
                mem_tiles[col],
                core_tiles[col][0:n_cols],  # broadcast along one row
                fifo_depth,
                a_l1_memref_ty,
                [
                    (m // r, r * k),
                    (k // s, s),
                    (r, k),
                    (s, 1),
                ],
            )
            object_fifo_link(a_l3l2_fifos[col], a_l2l1_fifos[col])

        # Input B
        for col in range(n_cols):
            b_l3l2_fifos[col] = object_fifo(
                f"B_L3L2_{col}",
                shim_tiles[col],
                mem_tiles[col],
                fifo_depth,
                b_l2_memref_ty,
            )
            b_l2l1_fifos[col] = object_fifo(
                f"B_L2L1_{col}",
                mem_tiles[col],
                [
                    core_tiles[j][col] for j in range(n_rows)
                ],  # broadcast along one column
                fifo_depth,
                b_l1_memref_ty,
                [
                    (k // s, s * n),
                    (n // t, t),
                    (s, n),
                    (t, 1),
                ],
            )
            object_fifo_link(b_l3l2_fifos[col], b_l2l1_fifos[col])

        # Output C
        for col in range(n_cols):
            for row in range(n_rows):
                c_l1l2_fifos[row][col] = object_fifo(
                    f"C_L1L2_{col}_{row}",
                    core_tiles[row][col],
                    mem_tiles[col],
                    fifo_depth,
                    c_l1_memref_ty,
                )
            c_l2l3_fifos[col] = object_fifo(
                f"C_L2L3_{col}",
                mem_tiles[col],
                shim_tiles[col],
                fifo_depth,
                c_l2_memref_ty,
                [
                    (m // r, r * n),
                    (r, t),
                    (n // t, r * t),
                    (t, 1),
                ],
            )
            object_fifo_link(
                [c_l1l2_fifos[j][col] for j in range(n_rows)], c_l2l3_fifos[col]
            )  # join along one column

        # Set up compute tiles
        for row in range(n_rows):
            for col in range(n_cols):

                @core(core_tiles[row][col], f"mm_{m}x{k}x{n}.o")
                def core_body():
                    for _ in for_(0xFFFFFFFF):
                        loop = (
                            for_(n_tiles) if n_tiles > 1 else range(1)
                        )  # Workaround for issue #1547
                        for _ in loop:
                            elem_out = c_l1l2_fifos[row][col].acquire(
                                ObjectFifoPort.Produce, 1
                            )
                            call(zero, [elem_out])

                            for _ in for_(K // k):
                                elem_in_a = a_l2l1_fifos[row].acquire(
                                    ObjectFifoPort.Consume, 1
                                )
                                elem_in_b = b_l2l1_fifos[col].acquire(
                                    ObjectFifoPort.Consume, 1
                                )
                                call(matmul, [elem_in_a, elem_in_b, elem_out])
                                a_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                                b_l2l1_fifos[col].release(ObjectFifoPort.Consume, 1)
                                yield_([])

                            c_l1l2_fifos[row][col].release(ObjectFifoPort.Produce, 1)
                            yield_([])

                        if n_tiles > 1:  # workaround for issue #1547
                            yield_([])

        # To/from AIE-array data movement
        @FuncOp.from_py_func(
            T.memref(M * K, T.bf16()),
            T.memref(K * N, T.bf16()),
            T.memref(M * N, T.bf16()),
        )
        def sequence(A, B, C):
            # only do 5 tile rows at a time before synchronizing, so we can reuse BDs
            rows_per_block = 5
            for tile_row_block in range(
                (M // m // n_rows + rows_per_block - 1) // rows_per_block
            ):
                num_tile_rows = min(
                    [
                        rows_per_block,
                        M // m // n_rows - tile_row_block * rows_per_block,
                    ]
                )
                C_row_offset = tile_row_block * rows_per_block * m * n_rows * N
                for col in range(n_cols):
                    C_col_offset = col * n
                    C_offset = C_col_offset + C_row_offset
                    npu_dma_memcpy_nd(
                        metadata=c_l2l3_fifos[col].sym_name.value,
                        bd_id=0,
                        mem=C,
                        offsets=[0, 0, 0, C_offset],
                        sizes=[num_tile_rows, N // n // n_cols, m * n_rows, n],
                        strides=[m * n_rows * N, n * n_cols, N, 1],
                    )
                    for tile_row in range(num_tile_rows):
                        A_row_offset = (
                            ((tile_row_block * rows_per_block) + tile_row)
                            * n_rows
                            * m
                            * K
                        )
                        A_col_offset = col * m * K
                        A_offset = A_row_offset + A_col_offset
                        B_col_offset = col * n
                        npu_dma_memcpy_nd(
                            metadata=a_l3l2_fifos[col].sym_name.value,
                            bd_id=2 * tile_row + 1,
                            mem=A,
                            offsets=[0, 0, 0, A_offset],
                            sizes=[N // n // n_cols, K // k, m, k],
                            strides=[0, k, K, 1],
                        )
                        npu_dma_memcpy_nd(
                            metadata=b_l3l2_fifos[col].sym_name.value,
                            bd_id=2 * tile_row + 2,
                            mem=B,
                            offsets=[0, 0, 0, B_col_offset],
                            sizes=[N // n // n_cols, K // k, k, n],
                            strides=[n * n_cols, k * N, N, 1],
                        )
                for col in range(n_cols):
                    npu_sync(column=col, row=0, direction=0, channel=0)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
