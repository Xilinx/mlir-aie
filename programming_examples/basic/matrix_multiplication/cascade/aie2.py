#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
import numpy as np
import sys
import argparse

from aie.helpers.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.util import bfloat16

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
    argparser.add_argument("-n", type=int, default=64)
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4], default=4)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i16"], default="i16"
    )
    argparser.add_argument(
        "--dtype_out", type=str, choices=["bf16", "i16", "f32", "i32"], default="i32"
    )
    args = argparser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(
            args.M,
            args.K,
            args.N,
            args.m,
            args.k,
            args.n,
            args.n_aie_cols,
            args.dtype_in,
            args.dtype_out,
        )
        # print(ctx.module.operation.verify())
        print(ctx.module)


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(M, K, N, m, k, n, n_aie_cols, dtype_in_str, dtype_out_str):

    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

    if dtype_in_str == "bf16":
        r = 4
        s = 8
        t = 4
    elif dtype_in_str == "i16":
        r = 1
        s = 1
        t = 1

    # Input matrix A:
    # Conceptually, we divide input A into (m * n_rows, k)-sized blocks. These
    # blocks are _broadcast_ across AIE core columns, then _distributed_ across
    # rows, s.t. each of the n_rows compute cores in a column receives a
    # contiguous (m, k)-sized block of A.
    assert M % m == 0, """A must be tileable into (m, k * n_aie_rows)-sized blocks"""

    # Both A and B are tiled in the K dimension into size k.
    assert K % (k * n_aie_rows) == 0

    # Input matrix B:
    # Conceptually, we do the same as with A, but instead of broadcasting
    # across columns we broadcast across rows and distribute across columns.
    assert (
        N % (n * n_aie_cols) == 0
    ), """B must be tileable into (k * n_aie_rows, n * n_aie_cols)-sized blocks"""

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    # If you get errors during CDO generation due to running out of program
    # memory, it may be because too much code is generated due to ObjectFIFO
    # loop unrollings. Reducing the depth to 1 here will work around that at
    # a big performance cost.
    fifo_depth = 1

    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    n_A_tiles_per_shim = n_aie_rows // n_aie_cols

    dev = None
    if n_aie_cols == 1:
        dev = AIEDevice.npu1_1col
    elif n_aie_cols == 2:
        dev = AIEDevice.npu1_2col
    elif n_aie_cols == 4:
        dev = AIEDevice.npu1_4col

    @device(dev)
    def device_body():
        A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[(k * n * n_aie_rows,), np.dtype[dtype_in]]
        C_l2_ty = np.ndarray[(m * n,), np.dtype[dtype_out]]
        A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

        # AIE Core Function declarations
        zero_scalar = external_func(f"zero_scalar_{dtype_out_str}", inputs=[C_l1_ty])
        matmul_scalar_cascade_get_only = external_func(
            f"matmul_scalar_cascade_get_only_{dtype_in_str}_{dtype_out_str}",
            inputs=[A_l1_ty, B_l1_ty, C_l1_ty],
        )
        matmul_scalar_cascade_put_only = external_func(
            f"matmul_scalar_cascade_put_only_{dtype_in_str}_{dtype_out_str}",
            inputs=[A_l1_ty, B_l1_ty, C_l1_ty],
        )
        matmul_scalar_cascade_put_get = external_func(
            f"matmul_scalar_cascade_put_get_{dtype_in_str}_{dtype_out_str}",
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
        A_l3l2_fifos = [None] * n_aie_cols
        A_l2l1_fifos = [None] * n_aie_rows

        B_l3l2_fifos = [None] * n_aie_cols
        B_l2l1_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]

        C_l1l2_fifos = [None] * n_aie_cols
        C_l1l2_buffers = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3_fifos = [None] * n_aie_cols

        # Input A
        for row in range(n_aie_rows):
            A_l2l1_fifos[row] = object_fifo(
                f"A_L2L1_{row}",
                mem_tiles[row // n_A_tiles_per_shim],
                core_tiles[row][0:n_aie_cols],  # broadcast along one row
                fifo_depth,
                A_l1_ty,
                [
                    (m // r, r * k),
                    (k // s, s),
                    (r, k),
                    (s, 1),
                ],
            )
        for col in range(n_aie_cols):
            A_l3l2_fifos[col] = object_fifo(
                f"A_L3L2_{col}",
                shim_tiles[col],
                mem_tiles[col],
                fifo_depth,
                A_l2_ty,
            )
            # If n_cols == n_rows, n_A_tiles_per_shim is 1 and
            # this simply links a_l3l2_fifos[col] to a_l2l1_fifos[row] directly,
            # where col == row.
            # If n_cols < n_rows, each column receives multiple rows of
            # tiles; distribute it along rows of AIE cores.
            start_row = col * n_A_tiles_per_shim
            stop_row = start_row + n_A_tiles_per_shim
            # Calculate the offsets into the input/output data for the join/distribute
            of_offsets = [m * k * i for i in range(n_aie_cols)]
            object_fifo_link(
                A_l3l2_fifos[col],
                [A_l2l1_fifos[row] for row in range(start_row, stop_row)],
                [],
                of_offsets,
            )

        # Input B
        for col in range(n_aie_cols):
            B_l3l2_fifos[col] = object_fifo(
                f"B_L3L2_{col}",
                shim_tiles[col],
                mem_tiles[col],
                fifo_depth,
                B_l2_ty,
            )
            for row in range(n_aie_rows):
                B_l2l1_fifos[row][col] = object_fifo(
                    f"B_L2L1_{col}_{row}",
                    mem_tiles[col],
                    core_tiles[row][col],
                    fifo_depth,
                    B_l1_ty,
                    [
                        (k // s, s * n),
                        (n // t, t),
                        (s, n),
                        (t, 1),
                    ],
                )
            of_offsets = [n * k * i for i in range(n_aie_cols)]
            object_fifo_link(
                B_l3l2_fifos[col],
                [B_l2l1_fifos[row][col] for row in range(n_aie_rows)],
                [],
                of_offsets,
            )

        # Output C
        for col in range(n_aie_cols):
            for row in range(n_aie_rows):
                if row == 0:
                    C_l1l2_fifos[col] = object_fifo(
                        f"C_L1L2_{col}_{row}",
                        core_tiles[row][col],
                        mem_tiles[col],
                        fifo_depth,
                        C_l1_ty,
                    )
                else:
                    C_l1l2_buffers[row][col] = buffer(
                        core_tiles[row][col], [m, n], dtype_out, f"C_L1L2_{col}_{row}"
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
            object_fifo_link(
                C_l1l2_fifos[col], C_l2l3_fifos[col]
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
                            if row == 0:
                                elem_out = C_l1l2_fifos[col].acquire(
                                    ObjectFifoPort.Produce, 1
                                )
                            else:
                                elem_out = C_l1l2_buffers[row][col]

                            if row == 0:
                                zero_scalar(elem_out)

                            for _ in range_(K // k // n_aie_rows):
                                elem_in_a = A_l2l1_fifos[row].acquire(
                                    ObjectFifoPort.Consume, 1
                                )
                                elem_in_b = B_l2l1_fifos[row][col].acquire(
                                    ObjectFifoPort.Consume, 1
                                )
                                if row == 0:
                                    matmul_scalar_cascade_get_only(
                                        elem_in_a, elem_in_b, elem_out
                                    )
                                elif row == n_aie_rows - 1:
                                    matmul_scalar_cascade_put_only(
                                        elem_in_a, elem_in_b, elem_out
                                    )
                                else:
                                    matmul_scalar_cascade_put_get(
                                        elem_in_a, elem_in_b, elem_out
                                    )

                                A_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                                B_l2l1_fifos[row][col].release(
                                    ObjectFifoPort.Consume, 1
                                )

                            if row == 0:
                                C_l1l2_fifos[col].release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):
            # We are limited in the number of BDs. After synchronizing, we can reuse BDs.
            # We only transfer 5 rows of tiles at once before starting a new transfer block.
            tb_max_n_rows = (
                5  # tb = transfer block; block of transfers before sync call
            )
            for tb in range(ceildiv(M // m, tb_max_n_rows)):
                tb_n_rows = min([tb_max_n_rows, M // m - tb * tb_max_n_rows])
                C_row_offset = tb * tb_max_n_rows * m * N
                for col in range(n_aie_cols):
                    C_col_offset = col * n
                    C_offset = C_col_offset + C_row_offset
                    npu_dma_memcpy_nd(
                        metadata=C_l2l3_fifos[col],
                        bd_id=0,
                        mem=C,
                        offsets=[0, 0, 0, C_offset],
                        sizes=[tb_n_rows, N // n // n_aie_cols, m, n],
                        strides=[m * N, n * n_aie_cols, N, 1],
                    )
                    for tile_row in range(tb_n_rows):
                        A_block_offset = ((tb * tb_max_n_rows) + tile_row) * m * K
                        A_row_offset = col * n_A_tiles_per_shim * k
                        A_offset = A_block_offset + A_row_offset
                        B_col_offset = col * n
                        npu_dma_memcpy_nd(
                            metadata=A_l3l2_fifos[col],
                            bd_id=2 * tile_row + 1,
                            mem=A,
                            offsets=[0, 0, 0, A_offset],
                            sizes=[
                                N // n // n_aie_cols,
                                K // k // n_aie_rows,
                                m * n_A_tiles_per_shim,
                                k,
                            ],
                            strides=[0, k * n_aie_rows, K, 1],
                        )
                        npu_dma_memcpy_nd(
                            metadata=B_l3l2_fifos[col],
                            bd_id=2 * tile_row + 2,
                            mem=B,
                            offsets=[0, 0, 0, B_col_offset],
                            sizes=[
                                N // n // n_aie_cols,
                                K // k // n_aie_rows,
                                k * n_aie_rows,
                                n,
                            ],
                            strides=[n * n_aie_cols, k * n_aie_rows * N, N, 1],
                        )
                dma_wait(*C_l2l3_fifos)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
