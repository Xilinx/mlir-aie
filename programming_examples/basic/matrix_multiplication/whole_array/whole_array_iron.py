#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import argparse
from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU1Col2, NPU1Col4, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D

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
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4], default=4)
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
    maybe_module = my_matmul(
        args.M,
        args.K,
        args.N,
        args.m,
        args.k,
        args.n,
        args.n_aie_cols,
        args.dtype_in,
        args.dtype_out,
        args.b_col_maj,
        args.trace_size,
        args.generate_taps,
    )
    if args.generate_taps:
        return maybe_module
    else:
        print(maybe_module)


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(
    M,
    K,
    N,
    m,
    k,
    n,
    n_aie_cols,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    trace_size,
    generate_taps=False,
):
    n_aie_rows = 4
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

    # Input matrix A:
    # Conceptually, we divide input A into (m * n_rows, k)-sized blocks. These
    # blocks are _broadcast_ across AIE core columns, then _distributed_ across
    # rows, s.t. each of the n_rows compute cores in a column receives a
    # contiguous (m, k)-sized block of A.
    assert (
        M % (m * n_aie_rows) == 0
    ), """A must be tileable into (m * n_aie_rows, k)-sized blocks"""

    # Both A and B are tiled in the K dimension into size k.
    assert K % k == 0

    # Input matrix B:
    # Conceptually, we do the same as with A, but instead of broadcasting
    # across columns we broadcast across rows and distribute across columns.
    assert (
        N % (n * n_aie_cols) == 0
    ), """B must be tileable into (k, n * n_aie_cols)-sized blocks"""

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    if b_col_maj:
        # These assertions are probably too broad.
        assert m % 32 == 0
        assert k % 32 == 0
        assert n % 32 == 0

    # If you get errors during CDO generation due to running out of program
    # memory, it may be because too much code is generated due to ObjectFIFO
    # loop unrollings. Reducing the depth to 1 here will work around that at
    # a big performance cost.
    fifo_depth = 2

    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    n_A_tiles_per_shim = n_aie_rows // n_aie_cols

    dev = None
    if n_aie_cols == 1:
        dev = NPU1Col1()
    elif n_aie_cols == 2:
        dev = NPU1Col2()
    elif n_aie_cols == 4:
        dev = NPU1Col4()

    # These will hold TensorAccessPattern objects that represent the runtime
    # npu_dma_memcpy_nd operations of this design. They are only used if generate_taps is true
    A_taps = []
    B_taps = []
    C_taps = []

    # Define tensor types
    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    # AIE Core Function declarations
    zero_kernel = Kernel(f"zero_{dtype_out_str}", f"mm_{m}x{k}x{n}.o", [C_l1_ty])
    matmul_vectorized_func_name = (
        f"matmul_{dtype_in_str}_{dtype_out_str}"
        if not b_col_maj
        else f"matmul_{dtype_in_str}_{dtype_out_str}_b_col_maj"
    )
    matmul_kernel = Kernel(
        matmul_vectorized_func_name,
        f"mm_{m}x{k}x{n}.o",
        [A_l1_ty, B_l1_ty, C_l1_ty],
    )

    # Tile declarations as tile[row][col]
    tiles = [[(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    # AIE-array data movement with object fifos
    A_l3l2_fifos = [None] * n_aie_cols
    A_l2l1_fifos = [None] * n_aie_rows

    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols

    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    # Input A
    for col in range(n_aie_cols):
        A_l3l2_fifos[col] = ObjectFifo(
            A_l2_ty, name=f"A_L3L2_{col}", default_depth=fifo_depth
        )
        # If n_cols == n_rows, n_A_tiles_per_shim is 1 and
        # this simply links a_l3l2_fifos[col] to a_l2l1_fifos[row] directly,
        # where col == row.
        # If n_cols < n_rows, each column receives multiple rows of
        # tiles; distribute it along rows of AIE cores.
        start_row = col * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * i for i in range(stop_row - start_row)]
        dims_to_stream = [
            [
                (m // r, r * k),
                (k // s, s),
                (r, k),
                (s, 1),
            ]
        ] * (stop_row - start_row)
        a_tmp_fifos = (
            A_l3l2_fifos[col]
            .cons()
            .split(
                of_offsets,
                obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
                dims_to_stream=dims_to_stream,
                placement=Tile(col, 1),
            )
        )

        for i in range(stop_row - start_row):
            A_l2l1_fifos[i + start_row] = a_tmp_fifos[i]

        # Input B
        B_l3l2_fifos[col] = ObjectFifo(
            B_l2_ty, name=f"B_L3L2_{col}", default_depth=fifo_depth
        )
        if b_col_maj:
            dims_to_stream = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            dims_to_stream = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos[col] = (
            B_l3l2_fifos[col]
            .cons()
            .forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=dims_to_stream,
                placement=Tile(col, 1),
            )
        )

        # Output C
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            default_depth=fifo_depth,
            dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)],
        )
        of_offsets = [m * n * i for i in range(n_aie_rows)]

        # join along one column
        c_tmp_fifos = (
            C_l2l3_fifos[col]
            .prod()
            .join(
                of_offsets,
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[fifo_depth] * n_aie_rows,
                placement=Tile(col, 1),
            )
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = c_tmp_fifos[j]

    # Tasks for each worker to perform
    def core_fn(in_a, in_b, out_c, zero, matmul):
        loop = range(1)  # Workaround for issue #1547
        if n_tiles_per_core > 1:
            loop = range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)

            for _ in range_(K // k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    # Set up compute tiles
    workers = []
    for row in range(n_aie_rows):
        for col in range(n_aie_cols):
            tile_col, tile_row = core_tiles[row][col]
            workers.append(
                Worker(
                    core_fn,
                    [
                        A_l2l1_fifos[row].cons(),
                        B_l2l1_fifos[col].cons(),
                        C_l1l2_fifos[row][col].prod(),
                        zero_kernel,
                        matmul_kernel,
                    ],
                    placement=Tile(tile_col, tile_row),
                )
            )

    # We are limited in the number of BDs. After synchronizing, we can reuse BDs.
    # We only transfer 6 rows of tiles at once before starting a new transfer block.
    # tb = transfer block; block of transfers before sync call
    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    # Define tensor access patterns (tiling) for A, B, and C
    A_tiles = TensorTiler2D.group_tiler(
        (M, K),  # Size of A matrix
        (m * n_A_tiles_per_shim, k),  # Size of A (smallest) tile
        (1, K // k),  # Size of "group" of tiles
        # Repeat data so can distribute across whole column
        pattern_repeat=N // n // n_aie_cols,
    )
    if b_col_maj:
        B_tiles = TensorTiler2D.step_tiler(
            (K, N),  # Size of B matrix
            (k, n),  # Size of B tile
            # Number of tiles per transfer in each dimension (whole col, partial row)
            tile_group_repeats=(K // k // n_aie_cols, N // n),
            # Contiguous tile group in col, but send every n_aie_cols-th tile in the row
            tile_group_steps=(n_aie_cols, 1),
        )
    else:
        B_tiles = TensorTiler2D.step_tiler(
            (K, N),  # Size of B matrix
            (k, n),  # Size of B tile
            # Number of tiles per transfer in each dimension (whole col, partial row)
            tile_group_repeats=(K // k, N // n // n_aie_cols),
            # Contiguous tile group in col, but send every n_aie_cols-th tile in the row
            tile_group_steps=(1, n_aie_cols),
            tile_group_col_major=True,  # Send all tiles in column before moving on to next column
        )
    C_tiles = TensorTiler2D.step_tiler(
        (M, N),  # Size of C matrix
        (m * n_aie_rows, n),  # Size of C tile
        # Number of tiles per transfer in each dimension (partial col, partial row)
        tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
        # Collect every n_aie_cols row at a time (mirroring how we sent in B data)
        tile_group_steps=(1, n_aie_cols),
    )
    c_index = 0

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(*workers)

        # Task groups will be used to determine when to sync/await/free DMA runtime ops
        tg = rt.task_group()
        for tb in range(ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    # May not have pong iteration in some cases
                    break

                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                )

                for col in range(n_aie_cols):

                    # This line does not change MLIR output at all - it's just for recording data movement
                    C_taps.append(C_tiles[c_index])

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
                    rt.drain(
                        C_l2l3_fifos[col].cons(),
                        C,
                        tap=C_tiles[c_index],
                        wait=True,
                        task_group=tg,
                        placement=Tile(col, 0),
                    )
                    c_index += 1

                    for tile_row in range(current_tb_n_rows):

                        # A input transfer:
                        #
                        # The smallest transfer unit is a (m*n_A_tiles_per_shim)-sized sub-tile of the input matrix.
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
                        tile_offset = ((row_base + tile_row) * n_aie_cols + col) % len(
                            A_tiles
                        )

                        rt.fill(
                            A_l3l2_fifos[col].prod(),
                            A,
                            tap=A_tiles[tile_offset],
                            task_group=tg,
                            placement=Tile(col, 0),
                        )
                        # Use the calculated sizes/strides/offsets to record the data movement
                        # caused by the above call to npu_dma_memcpy_nd.
                        # This line does not change MLIR output at all.

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
                        rt.fill(
                            B_l3l2_fifos[col].prod(),
                            B,
                            tap=B_tiles[col],
                            task_group=tg,
                            placement=Tile(col, 0),
                        )

                        # These lines do not change MLIR output at all - they are just for recording data movement
                        A_taps.append(A_tiles[tile_offset])
                        B_taps.append(B_tiles[col])
                if tb > 0 or (tb == 0 and pingpong > 0):
                    rt.finish_task_group(tg)
                    tg = rt.task_group()
        rt.finish_task_group(tg)

    if generate_taps:
        # If generate taps is true, return a representation of tensor access patterns
        # representing all the npu_dma_memcpy_nd runtime sequence operations per input/ouput tensor.
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )

    # Create the program from the device type and runtime
    my_program = Program(dev, rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())
    return module


if __name__ == "__main__":
    main()
