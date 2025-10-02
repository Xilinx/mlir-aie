#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.localbuffer import LocalBuffer
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D
from aie.dialects.aiex import v8bfp16ebs8


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Whole Array) using bfp16 and in core shuffle through the scalar unit",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-N", type=int, default=512)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=64)
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    args = argparser.parse_args()
    print(my_matmul(args.M, args.K, args.N, args.m, args.k, args.n, args.n_aie_cols))


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(M, K, N, m, k, n, n_aie_cols):
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    r, s, t = 8, 8, 8

    assert (
        M % (m * n_aie_rows) == 0
    ), """A must be tileable into (m * n_aie_rows, k)-sized blocks"""

    assert K % k == 0

    assert (
        N % (n * n_aie_cols) == 0
    ), """B must be tileable into (k, n * n_aie_cols)-sized blocks"""

    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    assert (
        m == 64 and k == 64 and n == 64
    ), """Only 64x64x64 tiles are supported in this example"""

    fifo_depth = 2

    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    if n_aie_cols > n_aie_rows:
        n_shim_mem_A = n_aie_rows
    else:
        n_shim_mem_A = n_aie_cols

    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    dev_ty = NPU2()

    # Define tensor types
    A_ty = np.ndarray[(M * K // 8,), np.dtype[v8bfp16ebs8]]
    B_ty = np.ndarray[(K * N // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(M * N // 8,), np.dtype[v8bfp16ebs8]]
    A_l2_ty = np.ndarray[(m * k // 8 * n_A_tiles_per_shim,), np.dtype[v8bfp16ebs8]]
    B_l2_ty = np.ndarray[(k * n // 8,), np.dtype[v8bfp16ebs8]]
    C_l2_ty = np.ndarray[(m * n // 8 * n_aie_rows,), np.dtype[v8bfp16ebs8]]
    A_l1_ty = np.ndarray[(m, k // 8), np.dtype[v8bfp16ebs8]]
    B_l1_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l1_ty = np.ndarray[(m, n // 8), np.dtype[v8bfp16ebs8]]

    zero_kernel = Kernel(f"zero_kernel", f"mm_{m}x{k}x{n}.o", [C_l1_ty])
    matmul_kernel = Kernel(
        "matmul_vectorized_bfp16",
        f"mm_{m}x{k}x{n}.o",
        [A_l1_ty, B_l1_ty, C_l1_ty],
    )
    shuffle_kernel = Kernel(
        "scalar_shuffle",
        f"mm_{m}x{k}x{n}.o",
        # Assumes all matrices are identical!
        [A_l1_ty, C_l1_ty, np.int16, np.int16, np.int16],
    )

    tiles = [[(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows

    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols

    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    for i in range(n_shim_mem_A):
        A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k // 8 * j for j in range(stop_row - start_row)]
        a_tmp_fifos = (
            A_l3l2_fifos[i]
            .cons()
            .split(
                of_offsets,
                obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
                placement=Tile(2 * i if n_aie_cols == 8 else i, 1),
            )
        )

        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = a_tmp_fifos[j]

    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        dims_to_stream = None
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

        C_l2l3_fifos[col] = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth)
        of_offsets = [m * n // 8 * i for i in range(n_aie_rows)]

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

    def core_fn(in_a, in_b, out_c, zero, matmul, shuffle):
        bufferA = LocalBuffer(A_l1_ty)
        loop = range(1)
        if n_tiles_per_core > 1:
            loop = range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)

            for _ in range_(K // k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                shuffle(elem_in_a, bufferA, k, m, False)
                matmul(bufferA, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

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
                        shuffle_kernel,
                    ],
                    placement=Tile(tile_col, tile_row),
                    stack_size=0xD00,
                )
            )

    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K // 8),
        (m * n_A_tiles_per_shim, k // 8),
        (1, K // k),
        pattern_repeat=N // n // n_aie_cols,
    )
    B_tiles = TensorTiler2D.step_tiler(
        (K, N // 8),
        (k, n // 8),
        tile_group_repeats=(K // k // n_aie_cols, N // n),
        tile_group_steps=(n_aie_cols, 1),
    )
    C_tiles = TensorTiler2D.step_tiler(
        (M, N // 8),
        (m * n_aie_rows, n // 8),
        tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
    )
    c_index = 0

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(*workers)

        tg = rt.task_group()
        for tb in range(ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break

                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                )

                for col in range(n_aie_cols):
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
                        tile_offset = (
                            (row_base + tile_row) * n_shim_mem_A + col
                        ) % len(A_tiles)

                        if col < n_aie_rows:
                            rt.fill(
                                A_l3l2_fifos[col].prod(),
                                A,
                                tap=A_tiles[tile_offset],
                                task_group=tg,
                                placement=Tile(2 * col if n_aie_cols == 8 else col, 0),
                            )

                        rt.fill(
                            B_l3l2_fifos[col].prod(),
                            B,
                            tap=B_tiles[col],
                            task_group=tg,
                            placement=Tile(col, 0),
                        )

                if tb > 0 or (tb == 0 and pingpong > 0):
                    rt.finish_task_group(tg)
                    tg = rt.task_group()
        rt.finish_task_group(tg)

    my_program = Program(dev_ty, rt)

    module = my_program.resolve_program(SequentialPlacer())
    return module


if __name__ == "__main__":
    main()
