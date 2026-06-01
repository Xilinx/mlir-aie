# whole_array.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Whole-array bfp16ebs8 matmul — ``@iron.jit`` IRON design.

n_aie_rows x n_aie_cols compute cores tile a (M,K,N) GEMM with per-core
(m,k,n). Strix-only; kernel is chess-built.
"""

import argparse
from pathlib import Path

import numpy as np

from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorTiler2D

import aie.iron as iron
from aie.iron import (
    Compile,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import device_from_args
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

_KERNEL_SRC = (
    Path(__file__).resolve().parents[5] / "aie_kernels" / "aie2p" / "mm_bfp.cc"
)


def ceildiv(a, b):
    return (a + b - 1) // b


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def whole_array_matmul(
    A: In,
    B: In,
    C: Out,
    *,
    M: Compile[int] = 512,
    K: Compile[int] = 512,
    N: Compile[int] = 512,
    m: Compile[int] = 64,
    k: Compile[int] = 64,
    n: Compile[int] = 64,
    n_aie_cols: Compile[int] = 4,
):
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols
    fifo_depth = 2

    assert M % (m * n_aie_rows) == 0, "M must be tileable into (m*n_aie_rows, k) blocks"
    assert K % k == 0
    assert N % (n * n_aie_cols) == 0, "N must be tileable into (k, n*n_aie_cols) blocks"

    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    n_shim_mem_A = n_aie_rows if n_aie_cols > n_aie_rows else n_aie_cols
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    A_l2_ty = np.ndarray[(m * k // 8 * n_A_tiles_per_shim,), np.dtype[v8bfp16ebs8]]
    B_l2_ty = np.ndarray[(k * n // 8,), np.dtype[v8bfp16ebs8]]
    C_l2_ty = np.ndarray[(m * n // 8 * n_aie_rows,), np.dtype[v8bfp16ebs8]]
    A_l1_ty = np.ndarray[(m, k // 8), np.dtype[v8bfp16ebs8]]
    B_l1_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l1_ty = np.ndarray[(m, n // 8), np.dtype[v8bfp16ebs8]]

    kernel_flags = [f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}"]

    zero_kernel = ExternalFunction(
        "zero_kernel",
        source_file=str(_KERNEL_SRC),
        arg_types=[C_l1_ty],
        compile_flags=kernel_flags + ["-DZERO_ONLY"],
        use_chess=True,
    )
    matmul_kernel = ExternalFunction(
        "matmul_vectorized_bfp16",
        source_file=str(_KERNEL_SRC),
        arg_types=[A_l1_ty, B_l1_ty, C_l1_ty],
        compile_flags=kernel_flags + ["-DMATMUL_ONLY"],
        use_chess=True,
    )

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
            )
        )
        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = a_tmp_fifos[j]

    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l2l1_fifos[col] = (
            B_l3l2_fifos[col].cons().forward(obj_type=B_l1_ty, name=f"B_L2L1_{col}")
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
            )
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = c_tmp_fifos[j]

    def core_fn(in_a, in_b, out_c, zero, matmul):
        loop = range_(n_tiles_per_core) if n_tiles_per_core > 1 else range(1)
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

    workers = Worker.grid(
        n_aie_rows,
        n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
            ],
            stack_size=0xD00,
        ),
    )

    A_ty = np.ndarray[(M * K // 8,), np.dtype[v8bfp16ebs8]]
    B_ty = np.ndarray[(K * N // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(M * N // 8,), np.dtype[v8bfp16ebs8]]

    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K // 8),
        (m * n_A_tiles_per_shim, k // 8),
        (1, K // k),
        pattern_repeat=N // n // n_aie_cols,
    )
    B_tiles = TensorTiler2D.step_tiler(
        (N, K // 8),
        (n, k // 8),
        tile_group_repeats=(N // n // n_aie_cols, K // k),
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
    with rt.sequence(A_ty, B_ty, C_ty) as (a, b, c):
        rt.start(*[w for row in workers for w in row])
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
                        c,
                        tap=C_tiles[c_index],
                        wait=True,
                        task_group=tg,
                    )
                    c_index += 1
                    for tile_row in range(current_tb_n_rows):
                        tile_offset = (
                            (row_base + tile_row) * n_shim_mem_A + col
                        ) % len(A_tiles)
                        if col < n_aie_rows:
                            rt.fill(
                                A_l3l2_fifos[col].prod(),
                                a,
                                tap=A_tiles[tile_offset],
                                task_group=tg,
                            )
                        rt.fill(
                            B_l3l2_fifos[col].prod(),
                            b,
                            tap=B_tiles[col],
                            task_group=tg,
                        )
                if tb > 0 or (tb == 0 and pingpong > 0):
                    rt.finish_task_group(tg)
                    tg = rt.task_group()
        rt.finish_task_group(tg)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(
        prog="AIE Whole-Array bfp16ebs8 Matmul",
    )
    add_compile_args(p, default_dev="npu2")
    p.add_argument("-M", type=int, default=512)
    p.add_argument("-K", type=int, default=512)
    p.add_argument("-N", type=int, default=512)
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=64)
    p.add_argument(
        "--n-aie-cols", dest="n_aie_cols", type=int, choices=[1, 2, 4, 8], default=4
    )
    return p


def _compile_kwargs(opts):
    return dict(
        M=opts.M,
        K=opts.K,
        N=opts.N,
        m=opts.m,
        k=opts.k,
        n=opts.n,
        n_aie_cols=opts.n_aie_cols,
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        whole_array_matmul,
        opts,
        compile_kwargs=_compile_kwargs,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
