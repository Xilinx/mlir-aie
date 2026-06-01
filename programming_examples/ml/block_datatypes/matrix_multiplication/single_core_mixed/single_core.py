# single_core.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Single-core mixed bf16/bfp16 matmul — ``@iron.jit`` IRON design.

One AIE2P core does a (bf16, bfp16) -> bf16 GEMM with on-shim shuffle
into the bf16 mac layout via memtile dims_to_stream. Strix-only;
kernel is chess-built.
"""

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib.tensortiler2d import TensorTiler2D

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
    Path(__file__).resolve().parents[5] / "aie_kernels" / "aie2p" / "mm_bfp_mixed.cc"
)


def ceildiv(a, b):
    return (a + b - 1) // b


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def single_core_mixed(
    A: In,
    B: In,
    C: Out,
    *,
    M: Compile[int] = 128,
    K: Compile[int] = 128,
    N: Compile[int] = 128,
    m: Compile[int] = 64,
    k: Compile[int] = 64,
    n: Compile[int] = 64,
):
    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    r, s, t = 8, 8, 8

    a_ty = np.ndarray[(m * k,), np.dtype[bfloat16]]
    b_ty = np.ndarray[(k * n // 8,), np.dtype[v8bfp16ebs8]]
    c_ty = np.ndarray[(m * n,), np.dtype[bfloat16]]

    kernel_flags = [f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}"]

    zero_kernel = ExternalFunction(
        "zero_kernel_bf16",
        source_file=str(_KERNEL_SRC),
        arg_types=[c_ty],
        compile_flags=kernel_flags + ["-DZERO_ONLY"],
        use_chess=True,
    )
    matmul_kernel = ExternalFunction(
        "matmul_vectorized_different_datatypes",
        source_file=str(_KERNEL_SRC),
        arg_types=[a_ty, b_ty, c_ty],
        compile_flags=kernel_flags + ["-DMATMUL_ONLY"],
        use_chess=True,
    )

    inA = ObjectFifo(a_ty, name="inA")
    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)
    inB = ObjectFifo(b_ty, name="inB")
    memB = inB.cons().forward(name="memB")
    memC = ObjectFifo(c_ty, name="memC")
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    def core_fn(of_a, of_b, of_c, zero, matmul):
        for _ in range_(tiles) if tiles > 1 else range(1):
            elem_out = of_c.acquire(1)
            zero(elem_out)
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [memA.cons(), memB.cons(), memC.prod(), zero_kernel, matmul_kernel],
        stack_size=0xF00,
    )

    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(K * N // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]

    rows_per_block = 4

    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n
    )
    b_tap = TensorTiler2D.group_tiler((N, K // 8), (n, k // 8), (N_div_n, K_div_k))[0]

    C_tiles = TensorTiler2D.group_tiler((M, N), (m, n), (rows_per_block // 2, N_div_n))
    c_index = 0

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (a, b, c):
        rt.start(worker)
        tgs = []
        for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
            for pingpong in [0, 1]:
                row_base = (
                    tile_row_block * rows_per_block + pingpong * rows_per_block // 2
                )
                num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                if num_tile_rows <= 0:
                    break
                tgs.append(rt.task_group())
                for tile_row in range(num_tile_rows):
                    tile_offset = (row_base + tile_row) % len(A_tiles)
                    rt.fill(inA.prod(), a, tap=A_tiles[tile_offset], task_group=tgs[-1])
                    rt.fill(inB.prod(), b, tap=b_tap, task_group=tgs[-1])
                rt.drain(
                    outC.cons(), c, tap=C_tiles[c_index], task_group=tgs[-1], wait=True
                )
                c_index += 1
                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]
        rt.finish_task_group(tgs[-1])
        del tgs[-1]

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(
        prog="AIE Single-Core Mixed bf16/bfp16 Matmul",
    )
    add_compile_args(p, default_dev="npu2")
    p.add_argument("-M", type=int, default=128)
    p.add_argument("-K", type=int, default=128)
    p.add_argument("-N", type=int, default=128)
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=64)
    return p


def _compile_kwargs(opts):
    return dict(M=opts.M, K=opts.K, N=opts.N, m=opts.m, k=opts.k, n=opts.n)


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        single_core_mixed,
        opts,
        compile_kwargs=_compile_kwargs,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
