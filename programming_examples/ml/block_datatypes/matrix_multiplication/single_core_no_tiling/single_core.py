# single_core.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Single-core bfp16ebs8 matmul, NO tiling — ``@iron.jit`` IRON design.

One AIE2P core processes a single 64x128x64 GEMM tile in one shot, no
host-side tile loop or memtile fanout. Strix-only; kernel is chess-built.
"""

import argparse
from pathlib import Path

import numpy as np

from aie.dialects.aiex import v8bfp16ebs8

import aie.iron as iron
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli

_KERNEL_SRC = (
    Path(__file__).resolve().parents[5] / "aie_kernels" / "aie2p" / "mm_bfp.cc"
)


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def single_core_no_tiling(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int] = 64,
    K: CompileTime[int] = 128,
    N: CompileTime[int] = 64,
    m: CompileTime[int] = 64,
    k: CompileTime[int] = 128,
    n: CompileTime[int] = 64,
):
    # bfp16ebs8 matmul mac unit is 8x8x8; m/k/n must be multiples of these.
    r = s = t = 8
    assert m % r == 0, f"m ({m}) must be a multiple of {r}"
    assert k % s == 0, f"k ({k}) must be a multiple of {s}"
    assert n % t == 0, f"n ({n}) must be a multiple of {t}"

    a_ty = np.ndarray[(m * k // 8,), np.dtype[v8bfp16ebs8]]
    b_ty = np.ndarray[(k * n // 8,), np.dtype[v8bfp16ebs8]]
    c_ty = np.ndarray[(m * n // 8,), np.dtype[v8bfp16ebs8]]

    kernel_flags = [f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}"]

    zero_kernel = ExternalFunction(
        "zero_kernel",
        source_file=str(_KERNEL_SRC),
        arg_types=[c_ty],
        compile_flags=kernel_flags + ["-DZERO_ONLY"],
        use_chess=True,
    )
    matmul_kernel = ExternalFunction(
        "matmul_vectorized_bfp16",
        source_file=str(_KERNEL_SRC),
        arg_types=[a_ty, b_ty, c_ty],
        compile_flags=kernel_flags + ["-DMATMUL_ONLY"],
        use_chess=True,
    )

    inA = ObjectFifo(a_ty, name="inA")
    memA = inA.cons().forward(name="memA")
    inB = ObjectFifo(b_ty, name="inB")
    memB = inB.cons().forward(name="memB")
    memC = ObjectFifo(c_ty, name="memC")
    outC = memC.cons().forward(name="outC")

    def core_fn(of_a, of_b, of_c, zero, matmul):
        elem_out = of_c.acquire(1)
        zero(elem_out)
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

    A_ty = np.ndarray[(M * K // 8,), np.dtype[v8bfp16ebs8]]
    B_ty = np.ndarray[(K * N // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(M * N // 8,), np.dtype[v8bfp16ebs8]]

    rt = Runtime()

    def sequence(a, b, c):
        inA.prod().fill(a)
        inB.prod().fill(b)
        outC.cons().drain(c, wait=True)

    rt.sequence(sequence, [A_ty, B_ty, C_ty])

    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(
        prog="AIE Single-Core bfp16ebs8 Matmul (No Tiling)",
    )
    add_compile_args(p, default_dev="npu2")
    p.add_argument("-M", type=int, default=64)
    p.add_argument("-K", type=int, default=128)
    p.add_argument("-N", type=int, default=64)
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=128)
    p.add_argument("-n", type=int, default=64)
    return p


def _compile_kwargs(opts):
    return dict(M=opts.M, K=opts.K, N=opts.N, m=opts.m, k=opts.k, n=opts.n)


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        single_core_no_tiling,
        opts,
        compile_kwargs=_compile_kwargs,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
