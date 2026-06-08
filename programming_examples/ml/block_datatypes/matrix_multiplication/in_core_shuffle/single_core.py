# single_core.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Scalar in-core shuffle for bfp16ebs8 tiles — ``@iron.jit`` IRON design.

One AIE2P core runs the per-tile scalarShuffle helper on an A-tile
(useful as a building block for matmul preparation). No B/C matmul
path; the host harness ingests A and reads back the shuffled C.
"""

import argparse
from pathlib import Path

import numpy as np

from aie.dialects.aiex import v8bfp16ebs8

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
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli

_KERNEL_SRC = (
    Path(__file__).resolve().parents[5] / "aie_kernels" / "aie2p" / "mm_bfp.cc"
)


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def in_core_shuffle(
    A: In,
    C: Out,
    *,
    M: Compile[int] = 64,
    K: Compile[int] = 64,
    N: Compile[int] = 64,
    m: Compile[int] = 64,
    k: Compile[int] = 64,
    n: Compile[int] = 64,
):
    a_ty = np.ndarray[(m * k // 8,), np.dtype[v8bfp16ebs8]]
    c_ty = np.ndarray[(m * n // 8,), np.dtype[v8bfp16ebs8]]

    kernel_flags = [f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}"]

    scalar_shuffle_kernel = ExternalFunction(
        "scalar_shuffle",
        source_file=str(_KERNEL_SRC),
        arg_types=[a_ty, c_ty, np.int16, np.int16, np.int16],
        compile_flags=kernel_flags + ["-DSHUFFLE_ONLY"],
        use_chess=True,
    )

    inA = ObjectFifo(a_ty, name="inA")
    memA = inA.cons().forward(name="memA")
    memC = ObjectFifo(c_ty, name="memC")
    outC = memC.cons().forward(name="outC")

    def core_fn(of_a, of_c, scalar_shuffle):
        elem_out = of_c.acquire(1)
        elem_in_a = of_a.acquire(1)
        scalar_shuffle(elem_in_a, elem_out, k, m, False)
        of_a.release(1)
        of_c.release(1)

    worker = Worker(
        core_fn,
        [memA.cons(), memC.prod(), scalar_shuffle_kernel],
        stack_size=0xF00,
    )

    A_ty = np.ndarray[(M * K // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(M * N // 8,), np.dtype[v8bfp16ebs8]]

    rt = Runtime()
    with rt.sequence(A_ty, C_ty) as (a, c):
        rt.start(worker)
        rt.fill(inA.prod(), a)
        rt.drain(outC.cons(), c, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(
        prog="AIE In-Core Scalar Shuffle (bfp16ebs8)",
    )
    add_compile_args(p, default_dev="npu2")
    p.add_argument("-M", type=int, default=64)
    p.add_argument("-K", type=int, default=64)
    p.add_argument("-N", type=int, default=64)
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=64)
    return p


def _compile_kwargs(opts):
    return dict(M=opts.M, K=opts.K, N=opts.N, m=opts.m, k=opts.k, n=opts.n)


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        in_core_shuffle,
        opts,
        compile_kwargs=_compile_kwargs,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
