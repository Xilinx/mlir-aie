# single_core.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Scalar in-core shuffle for bfp16ebs8 tiles — ``@iron.jit`` IRON design.

One AIE2P core runs the per-tile scalarShuffle helper on an A-tile
(useful as a building block for matmul preparation). No B/C matmul
path; the host harness ingests A and reads back the shuffled C.
"""

import argparse

import aie.iron as iron
import aie.iron.kernels as kernels
import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.utils.hostruntime.argparse import (
    add_compile_args,
    device_from_args,
)
from aie.utils.hostruntime.cli import run_design_cli


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def in_core_shuffle(
    A: In,
    C: Out,
    *,
    M: CompileTime[int] = 64,
    K: CompileTime[int] = 64,
    N: CompileTime[int] = 64,
    m: CompileTime[int] = 64,
    k: CompileTime[int] = 64,
    n: CompileTime[int] = 64,
):
    a_ty = np.ndarray[(m * k // 8,), np.dtype[v8bfp16ebs8]]
    c_ty = np.ndarray[(m * n // 8,), np.dtype[v8bfp16ebs8]]

    scalar_shuffle_kernel = kernels.mm_bfp_shuffle(dim_m=m, dim_k=k, dim_n=n)

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

    def sequence(a, c, inA_h, outC_h):
        inA_h.fill(a)
        outC_h.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [A_ty, C_ty, inA.prod(), outC.cons()],
    )

    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


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
