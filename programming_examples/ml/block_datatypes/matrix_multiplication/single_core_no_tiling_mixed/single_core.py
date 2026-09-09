# single_core.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Single-core mixed bf16/bfp16 matmul, NO tiling — ``@iron.jit`` IRON design.

One AIE2P core, one 64x64x64 mixed (bf16, bfp16) -> bf16 mac, no host
tile loop. Strix-only.
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
    StreamDims,
    Worker,
)
from aie.utils.hostruntime.argparse import (
    add_compile_args,
    device_from_args,
)
from aie.utils.hostruntime.cli import run_design_cli
from ml_dtypes import bfloat16


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def single_core_no_tiling_mixed(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int] = 64,
    K: CompileTime[int] = 64,
    N: CompileTime[int] = 64,
    m: CompileTime[int] = 64,
    k: CompileTime[int] = 64,
    n: CompileTime[int] = 64,
):
    a_ty = np.ndarray[(m * k,), np.dtype[bfloat16]]
    b_ty = np.ndarray[(k * n // 8,), np.dtype[v8bfp16ebs8]]
    c_ty = np.ndarray[(m * n,), np.dtype[bfloat16]]

    matmul_kernel = kernels.mm_bfp(dim_m=m, dim_k=k, dim_n=n, mixed=True)
    zero_kernel = matmul_kernel.zero

    inA = ObjectFifo(a_ty, name="inA")
    a_dims: StreamDims = matmul_kernel.stream_dims["A"]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)
    inB = ObjectFifo(b_ty, name="inB")
    memB = inB.cons().forward(name="memB")
    memC = ObjectFifo(c_ty, name="memC")
    c_dims: StreamDims = matmul_kernel.stream_dims["C"]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

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

    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(K * N // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]

    def sequence(a, b, c, inA_h, inB_h, outC_h):
        inA_h.fill(a)
        inB_h.fill(b)
        outC_h.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [A_ty, B_ty, C_ty, inA.prod(), inB.prod(), outC.cons()],
    )

    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(
        prog="AIE Single-Core Mixed bf16/bfp16 Matmul (No Tiling)",
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
        single_core_no_tiling_mixed,
        opts,
        compile_kwargs=_compile_kwargs,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
