#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.
"""Matrix-vector multiply — IRON API design with ``@iron.jit`` compilation.

A single AIE compute core computes ``c = A @ b`` (M-row matrix x K-vector).
Default config: ``M=K=288``, kernel tile ``m=k=32``, vectorized mv kernel.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def matrix_vector(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    vectorized: CompileTime[bool] = True,
    use_chess: CompileTime[bool] = False,
):
    n_cores = 1
    M_div_n_cores = M // n_cores
    M_div_m_div_n_cores = M // (m * n_cores)
    K_div_k = K // k

    matvec_kernel = kernels.mv(
        dim_m=m, dim_k=k, vectorized=vectorized, use_chess=use_chess
    )
    zero_kernel = matvec_kernel.zero

    dtype_in = np.dtype[np.int16]
    dtype_out = np.dtype[np.int32]
    A_ty = np.ndarray[(M, K), dtype_in]
    B_ty = np.ndarray[(1, K), dtype_in]
    C_ty = np.ndarray[(1, M), dtype_out]
    inA_ty = np.ndarray[(m, k), dtype_in]
    inB_ty = np.ndarray[(k,), dtype_in]
    outC_ty = np.ndarray[(m,), dtype_out]

    # The vectorized mv kernel reads A in a "32-bit-word transposed" layout
    # (see aie_kernels/aie2/mv.cc): for 2-byte elements the transpose
    # granularity is 2 elements, packing rows of each 2-column word slowly,
    # m rows then the next 2-col word.
    a_dims_from_stream = [(m, 2), (k // 2, 2 * m), (2, 1)] if vectorized else None

    def core_fn(of_a, of_b, of_c, zero, matvec):
        elem_out = of_c.acquire(1)
        zero(elem_out)
        for _ in range_(K_div_k):
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            matvec(elem_in_a, elem_in_b, elem_out)
            of_a.release(1)
            of_b.release(1)
        of_c.release(1)

    memA_fifos = []
    coreA_fifos = []
    outC_fifos = []
    workers = []
    B_fifo = ObjectFifo(inB_ty)
    for i in range(n_cores):
        a_fifo = ObjectFifo(inA_ty, name=f"memA{i}")
        memA_fifos.append(a_fifo)
        coreA_fifos.append(a_fifo.cons().forward(dims_from_stream=a_dims_from_stream))
        outC_fifos.append(ObjectFifo(outC_ty, name=f"outC{i}"))
        workers.append(
            Worker(
                core_fn,
                [
                    coreA_fifos[i].cons(),
                    B_fifo.cons(),
                    outC_fifos[i].prod(),
                    zero_kernel,
                    matvec_kernel,
                ],
            )
        )

    A_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (M_div_m_div_n_cores, K_div_k), prune_step=False
    )
    C_taps = TensorTiler2D.simple_tiler((1, M), (1, M_div_n_cores), prune_step=False)
    b_tap = TensorTiler2D.simple_tiler(
        (1, K), pattern_repeat=M_div_m_div_n_cores, prune_step=False
    )[0]

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (a_in, b_in, c_out):
        rt.start(*workers)
        rt.fill(B_fifo.prod(), b_in, b_tap)
        for i, (a_tap, c_tap) in enumerate(zip(A_taps, C_taps)):
            rt.fill(memA_fifos[i].prod(), a_in, a_tap)
            rt.drain(outC_fifos[i].cons(), c_out, c_tap, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Matrix-Vector Multiplication")
    add_compile_args(p, short_dev=None)
    p.add_argument("-M", type=int, default=288)
    p.add_argument("-K", type=int, default=288)
    p.add_argument("-N", type=int, default=1)  # accepted but unused (mv has N=1)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("-k", type=int, default=32)
    p.add_argument("--dtype_in", type=str, default="i16")
    p.add_argument("--dtype_out", type=str, default="i32")
    p.add_argument("--scalar", action="store_true", help="use scalar mv kernel")
    p.add_argument("--use-chess", type=int, choices=[0, 1], default=0)
    p.add_argument(
        "--emulate-bf16-mmul-with-bfp16", type=int, choices=[0, 1], default=0
    )
    add_benchmark_args(p)
    return p


def _run_and_verify(opts):
    rng = np.random.default_rng(1726250518)
    A_np = rng.integers(-1000, 1000, size=(opts.M, opts.K), dtype=np.int16)
    B_np = rng.integers(-1000, 1000, size=(opts.K,), dtype=np.int16)
    A_t = iron.tensor(A_np.reshape(-1), dtype=np.int16, device="npu")
    B_t = iron.tensor(B_np.reshape(-1), dtype=np.int16, device="npu")
    C_t = iron.zeros(opts.M, dtype=np.int32, device="npu")

    bench = run_iters(
        matrix_vector,
        A_t,
        B_t,
        C_t,
        M=opts.M,
        K=opts.K,
        m=opts.m,
        k=opts.k,
        vectorized=not opts.scalar,
        use_chess=bool(opts.use_chess),
        warmup=opts.warmup,
        iters=opts.iters,
    )

    expected = (A_np.astype(np.int64) @ B_np.astype(np.int64)).astype(np.int32)
    actual = C_t.numpy().reshape(opts.M)
    assert_close_with_benchmark(
        actual,
        expected,
        bench=bench,
        ops=2.0 * opts.M * opts.K,
        gflops_fmt=".4f",
        fail_msg="output does not match A @ b",
    )


def _compile_kwargs(opts):
    return dict(
        M=opts.M,
        K=opts.K,
        m=opts.m,
        k=opts.k,
        vectorized=not opts.scalar,
        use_chess=bool(opts.use_chess),
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        matrix_vector,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
    )


if __name__ == "__main__":
    main()
