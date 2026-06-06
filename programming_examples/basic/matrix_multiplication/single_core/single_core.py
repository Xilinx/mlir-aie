#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.
"""Single-core matrix multiply — IRON API design with ``@iron.jit`` compilation.

A single AIE compute core computes ``C = A @ B``, optionally with B laid out
column-major.  The host streams (m, k) x (k, n) tile pairs through an
ObjectFifo pipeline; the core multiply-accumulates into an (m, n) output tile.

The script has two modes (matching whole_array.py):

* ``--xclbin-path=... --insts-path=...`` — compile-only, used by the Makefile
  so ``test.cpp`` + ``sweep.sh`` drive the design.
* default — JIT-compile + run on the attached NPU + verify against numpy.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import (
    Compile,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import (
    add_benchmark_args,
    add_compile_args,
    add_trace_arg,
)
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.trace import TraceConfig
from aie.utils.verify import assert_close_with_benchmark


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def single_core(
    A: In,
    B: In,
    C: Out,
    *,
    M: Compile[int],
    K: Compile[int],
    N: Compile[int],
    m: Compile[int],
    k: Compile[int],
    n: Compile[int],
    dtype_in_str: Compile[str],
    dtype_out_str: Compile[str],
    b_col_maj: Compile[int] = 0,
    emulate_bf16_mmul_with_bfp16: Compile[bool] = False,
    use_chess: Compile[bool] = False,
    trace_config: Compile[TraceConfig | None] = None,
):
    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    assert M % m == 0
    assert K % k == 0
    assert N % n == 0
    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), "input and output dtypes must both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), "output dtype must be equal or larger to input dtype"

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        b_col_maj=bool(b_col_maj),
        use_chess=use_chess,
        emulate_bf16_mmul_with_bfp16=emulate_bf16_mmul_with_bfp16,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    b_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    inA = ObjectFifo(a_ty, name="inA")
    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    inB = ObjectFifo(b_ty, name="inB")
    if b_col_maj:
        b_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
    else:
        b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

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
        stack_size=0xD00,
        trace=1 if trace_config else 0,
    )

    rows_per_block = 4

    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n, prune_step=False
    )
    if b_col_maj:
        b_tap = TensorTiler2D.group_tiler(
            (N, K), (n, k), (N_div_n, K_div_k), prune_step=False
        )[0]
    else:
        b_tap = TensorTiler2D.group_tiler(
            (K, N),
            (k, n),
            (K_div_k, N_div_n),
            tile_group_col_major=True,
            prune_step=False,
        )[0]
    C_tiles = TensorTiler2D.group_tiler(
        (M, N), (m, n), (rows_per_block // 2, N_div_n), prune_step=False
    )
    c_index = 0

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        if trace_config:
            rt.enable_trace(trace_config.trace_size, workers=[worker])
        rt.start(worker)

        tgs = []
        for tile_row_block in range(iron.ceildiv(M_div_m, rows_per_block)):
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
                    rt.fill(inA.prod(), A, tap=A_tiles[tile_offset], task_group=tgs[-1])
                    rt.fill(inB.prod(), B, tap=b_tap, task_group=tgs[-1])
                rt.drain(
                    outC.cons(), C, tap=C_tiles[c_index], task_group=tgs[-1], wait=True
                )
                c_index += 1
                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]
        rt.finish_task_group(tgs[-1])
        del tgs[-1]

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Matrix Multiplication (Single Core)")
    add_compile_args(p, short_dev=None)
    p.add_argument("-M", type=int, default=512)
    p.add_argument("-K", type=int, default=512)
    p.add_argument("-N", type=int, default=512)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("-k", type=int, default=32)
    p.add_argument("-n", type=int, default=32)
    p.add_argument("--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16")
    p.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i32",
    )
    p.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument(
        "--emulate-bf16-mmul-with-bfp16", type=int, choices=[0, 1], default=0
    )
    p.add_argument("--use-chess", type=int, choices=[0, 1], default=0)
    add_trace_arg(p, with_short=False)
    add_benchmark_args(p)
    return p


def _numpy_reference(A_np, B_np, b_col_maj, dtype_out):
    B_logical = B_np.T if b_col_maj else B_np
    if np.issubdtype(A_np.dtype, np.integer):
        return (A_np.astype(np.int64) @ B_logical.astype(np.int64)).astype(dtype_out)
    return (A_np.astype(np.float32) @ B_logical.astype(np.float32)).astype(dtype_out)


def _trace_config(opts):
    return TraceConfig(trace_size=opts.trace_size) if opts.trace_size > 0 else None


def _run_and_verify(opts):
    dtype_in = str_to_dtype(opts.dtype_in)
    dtype_out = str_to_dtype(opts.dtype_out)

    rng = np.random.default_rng(1726250518)
    if np.issubdtype(dtype_in, np.integer):
        info = np.iinfo(dtype_in)
        A_np = rng.integers(
            info.min // 4, info.max // 4, size=(opts.M, opts.K), dtype=dtype_in
        )
        B_shape = (opts.N, opts.K) if opts.b_col_maj else (opts.K, opts.N)
        B_np = rng.integers(info.min // 4, info.max // 4, size=B_shape, dtype=dtype_in)
    else:
        A_np = (rng.random((opts.M, opts.K)) * 4.0).astype(dtype_in)
        B_shape = (opts.N, opts.K) if opts.b_col_maj else (opts.K, opts.N)
        B_np = (rng.random(B_shape) * 4.0).astype(dtype_in)
    A_t = iron.tensor(A_np.reshape(-1), dtype=dtype_in, device="npu")
    B_t = iron.tensor(B_np.reshape(-1), dtype=dtype_in, device="npu")
    C_t = iron.zeros(opts.M * opts.N, dtype=dtype_out, device="npu")

    bench = run_iters(
        single_core,
        A_t,
        B_t,
        C_t,
        M=opts.M,
        K=opts.K,
        N=opts.N,
        m=opts.m,
        k=opts.k,
        n=opts.n,
        dtype_in_str=opts.dtype_in,
        dtype_out_str=opts.dtype_out,
        b_col_maj=opts.b_col_maj,
        emulate_bf16_mmul_with_bfp16=bool(opts.emulate_bf16_mmul_with_bfp16),
        use_chess=bool(opts.use_chess),
        trace_config=_trace_config(opts),
        warmup=opts.warmup,
        iters=opts.iters,
    )

    expected = _numpy_reference(A_np, B_np, opts.b_col_maj, dtype_out)
    actual = C_t.numpy().reshape(opts.M, opts.N)

    assert_close_with_benchmark(
        actual,
        expected,
        bench=bench,
        ops=2.0 * opts.M * opts.K * opts.N,
        fail_msg="output does not match A @ B",
    )


def _compile_kwargs(opts):
    return dict(
        M=opts.M,
        K=opts.K,
        N=opts.N,
        m=opts.m,
        k=opts.k,
        n=opts.n,
        dtype_in_str=opts.dtype_in,
        dtype_out_str=opts.dtype_out,
        b_col_maj=opts.b_col_maj,
        emulate_bf16_mmul_with_bfp16=bool(opts.emulate_bf16_mmul_with_bfp16),
        use_chess=bool(opts.use_chess),
        trace_config=_trace_config(opts),
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        single_core,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
    )


if __name__ == "__main__":
    main()
