#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Whole-array matrix multiplication with dispatch-time M, K and N.

The design in whole_array.py bakes the problem shape into the compiled
artifact. This variant compiles once for a *capacity* (``M_max``, ``K_max``,
``N_max``, the largest matrices the host buffers hold) and takes the live
shape as ``DispatchTime`` scalars: every call rebuilds the instruction stream
on the host from the same xclbin, in C++, in microseconds, with no Python in
the loop. The tiling logic is the same taplib algebra as the static design,
evaluated inside the runtime-sequence body on staged values: taps come out as
arithmetic on ``M``, ``K``, ``N``, the time-block loop stays rolled
(``range_``) with the in-flight step's ``TaskGroup`` carried as an iter_arg so
two halves stay in flight as in the static design, the ragged last row block is
a peeled ``if_``, and the shape constraints the static design asserts become
``require`` guards that refuse an illegal dispatch before anything reaches
the NPU.

Per-core trip counts (``K // k`` and the number of output tiles a core
produces) reach the workers as runtime parameters: each worker owns an RTP
buffer the sequence writes before releasing a barrier the worker waits on.

A fully static specialization (``whole_array_dyn.specialize(M=.., K=.., N=..)``)
takes the ordinary static path and issues exactly the transfers whole_array.py
issues for that shape; ``test/python/dispatch_taplib_gemm.py`` checks the
dynamic builder against it with ``aie.utils.txn_trace``.
"""

import argparse
import sys
from typing import Any

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.taplib.symbolic import require
from aie.iron import (
    Buffer,
    CompileTime,
    DispatchTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    WorkerRuntimeBarrier,
    kernels,
    str_to_dtype,
)
from aie.iron.controlflow import if_, range_, yield_
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark
from whole_array import _device_for


def _build_design(
    dev,
    M_max,
    K_max,
    N_max,
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
    c_col_maj,
    use_chess,
):
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols
    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        b_col_maj=bool(b_col_maj),
        c_col_maj=bool(c_col_maj),
        use_chess=use_chess,
    )
    zero_kernel = kernels.zero(m * n, dtype_out, use_chess=use_chess)
    dims = matmul_kernel.stream_dims

    if n_aie_cols > dev.cols:
        raise ValueError(
            f"n_aie_cols={n_aie_cols} but the device has {dev.cols} columns"
        )
    # Capacity constraints are static; the live-shape constraints are guards in
    # the sequence body.
    assert M_max % (m * n_aie_rows) == 0
    assert K_max % k == 0
    assert N_max % (n * n_aie_cols) == 0
    assert max(M_max * K_max, K_max * N_max, M_max * N_max) < 2**31

    fifo_depth = 2
    n_shim_mem_A = n_aie_rows if n_aie_cols > n_aie_rows else n_aie_cols
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    # Host buffers are allocated at capacity; the live (M, K) etc. matrices are
    # packed row-major at the front.
    A_ty = np.ndarray[(M_max * K_max,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K_max * N_max,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M_max * N_max,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
    rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]

    A_l3l2_fifos: list[ObjectFifo] = []
    A_l2l1_fifos: list[ObjectFifo] = []
    B_l3l2_fifos: list[ObjectFifo] = []
    B_l2l1_fifos: list[ObjectFifo] = []
    C_l1l2_fifos: list[list[ObjectFifo]] = [[] for _ in range(n_aie_rows)]
    C_l2l3_fifos: list[ObjectFifo] = []

    for i in range(n_shim_mem_A):
        a_l3l2 = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        A_l3l2_fifos.append(a_l3l2)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        A_l2l1_fifos.extend(
            a_l3l2.cons().split(
                [m * k * j for j in range(stop_row - start_row)],
                obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
                to_stream=[dims.A or []] * (stop_row - start_row),
            )
        )

    for col in range(n_aie_cols):
        b_l3l2 = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2_fifos.append(b_l3l2)
        B_l2l1_fifos.append(
            b_l3l2.cons().forward(
                obj_type=B_l1_ty, name=f"B_L2L1_{col}", to_stream=dims.B
            )
        )
        c_l2l3 = ObjectFifo(
            C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth, to_stream=dims.C
        )
        C_l2l3_fifos.append(c_l2l3)
        c_tmp_fifos = c_l2l3.prod().join(
            [m * n * i for i in range(n_aie_rows)],
            obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[fifo_depth] * n_aie_rows,
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j].append(c_tmp_fifos[j])

    # Each worker's trip counts arrive as runtime parameters: rtp[0] = K // k,
    # rtp[1] = output tiles per core. The sequence writes them, then releases
    # the barrier the worker waits on at the top of every dispatch.
    rtps = [
        [
            Buffer(rtp_ty, name=f"rtp_{row}_{col}", use_write_rtp=True)
            for col in range(n_aie_cols)
        ]
        for row in range(n_aie_rows)
    ]
    barriers = [
        [WorkerRuntimeBarrier() for _ in range(n_aie_cols)] for _ in range(n_aie_rows)
    ]

    def core_fn(in_a, in_b, out_c, zero, matmul, rtp, barrier):
        barrier.wait_for_value(1)
        k_iters = rtp[0]
        n_tiles = rtp[1]
        for _ in range_(n_tiles):
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(k_iters):
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
                rtps[row][col],
                barriers[row][col],
            ],
            stack_size=0xD00,
        ),
    )
    flat_workers = [w for row in workers for w in row]

    tb_max_n_rows = 4 if not c_col_maj else 2
    tb_n_rows = tb_max_n_rows // 2

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]

    def sequence(A, B, C, M, K, N, A_hs, B_hs, C_hs):
        # The static design's asserts, as guards: a ValueError on a static
        # specialization, a refused dispatch (no stream) on the dynamic path.
        require(M > 0, "M must be positive")
        require(K > 0, "K must be positive")
        require(N > 0, "N must be positive")
        require(M % (m * n_aie_rows) == 0, "M must be a multiple of m * n_aie_rows")
        require(K % k == 0, "K must be a multiple of k")
        require(N % (n * n_aie_cols) == 0, "N must be a multiple of n * n_aie_cols")
        # x * y <= cap as x <= cap // y: the i32 product could overflow.
        require(M <= (M_max * K_max) // K, "A exceeds the compiled capacity")
        require(K <= (K_max * N_max) // N, "B exceeds the compiled capacity")
        require(M <= (M_max * N_max) // N, "C exceeds the compiled capacity")

        k_iters = K // k
        n_tiles_per_core = (M // m) * (N // n) // n_aie_cores
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                rtps[row][col][0] = k_iters
                rtps[row][col][1] = n_tiles_per_core
                barriers[row][col].set(1)

        # The same tilers as whole_array.py, on the live shape. Every grid
        # size, stride and index below is staged arithmetic on M, K, N.
        A_tiles = (
            TensorAccessPattern.full((M, K))
            .tile((m * n_A_tiles_per_shim, k))
            .group((1, K // k))
            .repeat(N // n // n_aie_cols)
        )
        if b_col_maj:
            B_tiles = (
                TensorAccessPattern.full((N, K))
                .tile((n, k))
                .group((N // n // n_aie_cols, K // k), steps=(n_aie_cols, 1))
            )
        else:
            B_tiles = (
                TensorAccessPattern.full((K, N))
                .tile((k, n))
                .group(
                    (K // k, N // n // n_aie_cols),
                    steps=(1, n_aie_cols),
                    order="col",
                )
            )
        if c_col_maj:
            C_tiles = (
                TensorAccessPattern.full((N, M))
                .tile((n, m))
                .order("col")
                .group((N // n // n_aie_cols, n_aie_rows), steps=(n_aie_cols, 1))
            )
        else:
            # partial: the last group may hold fewer than tb_n_rows row blocks.
            C_tiles = (
                TensorAccessPattern.full((M, N))
                .tile((m * n_aie_rows, n))
                .group(
                    (tb_n_rows, N // n // n_aie_cols),
                    steps=(1, n_aie_cols),
                    partial=True,
                )
            )

        n_row_tiles = M // m // n_aie_rows
        n_full_steps = n_row_tiles // tb_n_rows
        n_ragged_rows = n_row_tiles % tb_n_rows

        def issue_step(step, n_rows):
            """Issue one time block half: n_rows (static) row blocks per column."""
            tg = TaskGroup()
            row_base = step * tb_n_rows
            for col in range(n_aie_cols):
                C_hs[col].drain(
                    C, tap=C_tiles[step * n_aie_cols + col], wait=True, group=tg
                )
                for tile_row in range(n_rows):
                    tile_offset = (
                        (row_base + tile_row) * n_shim_mem_A + col
                    ) % A_tiles.num_steps
                    if col < n_aie_rows:
                        A_hs[col].fill(A, tap=A_tiles[tile_offset], group=tg)
                    B_hs[col].fill(B, tap=B_tiles[col], group=tg)
            return tg

        # Two time-block halves in flight, as in whole_array.py: a step's group
        # is finished only after the next step's is issued. The in-flight group
        # rides the loop as an iter_arg, so the loop stays rolled over the
        # dispatch-time trip count.
        with if_(n_full_steps > 0):
            prev = issue_step(0, tb_n_rows)
            last = prev
            for iv, prev, last in range_(
                1, n_full_steps, iter_args=[prev], insert_yield=False
            ):
                current = issue_step(iv, tb_n_rows)
                prev.finish()
                yield_([current])
            last.finish()
        if tb_n_rows > 1:
            with if_(n_ragged_rows > 0):
                issue_step(n_full_steps, 1).finish()

    rt = Runtime(sequence, [A_ty, B_ty, C_ty, M, K, N, A_prods, B_prods, C_conses])
    return Program(dev, rt, workers=flat_workers).resolve_program()


@iron.jit
def whole_array_dyn(
    A: In,
    B: In,
    C: Out,
    *,
    M: DispatchTime[np.int32],
    K: DispatchTime[np.int32],
    N: DispatchTime[np.int32],
    M_max: CompileTime[int],
    K_max: CompileTime[int],
    N_max: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    n: CompileTime[int],
    n_aie_cols: CompileTime[int],
    dtype_in_str: CompileTime[str],
    dtype_out_str: CompileTime[str],
    b_col_maj: CompileTime[int] = 0,
    c_col_maj: CompileTime[int] = 0,
    use_chess: CompileTime[bool] = False,
):
    return _build_design(
        iron.get_current_device(),
        M_max,
        K_max,
        N_max,
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
        c_col_maj,
        use_chess,
    )


def _make_argparser():
    p = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication (Whole Array, dispatch-time shape)"
    )
    add_compile_args(p, short_dev=None)
    p.add_argument("-M", type=int, default=512)
    p.add_argument("-K", type=int, default=512)
    p.add_argument("-N", type=int, default=512)
    p.add_argument("--M-max", type=int, default=None, help="capacity (default: M)")
    p.add_argument("--K-max", type=int, default=None, help="capacity (default: K)")
    p.add_argument("--N-max", type=int, default=None, help="capacity (default: N)")
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=32)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    p.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument("--c-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument("--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16")
    p.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i32",
    )
    p.add_argument("--use-chess", type=int, choices=[0, 1], default=0)
    add_benchmark_args(p)
    return p


def _compile_kwargs(opts) -> dict[str, Any]:
    return dict(
        M_max=opts.M_max or opts.M,
        K_max=opts.K_max or opts.K,
        N_max=opts.N_max or opts.N,
        m=opts.m,
        k=opts.k,
        n=opts.n,
        n_aie_cols=opts.n_aie_cols,
        dtype_in_str=opts.dtype_in,
        dtype_out_str=opts.dtype_out,
        b_col_maj=opts.b_col_maj,
        c_col_maj=opts.c_col_maj,
        use_chess=bool(opts.use_chess),
    )


def _run_and_verify(opts):
    """Run the live shape through the capacity-compiled design and check it."""
    dtype_in = str_to_dtype(opts.dtype_in)
    dtype_out = str_to_dtype(opts.dtype_out)
    M_max, K_max, N_max = (
        opts.M_max or opts.M,
        opts.K_max or opts.K,
        opts.N_max or opts.N,
    )

    rng = np.random.default_rng(1726250518)
    B_shape = (opts.N, opts.K) if opts.b_col_maj else (opts.K, opts.N)
    if np.issubdtype(dtype_in, np.integer):
        info = np.iinfo(dtype_in)
        A_np = rng.integers(
            info.min // 4, info.max // 4, size=(opts.M, opts.K), dtype=dtype_in
        )
        B_np = rng.integers(info.min // 4, info.max // 4, size=B_shape, dtype=dtype_in)
    else:
        A_np = (rng.random((opts.M, opts.K)) * 4.0).astype(dtype_in)
        B_np = (rng.random(B_shape) * 4.0).astype(dtype_in)

    # Host buffers are the compiled capacity; the live matrices sit packed at
    # the front.
    def at_capacity(arr, capacity):
        buf = np.zeros((capacity,), dtype=arr.dtype)
        buf[: arr.size] = arr.reshape(-1)
        return iron.tensor(buf, dtype=arr.dtype, device="npu")

    A_t = at_capacity(A_np, M_max * K_max)
    B_t = at_capacity(B_np, K_max * N_max)
    C_t = iron.zeros((M_max * N_max,), dtype=dtype_out, device="npu")

    bench = run_iters(
        whole_array_dyn,
        A_t,
        B_t,
        C_t,
        M=opts.M,
        K=opts.K,
        N=opts.N,
        **_compile_kwargs(opts),
        warmup=opts.warmup,
        iters=opts.iters,
    )

    B_logical = B_np.T if opts.b_col_maj else B_np
    expected_logical = kernels.mm_ref(A_np, B_logical).astype(dtype_out)
    live = C_t.numpy()[: opts.M * opts.N]
    if opts.c_col_maj:
        actual, expected = live.reshape(opts.N, opts.M), expected_logical.T
    else:
        actual, expected = live.reshape(opts.M, opts.N), expected_logical
    kernel = kernels.mm(
        dim_m=opts.m,
        dim_k=opts.k,
        dim_n=opts.n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        b_col_maj=bool(opts.b_col_maj),
    )
    assert_close_with_benchmark(
        actual,
        expected,
        bench=bench,
        ops=2.0 * opts.M * opts.K * opts.N,
        tolerance=kernel.contract.tolerance,
        fail_msg="output does not match A @ B",
        mismatch_indices=True,
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        whole_array_dyn,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: _device_for(o.dev, o.n_aie_cols),
    )


if __name__ == "__main__":
    sys.exit(main())
