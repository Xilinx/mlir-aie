#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc.
"""Cascade matrix multiply — Iron API design with ``@iron.jit`` compilation.

A 4xN_cols AIE array computes ``C = A @ B`` using AIE hardware cascade
streams to accumulate partial products vertically within each column.  Row 3
(bottom) puts onto the cascade; rows 1-2 read+put; row 0 (top) reads and
writes the final C tile to L2.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import (
    Buffer,
    CascadeFlow,
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
from aie.iron.device import Tile, from_name
from aie.helpers.taplib import TensorTiler2D
from aie.utils.benchmark import print_benchmark, run_iters
from aie.utils.hostruntime.argparse import (
    add_benchmark_args,
    add_compile_args,
    add_trace_arg,
)
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


def _device_for(dev_str, n_aie_cols):
    # On NPU1 pick the matching ColN variant (or NPU1 itself when
    # n_aie_cols == max = 4).  On NPU2 use the unrestricted device
    # regardless of n_aie_cols so the placer has the full 8-column array.
    return from_name(dev_str, n_cols=n_aie_cols if dev_str == "npu" else None)


def ceildiv(a, b):
    return (a + b - 1) // b


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def cascade(
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
    n_aie_cols: Compile[int],
    dtype_in_str: Compile[str],
    dtype_out_str: Compile[str],
):
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    cascade_kernel = kernels.cascade_mm(
        dim_m=m, dim_k=k, dim_n=n, input_dtype=dtype_in, output_dtype=dtype_out
    )
    matmul_get_only = cascade_kernel.get_only
    matmul_put_only = cascade_kernel.put_only
    matmul_put_get = cascade_kernel.put_get
    zero_kernel = cascade_kernel.zero

    if dtype_in_str == "bf16":
        r, s, t = 4, 8, 4
    elif dtype_in_str == "i16":
        r, s, t = 1, 1, 1
    else:
        raise ValueError(f"unsupported dtype_in={dtype_in_str!r}")

    assert M % m == 0
    assert K % (k * n_aie_rows) == 0
    assert N % (n * n_aie_cols) == 0
    assert m % r == 0 and k % s == 0 and n % t == 0

    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < n_aie_rows else 1

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n * n_aie_rows,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    fifo_depth = 2

    # A: shim → mem → broadcast across columns within a row
    A_l3l2_fifos = [
        ObjectFifo(A_l2_ty, name=f"A_L3L2_{col}", depth=fifo_depth)
        for col in range(n_aie_cols)
    ]
    A_l2l1_fifos = [None] * n_aie_rows
    for col in range(n_aie_cols):
        start_row = col * n_A_tiles_per_shim
        of_offsets = [m * k * j for j in range(n_A_tiles_per_shim)]
        a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
        # Each row's L2→L1 fifo is broadcast to all n_aie_cols core_tiles in
        # that row.  split() returns one handle per output, but we want one
        # logical broadcast fifo per row.  Using forward() with a list of
        # destination tiles broadcasts.
        fifos = (
            A_l3l2_fifos[col]
            .cons()
            .split(
                of_offsets,
                obj_types=[A_l1_ty] * n_A_tiles_per_shim,
                names=[
                    f"A_L2L1_{r_}"
                    for r_ in range(start_row, start_row + n_A_tiles_per_shim)
                ],
                dims_to_stream=[a_dims] * n_A_tiles_per_shim,
                tile=Tile(col, 1),
            )
        )
        for j, f in enumerate(fifos):
            A_l2l1_fifos[start_row + j] = f

    # B: shim → mem → distribute across rows within a column
    B_l3l2_fifos = [
        ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        for col in range(n_aie_cols)
    ]
    B_l2l1_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    for col in range(n_aie_cols):
        of_offsets = [k * n * row for row in range(n_aie_rows)]
        b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        fifos = (
            B_l3l2_fifos[col]
            .cons()
            .split(
                of_offsets,
                obj_types=[B_l1_ty] * n_aie_rows,
                names=[f"B_L2L1_{col}_{row}" for row in range(n_aie_rows)],
                dims_to_stream=[b_dims] * n_aie_rows,
                tile=Tile(col, 1),
            )
        )
        for row in range(n_aie_rows):
            B_l2l1_fifos[row][col] = fifos[row]

    # C output (row 0 only): L1 → mem → shim.  Only one producer per column,
    # so this is a simple forward chain with the L2→L3 dim transform applied
    # at the mem-tile boundary.
    C_l1l2_fifos = []
    C_l2l3_fifos = []
    for col in range(n_aie_cols):
        c_l1l2 = ObjectFifo(C_l1_ty, name=f"C_L1L2_{col}", depth=fifo_depth)
        c_l2l3 = c_l1l2.cons().forward(
            obj_type=C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)],
            tile=Tile(col, 1),
        )
        C_l1l2_fifos.append(c_l1l2)
        C_l2l3_fifos.append(c_l2l3)

    # Per-row worker bodies.  The cascade kernels themselves talk to the
    # hardware cascade ports via put_mcd/get_scd inside cascade_mm.cc; the
    # only thing the design has to declare is the directed cascade edge
    # (CascadeFlow) between vertically adjacent workers.
    #
    # Accumulation flows bottom-up: row 3 (put_only) → row 2 (put_get) →
    # row 1 (put_get) → row 0 (get_only, also zeroes + writes C).

    def _row0_fn(in_a, in_b, out_c, zero, get_only):
        loop = range_(n_tiles_per_core) if n_tiles_per_core > 1 else range(1)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(K // k // n_aie_rows):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                get_only(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    def _row_mid_fn(in_a, in_b, c_buf, put_get):
        loop = range_(n_tiles_per_core) if n_tiles_per_core > 1 else range(1)
        for _ in loop:
            for _ in range_(K // k // n_aie_rows):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                # The kernel reads cascade-in via get_scd, writes cascade-out
                # via put_mcd, and uses c_buf as scratch for the local accum
                # that gets passed up the cascade.
                put_get(elem_in_a, elem_in_b, c_buf)
                in_a.release(1)
                in_b.release(1)

    def _row_bot_fn(in_a, in_b, c_buf, put_only):
        loop = range_(n_tiles_per_core) if n_tiles_per_core > 1 else range(1)
        for _ in loop:
            for _ in range_(K // k // n_aie_rows):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                put_only(elem_in_a, elem_in_b, c_buf)
                in_a.release(1)
                in_b.release(1)

    workers = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    for col in range(n_aie_cols):
        # Row 0 (top — get_only, zeroes + writes C)
        workers[0][col] = Worker(
            _row0_fn,
            [
                A_l2l1_fifos[0].cons(),
                B_l2l1_fifos[0][col].cons(),
                C_l1l2_fifos[col].prod(),
                zero_kernel,
                matmul_get_only,
            ],
            tile=Tile(col, 2),
        )
        # Mid rows (put_get) — each gets a per-tile C scratch Buffer.  The
        # cascade kernel's C arg signature is 1D (matches kernels.cascade_mm's
        # arg_types), so allocate the scratch buffer 1D too.
        c_buf_ty = np.ndarray[(m * n,), np.dtype[dtype_out]]
        for row in (1, 2):
            c_buf = Buffer(c_buf_ty, name=f"C_scratch_{col}_{row}")
            workers[row][col] = Worker(
                _row_mid_fn,
                [
                    A_l2l1_fifos[row].cons(),
                    B_l2l1_fifos[row][col].cons(),
                    c_buf,
                    matmul_put_get,
                ],
                tile=Tile(col, 2 + row),
            )
        # Row n_aie_rows-1 (bottom — put_only)
        c_buf_bot = Buffer(c_buf_ty, name=f"C_scratch_{col}_{n_aie_rows - 1}")
        workers[n_aie_rows - 1][col] = Worker(
            _row_bot_fn,
            [
                A_l2l1_fifos[n_aie_rows - 1].cons(),
                B_l2l1_fifos[n_aie_rows - 1][col].cons(),
                c_buf_bot,
                matmul_put_only,
            ],
            tile=Tile(col, 2 + n_aie_rows - 1),
        )

    # Cascade edges: row 3 → row 2 → row 1 → row 0 (within each column).
    for col in range(n_aie_cols):
        for row in range(n_aie_rows - 1, 0, -1):
            CascadeFlow(workers[row][col], workers[row - 1][col])

    flat_workers = [w for row in workers for w in row]

    tb_max_n_rows = 5

    # C drain TAPs: one per (tb, col).  step_tiler with allow_partial=True
    # handles the trailing tb that has fewer than tb_max_n_rows rows.
    C_taps = TensorTiler2D.step_tiler(
        (M, N),
        (m, n),
        tile_group_repeats=(tb_max_n_rows, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
        allow_partial=True,
        prune_step=False,
    )

    # B fill TAPs: one per col, reused across all (tb, tile_row) for that col.
    B_taps = TensorTiler2D.step_tiler(
        (K, N),
        (k * n_aie_rows, n),
        tile_group_repeats=(K // k // n_aie_rows, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
        tile_group_col_major=True,
        prune_step=False,
    )

    # A fill TAPs: one per (col, m-block).  Indexed by (m_block_idx * n_aie_cols
    # + col) — m_block_idx walks all M//m rows once, col iterates the columns
    # for each row.  Each TAP repeats N//n//n_aie_cols times (broadcast across
    # the N output-column axis) via pattern_repeat.
    A_taps = TensorTiler2D.step_tiler(
        (M, K),
        (m * n_A_tiles_per_shim, k),
        tile_group_repeats=(1, K // k // n_aie_rows),
        tile_group_steps=(1, n_aie_rows),
        pattern_repeat=N // n // n_aie_cols,
        prune_step=False,
    )

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(*flat_workers)

        c_index = 0
        for tb in range(ceildiv(M // m, tb_max_n_rows)):
            tb_n_rows = min([tb_max_n_rows, M // m - tb * tb_max_n_rows])
            tg = rt.task_group()
            for col in range(n_aie_cols):
                rt.drain(
                    C_l2l3_fifos[col].cons(),
                    C,
                    tap=C_taps[c_index],
                    wait=True,
                    task_group=tg,
                    tile=Tile(col, 0),
                )
                c_index += 1
                for tile_row in range(tb_n_rows):
                    a_idx = ((tb * tb_max_n_rows) + tile_row) * n_aie_cols + col
                    rt.fill(
                        A_l3l2_fifos[col].prod(),
                        A,
                        tap=A_taps[a_idx],
                        task_group=tg,
                        tile=Tile(col, 0),
                    )
                    rt.fill(
                        B_l3l2_fifos[col].prod(),
                        B,
                        tap=B_taps[col],
                        task_group=tg,
                        tile=Tile(col, 0),
                    )
            rt.finish_task_group(tg)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Matrix Multiplication (Cascade)")
    add_compile_args(p, short_dev=None)
    p.add_argument("-M", type=int, default=512)
    p.add_argument("-K", type=int, default=512)
    p.add_argument("-N", type=int, default=512)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("-k", type=int, default=32)
    p.add_argument("-n", type=int, default=32)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4], default=4)
    p.add_argument("--dtype_in", type=str, choices=["bf16", "i16"], default="i16")
    p.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i16", "f32", "i32"],
        default="i32",
    )
    p.add_argument("--use-chess", type=int, choices=[0, 1], default=0)
    p.add_argument(
        "--emulate-bf16-mmul-with-bfp16", type=int, choices=[0, 1], default=0
    )
    add_trace_arg(p, with_short=False)
    add_benchmark_args(p)
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
        dtype_in_str=opts.dtype_in,
        dtype_out_str=opts.dtype_out,
    )


def _run_and_verify(opts):
    dtype_in = str_to_dtype(opts.dtype_in)
    dtype_out = str_to_dtype(opts.dtype_out)
    rng = np.random.default_rng(1726250518)
    info = np.iinfo(dtype_in) if np.issubdtype(dtype_in, np.integer) else None
    if info is not None:
        A_np = rng.integers(
            info.min // 4, info.max // 4, size=(opts.M, opts.K), dtype=dtype_in
        )
        B_np = rng.integers(
            info.min // 4, info.max // 4, size=(opts.K, opts.N), dtype=dtype_in
        )
    else:
        A_np = (rng.random((opts.M, opts.K)) * 4.0).astype(dtype_in)
        B_np = (rng.random((opts.K, opts.N)) * 4.0).astype(dtype_in)
    C_np = np.zeros((opts.M, opts.N), dtype=dtype_out)

    A_t = iron.tensor(A_np.reshape(-1), dtype=dtype_in, device="npu")
    B_t = iron.tensor(B_np.reshape(-1), dtype=dtype_in, device="npu")
    C_t = iron.tensor(C_np.reshape(-1), dtype=dtype_out, device="npu")

    bench = run_iters(
        cascade,
        A_t,
        B_t,
        C_t,
        M=opts.M,
        K=opts.K,
        N=opts.N,
        m=opts.m,
        k=opts.k,
        n=opts.n,
        n_aie_cols=opts.n_aie_cols,
        dtype_in_str=opts.dtype_in,
        dtype_out_str=opts.dtype_out,
        warmup=opts.warmup,
        iters=opts.iters,
    )

    actual = C_t.numpy().reshape(opts.M, opts.N)
    if np.issubdtype(dtype_out, np.integer):
        expected = (A_np.astype(np.int64) @ B_np.astype(np.int64)).astype(dtype_out)
        assert_pass(
            actual, expected, fail_msg="output does not match A @ B", print_pass=False
        )
    else:
        expected = (A_np.astype(np.float32) @ B_np.astype(np.float32)).astype(dtype_out)
        assert_pass(
            actual,
            expected,
            rtol=0.05,
            atol=0.5,
            fail_msg="output does not match A @ B",
            print_pass=False,
        )

    print()
    print_benchmark(bench)
    macs = 2.0 * opts.M * opts.K * opts.N
    if bench.npu is not None:
        gflops = macs / (1000 * bench.npu.avg_us)
        print(f"NPU GFLOPS                    : {gflops:.2f}")
    print("PASS!")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        cascade,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: _device_for(o.dev, o.n_aie_cols),
    )


if __name__ == "__main__":
    main()
