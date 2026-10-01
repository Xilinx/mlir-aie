#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Whole-array matrix multiply — IRON API design with ``@iron.jit`` compilation.

A 4xN_cols AIE array computes ``C = A @ B`` (optionally with ``B`` column-major
or ``C`` column-major).  Each compute tile owns one (m, n) output sub-tile and
streams (m, k) x (k, n) inputs through three layers of ObjectFifos.

The script has two modes:

* ``--xclbin-path=... --insts-path=...`` — compile the design ahead-of-time
  and write artifacts to the given paths (bypasses the JIT cache).  Used by
  ``makefile-common`` so ``test.cpp`` + ``sweep.sh`` can drive the design
  via ``make``.
* default — JIT-compile + run on the attached NPU + verify against numpy.
"""

import argparse
import sys
from typing import NamedTuple

import aie.iron as iron
import aie.iron.kernels as kernels
import numpy as np
from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence, TileGrid
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, from_name
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark


def _device_for(dev_str, n_aie_cols):
    # On NPU1 pick the matching ColN variant (or NPU1 itself when
    # n_aie_cols == max = 4).  On NPU2 use the unrestricted device
    # regardless of n_aie_cols so the placer has the full 8-column array.
    return from_name(dev_str, n_cols=n_aie_cols if dev_str == "npu" else None)


class TileGrids(NamedTuple):
    """The A, B and C tile grids the runtime sequence fills and drains."""

    A: TileGrid
    B: TileGrid
    C: TileGrid
    rows_per_step: int  # row blocks per time-block half


def tile_grids(M, K, N, m, k, n, n_aie_cols, b_col_maj, c_col_maj) -> TileGrids:
    """Tile A, B and C for the runtime sequence.

    ``M``, ``K`` and ``N`` may be ints or dispatch-time scalars:
    whole_array_dyn.py builds the same grids on the live shape.
    """
    n_aie_rows = 4
    n_A_tiles_per_shim = max(1, n_aie_rows // n_aie_cols)
    rows_per_step = 1 if c_col_maj else 2

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
            .group((K // k, N // n // n_aie_cols), steps=(1, n_aie_cols), order="col")
        )
    if c_col_maj:
        # Splitting n_aie_rows out of the tile dim is what lets the grouping emit
        # the (col-fast, row_block-slow) DMA pattern; order("col") matches it.
        C_tiles = (
            TensorAccessPattern.full((N, M))
            .tile((n, m))
            .order("col")
            .group((N // n // n_aie_cols, n_aie_rows), steps=(n_aie_cols, 1))
        )
    else:
        # partial: the last step may hold fewer than rows_per_step row blocks.
        C_tiles = (
            TensorAccessPattern.full((M, N))
            .tile((m * n_aie_rows, n))
            .group(
                (rows_per_step, N // n // n_aie_cols),
                steps=(1, n_aie_cols),
                partial=True,
            )
        )
    return TileGrids(A_tiles, B_tiles, C_tiles, rows_per_step)


def step_transfers(grids: TileGrids, step, n_rows: int, n_aie_cols: int) -> list:
    """Return the shim transfers of one time-block half.

    Each transfer is ``(tensor, col, tap)``: the shim column ``col`` fills
    ``"A"`` or ``"B"``, or drains ``"C"``, with the access pattern ``tap``.
    ``step`` may be a dispatch-time scalar; ``n_rows`` row blocks are issued.
    """
    n_aie_rows = 4
    n_shim_mem_A = min(n_aie_rows, n_aie_cols)
    row_base = step * grids.rows_per_step
    transfers = []
    for col in range(n_aie_cols):
        transfers.append(("C", col, grids.C[step * n_aie_cols + col]))
        for tile_row in range(n_rows):
            tile_offset = (
                (row_base + tile_row) * n_shim_mem_A + col
            ) % grids.A.num_steps
            if col < n_aie_rows:
                transfers.append(("A", col, grids.A[tile_offset]))
            transfers.append(("B", col, grids.B[col]))
    return transfers


def issue(transfers, A, B, C, A_hs, B_hs, C_hs) -> TaskGroup:
    """Issue ``transfers`` as one group; the C drains are waited on."""
    tg = TaskGroup()
    for tensor, col, tap in transfers:
        if tensor == "C":
            C_hs[col].drain(C, tap=tap, wait=True, group=tg)
        elif tensor == "A":
            A_hs[col].fill(A, tap=tap, group=tg)
        else:
            B_hs[col].fill(B, tap=tap, group=tg)
    return tg


def _transfer_groups(M, K, N, m, k, n, n_aie_cols, b_col_maj, c_col_maj):
    """Return the runtime sequence's shim transfers, one list per time-block half."""
    grids = tile_grids(M, K, N, m, k, n, n_aie_cols, b_col_maj, c_col_maj)
    n_row_tiles = M // m // 4
    return [
        step_transfers(
            grids,
            step,
            min(grids.rows_per_step, n_row_tiles - step * grids.rows_per_step),
            n_aie_cols,
        )
        for step in range(iron.ceildiv(n_row_tiles, grids.rows_per_step))
    ]


def fifos(m, k, n, n_aie_cols, dtype_in, dtype_out, dims):
    """Build the shim -> memtile -> core fifos.

    Returns the A and B fifos the runtime fills, the C fifos it drains, and
    the core-facing A (per row), B (per column) and C (``[row][col]``) fifos.
    """
    n_aie_rows = 4
    fifo_depth = 2
    n_shim_mem_A = min(n_aie_rows, n_aie_cols)
    n_A_tiles_per_shim = max(1, n_aie_rows // n_aie_cols)

    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

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

    return (
        A_l3l2_fifos,
        B_l3l2_fifos,
        C_l2l3_fifos,
        A_l2l1_fifos,
        B_l2l1_fifos,
        C_l1l2_fifos,
    )


def _build_design(
    dev,
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
    emulate_bf16_mmul_with_bfp16,
    use_chess,
    scalar,
):
    """Build the whole-array matmul IRON design and resolve to MLIR."""
    dev_str = "npu2" if isinstance(dev, NPU2) else "npu"

    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        b_col_maj=bool(b_col_maj),
        c_col_maj=bool(c_col_maj),
        use_chess=use_chess,
        emulate_bf16_mmul_with_bfp16=emulate_bf16_mmul_with_bfp16,
        vectorized=not scalar,
    )
    zero_kernel = kernels.zero(m * n, dtype_out, use_chess=use_chess)
    r, s, t = matmul_kernel.mac_dims
    dims = matmul_kernel.stream_dims

    if n_aie_cols > dev.cols:
        raise ValueError(
            f"n_aie_cols={n_aie_cols} but {dev_str} has {dev.cols} columns"
        )

    assert (
        M % (m * n_aie_rows) == 0
    ), "A must be tileable into (m * n_aie_rows, k)-sized blocks"
    assert K % k == 0
    assert (
        N % (n * n_aie_cols) == 0
    ), "B must be tileable into (k, n * n_aie_cols)-sized blocks"
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    (
        A_l3l2_fifos,
        B_l3l2_fifos,
        C_l2l3_fifos,
        A_l2l1_fifos,
        B_l2l1_fifos,
        C_l1l2_fifos,
    ) = fifos(m, k, n, n_aie_cols, dtype_in, dtype_out, dims)

    def core_fn(in_a, in_b, out_c, zero, matmul):
        loop = range(1)  # Workaround for issue #1547
        if n_tiles_per_core > 1:
            loop = range_(n_tiles_per_core)
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

    flat_workers = [w for row in workers for w in row]

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]
    groups = _transfer_groups(M, K, N, m, k, n, n_aie_cols, b_col_maj, c_col_maj)

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        # Two time-block halves in flight: each half's transfers form a
        # group that is finished only after the next half's are issued.
        prev = None
        for transfers in groups:
            tg = issue(transfers, A, B, C, A_hs, B_hs, C_hs)
            if prev is not None:
                prev.finish()
            prev = tg
        if prev is not None:
            prev.finish()

    rt = Runtime(
        sequence,
        [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses],
    )

    return Program(dev, rt, workers=flat_workers).resolve_program()


@iron.jit
def whole_array(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    n: CompileTime[int],
    n_aie_cols: CompileTime[int],
    dtype_in_str: CompileTime[str],
    dtype_out_str: CompileTime[str],
    b_col_maj: CompileTime[int] = 0,
    c_col_maj: CompileTime[int] = 0,
    emulate_bf16_mmul_with_bfp16: CompileTime[bool] = False,
    use_chess: CompileTime[bool] = False,
    scalar: CompileTime[bool] = False,
):
    return _build_design(
        iron.get_current_device(),
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
        emulate_bf16_mmul_with_bfp16,
        use_chess,
        scalar,
    )


def generate_taps(M, K, N, m, k, n, n_aie_cols, b_col_maj=0, c_col_maj=0):
    """Return ``(A_taps, B_taps, C_taps)`` for the visualization notebook.

    Each is a ``TensorAccessSequence`` of the patterns the runtime sequence
    fills or drains for that matrix, in order.
    """
    groups = _transfer_groups(M, K, N, m, k, n, n_aie_cols, b_col_maj, c_col_maj)
    return tuple(
        TensorAccessSequence.from_taps(
            [tap for transfers in groups for t, _, tap in transfers if t == tensor]
        )
        for tensor in "ABC"
    )


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Matrix Multiplication (Whole Array)")
    add_compile_args(p, short_dev=None)
    p.add_argument("-M", type=int, default=512)
    p.add_argument("-K", type=int, default=512)
    p.add_argument("-N", type=int, default=512)
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=32)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    p.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument("--c-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument(
        "--emulate-bf16-mmul-with-bfp16", type=int, choices=[0, 1], default=0
    )
    p.add_argument("--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16")
    p.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i16",
    )
    p.add_argument("--use-chess", type=int, choices=[0, 1], default=0)
    p.add_argument(
        "--scalar",
        type=int,
        choices=[0, 1],
        default=0,
        help="use scalar (non-vector) matmul/zero kernels for debugging smaller sizes",
    )
    add_benchmark_args(p)
    return p


def _validate_shape_args(opts):
    n_aie_rows = 4
    if opts.M % (opts.m * n_aie_rows) != 0:
        sys.exit(
            f"-M {opts.M} must be a multiple of -m * n_aie_rows ({opts.m} * {n_aie_rows} = {opts.m * n_aie_rows})"
        )
    if opts.K % opts.k != 0:
        sys.exit(f"-K {opts.K} must be a multiple of -k {opts.k}")
    if opts.N % (opts.n * opts.n_aie_cols) != 0:
        sys.exit(
            f"-N {opts.N} must be a multiple of -n * --n-aie-cols ({opts.n} * {opts.n_aie_cols} = {opts.n * opts.n_aie_cols})"
        )
    tb_n_rows = 2
    n_row_blocks = opts.M // opts.m // n_aie_rows
    if n_row_blocks % tb_n_rows != 0:
        sys.exit(
            f"M/m/n_aie_rows = {n_row_blocks} must be a multiple of "
            f"{tb_n_rows} (transfer-block row count). Try a larger -M or smaller -m."
        )
    if opts.dev == "npu" and opts.n_aie_cols > 4:
        sys.exit(
            f"--n-aie-cols {opts.n_aie_cols} > 4 not supported on NPU1 (Phoenix/Hawk)"
        )
    if opts.dev == "npu2" and opts.n_aie_cols > 8:
        sys.exit(
            f"--n-aie-cols {opts.n_aie_cols} > 8 not supported on NPU2 (Strix/Strix Halo/Krackan)"
        )


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
        b_col_maj=opts.b_col_maj,
        c_col_maj=opts.c_col_maj,
        emulate_bf16_mmul_with_bfp16=bool(opts.emulate_bf16_mmul_with_bfp16),
        use_chess=bool(opts.use_chess),
        scalar=bool(opts.scalar),
    )


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
    A_t = iron.tensor(A_np, dtype=dtype_in, device="npu")
    B_t = iron.tensor(B_np, dtype=dtype_in, device="npu")
    C_t = iron.zeros((opts.M, opts.N), dtype=dtype_out, device="npu")

    bench = run_iters(
        whole_array,
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
        b_col_maj=opts.b_col_maj,
        c_col_maj=opts.c_col_maj,
        emulate_bf16_mmul_with_bfp16=bool(opts.emulate_bf16_mmul_with_bfp16),
        use_chess=bool(opts.use_chess),
        scalar=bool(opts.scalar),
        warmup=opts.warmup,
        iters=opts.iters,
    )

    B_logical = B_np.T if opts.b_col_maj else B_np  # b_col_maj stores B transposed
    expected_logical = kernels.mm_ref(A_np, B_logical).astype(dtype_out)
    if opts.c_col_maj:
        actual = C_t.numpy().reshape(opts.N, opts.M)
        expected = expected_logical.T
    else:
        actual = C_t.numpy().reshape(opts.M, opts.N)
        expected = expected_logical

    # The same kernel the design binds; kernels.mm is memoized, so this is
    # the object the design built, asked for its tolerance.
    kernel = kernels.mm(
        dim_m=opts.m,
        dim_k=opts.k,
        dim_n=opts.n,
        input_dtype=str_to_dtype(opts.dtype_in),
        output_dtype=str_to_dtype(opts.dtype_out),
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
        whole_array,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: _device_for(o.dev, o.n_aie_cols),
        validate=_validate_shape_args,
    )


if __name__ == "__main__":
    main()
