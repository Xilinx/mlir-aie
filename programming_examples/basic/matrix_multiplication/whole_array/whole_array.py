#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
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
from aie.iron.device import NPU2, from_name
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark


def _device_for(dev_str, n_aie_cols):
    # On NPU1 pick the matching ColN variant (or NPU1 itself when
    # n_aie_cols == max = 4).  On NPU2 use the unrestricted device
    # regardless of n_aie_cols so the placer has the full 8-column array.
    return from_name(dev_str, n_cols=n_aie_cols if dev_str == "npu" else None)


def ceildiv(a, b):
    return (a + b - 1) // b


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
    *,
    generate_taps=False,
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
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims

    if dev_str == "npu" and n_aie_cols > 4:
        raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 4 columns")
    if dev_str == "npu2" and n_aie_cols > 8:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
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

    fifo_depth = 2
    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    if n_aie_cols > n_aie_rows:
        n_shim_mem_A = n_aie_rows
    else:
        n_shim_mem_A = n_aie_cols

    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    A_taps = []
    B_taps = []
    C_taps = []

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    # L1 ObjectFifo elements are 2D; BaseKernel.__call__ inserts a
    # memref.collapse_shape to bridge to kernels.mm's 1D arg signature.
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    tiles = [[(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows
    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols
    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    for i in range(n_shim_mem_A):
        A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        dims_to_stream = [
            [
                (m // r, r * k),
                (k // s, s),
                (r, k),
                (s, 1),
            ]
        ] * (stop_row - start_row)
        a_tmp_fifos = (
            A_l3l2_fifos[i]
            .cons()
            .split(
                of_offsets,
                obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
                dims_to_stream=dims_to_stream,
            )
        )
        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = a_tmp_fifos[j]

    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        if b_col_maj:
            dims_to_stream = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            dims_to_stream = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos[col] = (
            B_l3l2_fifos[col]
            .cons()
            .forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=dims_to_stream,
            )
        )

        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=(
                [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
                if not c_col_maj
                else [(n // t, t * m), (t, r), (m // r, r * t), (r, 1)]
            ),
        )
        of_offsets = [m * n * i for i in range(n_aie_rows)]

        c_tmp_fifos = (
            C_l2l3_fifos[col]
            .prod()
            .join(
                of_offsets,
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[fifo_depth] * n_aie_rows,
            )
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = c_tmp_fifos[j]

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

    tb_max_n_rows = 4 if not c_col_maj else 2
    tb_n_rows = tb_max_n_rows // 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K),
        (m * n_A_tiles_per_shim, k),
        (1, K // k),
        pattern_repeat=N // n // n_aie_cols,
        prune_step=False,
    )
    if b_col_maj:
        B_tiles = TensorTiler2D.step_tiler(
            (N, K),
            (n, k),
            tile_group_repeats=(N // n // n_aie_cols, K // k),
            tile_group_steps=(n_aie_cols, 1),
            prune_step=False,
        )
    else:
        B_tiles = TensorTiler2D.step_tiler(
            (K, N),
            (k, n),
            tile_group_repeats=(K // k, N // n // n_aie_cols),
            tile_group_steps=(1, n_aie_cols),
            tile_group_col_major=True,
            prune_step=False,
        )
    if c_col_maj:
        # Splitting n_aie_rows out of the tile dim is what lets TensorTiler emit
        # the (col-fast, row_block-slow) DMA pattern; iter_col_major matches it.
        C_tiles = TensorTiler2D.step_tiler(
            (N, M),
            (n, m),
            tile_group_repeats=(N // n // n_aie_cols, n_aie_rows),
            tile_group_steps=(n_aie_cols, 1),
            iter_col_major=True,
            prune_step=False,
        )
    else:
        C_tiles = TensorTiler2D.step_tiler(
            (M, N),
            (m * n_aie_rows, n),
            tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
            tile_group_steps=(1, n_aie_cols),
            prune_step=False,
        )
    c_index = 0

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(*[w for row in workers for w in row])

        tg = rt.task_group()
        for tb in range(ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break

                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                )

                for col in range(n_aie_cols):
                    C_taps.append(C_tiles[c_index])
                    rt.drain(
                        C_l2l3_fifos[col].cons(),
                        C,
                        tap=C_tiles[c_index],
                        wait=True,
                        task_group=tg,
                    )
                    c_index += 1

                    for tile_row in range(current_tb_n_rows):
                        tile_offset = (
                            (row_base + tile_row) * n_shim_mem_A + col
                        ) % len(A_tiles)
                        if col < n_aie_rows:
                            rt.fill(
                                A_l3l2_fifos[col].prod(),
                                A,
                                tap=A_tiles[tile_offset],
                                task_group=tg,
                            )
                        rt.fill(
                            B_l3l2_fifos[col].prod(),
                            B,
                            tap=B_tiles[col],
                            task_group=tg,
                        )
                        A_taps.append(A_tiles[tile_offset])
                        B_taps.append(B_tiles[col])

                if tb > 0 or (tb == 0 and pingpong > 0):
                    rt.finish_task_group(tg)
                    tg = rt.task_group()
        rt.finish_task_group(tg)

    if generate_taps:
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )

    return Program(dev, rt).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def whole_array(
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
    b_col_maj: Compile[int] = 0,
    c_col_maj: Compile[int] = 0,
    emulate_bf16_mmul_with_bfp16: Compile[bool] = False,
    use_chess: Compile[bool] = False,
    scalar: Compile[bool] = False,
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


def generate_taps(
    M,
    K,
    N,
    m,
    k,
    n,
    n_aie_cols,
    dtype_in_str,
    dtype_out_str,
    b_col_maj=0,
    c_col_maj=0,
    emulate_bf16_mmul_with_bfp16=False,
    dev="npu",
):
    """Return ``(A_taps, B_taps, C_taps)`` for the visualization notebook."""
    dev_obj = _device_for(dev, n_aie_cols)
    set_current_device(dev_obj)
    return _build_design(
        dev_obj,
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
        use_chess=False,
        scalar=False,
        generate_taps=True,
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


def _numpy_reference(A_np, B_np, b_col_maj, dtype_out):
    B_logical = B_np.T if b_col_maj else B_np
    if np.issubdtype(A_np.dtype, np.integer):
        return (A_np.astype(np.int64) @ B_logical.astype(np.int64)).astype(dtype_out)
    return (A_np.astype(np.float32) @ B_logical.astype(np.float32)).astype(dtype_out)


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
    C_np = np.zeros((opts.M, opts.N), dtype=dtype_out)

    A_t = iron.tensor(A_np.reshape(-1), dtype=dtype_in, device="npu")
    B_t = iron.tensor(B_np.reshape(-1), dtype=dtype_in, device="npu")
    C_t = iron.tensor(C_np.reshape(-1), dtype=dtype_out, device="npu")

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

    expected_logical = _numpy_reference(A_np, B_np, opts.b_col_maj, dtype_out)
    if opts.c_col_maj:
        actual = C_t.numpy().reshape(opts.N, opts.M)
        expected = expected_logical.T
    else:
        actual = C_t.numpy().reshape(opts.M, opts.N)
        expected = expected_logical

    assert_close_with_benchmark(
        actual,
        expected,
        bench=bench,
        ops=2.0 * opts.M * opts.K * opts.N,
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
