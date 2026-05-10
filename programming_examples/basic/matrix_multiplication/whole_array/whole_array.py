#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Whole-array matrix multiply — Iron API design with ``@iron.jit`` compilation.

A 4×N_cols AIE array computes ``C = A @ B`` (optionally with ``B`` column-major
or ``C`` column-major).  Each compute tile owns one (m, n) output sub-tile and
streams (m, k) × (k, n) inputs through three layers of ObjectFifos.

This script has two modes:

* default — emits MLIR to stdout for the legacy ``aiecc`` + ``test.cpp``
  pipeline driven by ``makefile-common`` (used by every existing lit config
  and the matmul sweep).  This is the default so existing ``make run``
  invocations keep working unchanged.
* ``--jit`` — JIT-compiles the design via ``@iron.jit``, runs it on the
  attached NPU, then verifies against ``A @ B`` computed on the host.

The design body is shared between the two paths.
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import (
    Compile,
    In,
    Kernel,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, NPU1Col1, NPU1Col2, NPU2, Tile
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D
from aie.utils.benchmark import print_benchmark, run_iters

# bf16 BFP-16 emulation needs (8, 8, 8) on NPU2; kernels.mm doesn't expose the
# AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 toggle yet, so this single override
# stays local instead of going through kernels.mm(...).mac_dims.
_BF16_EMULATED_MAC_DIMS_NPU2 = (8, 8, 8)

# Per-(arch, dtype_in) (r, s, t) micro-kernel dims for the legacy --print-mlir
# path (which binds against the combined mm_MxKxN.o produced by makefile-common
# and so cannot ask kernels.mm.mac_dims, since that would query the active host
# device rather than the requested target dev_str).
_LEGACY_MAC_DIMS = {
    "npu": {"bf16": (4, 8, 4), "i8": (4, 8, 8), "i16": (4, 4, 4)},
    "npu2": {
        "bf16": {True: (8, 8, 8), False: (4, 8, 8)},
        "i8": (8, 8, 8),
        "i16": (4, 4, 8),
    },
}


def _legacy_mac_dims(dev_str, dtype_in_str, emulate_bf16_mmul_with_bfp16):
    entry = _LEGACY_MAC_DIMS[dev_str][dtype_in_str]
    if isinstance(entry, dict):
        return entry[emulate_bf16_mmul_with_bfp16]
    return entry


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
    emulate_bf16_mmul_with_bfp16,
    *,
    for_jit,
    generate_taps=False,
):
    """Build the whole-array matmul IRON design and resolve to MLIR.

    Shared by the JIT entry point and the ``--print-mlir`` path.  Two
    branch points based on ``for_jit``:

    * Kernel binding — JIT uses ``kernels.mm/mm_zero`` (ExternalFunction;
      the JIT pipeline compiles each from ``mm.cc`` on first call).  The
      legacy MLIR-emit path uses ``Kernel(name, "mm_MxKxN.o", ...)`` so the
      emitted ``link_with`` matches the combined object file built by
      ``makefile-common``'s existing rule (consumed by ``test.cpp`` + the
      matmul sweep).
    * L1 ObjectFifo element type — ``kernels.mm`` declares 1D arg types,
      so the JIT path uses 1D L1 types to match.  The legacy path keeps the
      original 2D ``(m, k)``-style L1 types so the emitted MLIR is
      byte-identical to what the pre-port iron variant produced.
    """
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

    # mac_dims: JIT path can ask kernels.mm.mac_dims (queries the active
    # device, which IS the JIT target).  Legacy path can't — dev is just a
    # static label and may not match the host — so use the local table.
    if for_jit:
        if (
            dev_str == "npu2"
            and dtype_in_str == "bf16"
            and emulate_bf16_mmul_with_bfp16
        ):
            r, s, t = _BF16_EMULATED_MAC_DIMS_NPU2
        else:
            mm_for_dims = kernels.mm(
                dim_m=m,
                dim_k=k,
                dim_n=n,
                input_dtype=dtype_in,
                output_dtype=dtype_out,
            )
            r, s, t = mm_for_dims.mac_dims
    else:
        r, s, t = _legacy_mac_dims(dev_str, dtype_in_str, emulate_bf16_mmul_with_bfp16)

    if dev_str == "npu" and n_aie_cols > 4:
        raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 4 columns")
    if dev_str == "npu2" and n_aie_cols > 8:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
        )

    # Tiling preconditions
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

    # When using more AIE columns than n_aie_rows (4) (NPU2 only), restrict
    # shim/mem tiles to n_aie_rows — there are only n_aie_rows row tiles for A.
    if n_aie_cols > n_aie_rows:
        n_shim_mem_A = n_aie_rows
    else:
        n_shim_mem_A = n_aie_cols

    # Integer division when n_aie_cols < 4, otherwise 1
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    if dev_str == "npu":
        if n_aie_cols == 1:
            dev_ty = NPU1Col1()
        elif n_aie_cols == 2:
            dev_ty = NPU1Col2()
        elif n_aie_cols == 4:
            dev_ty = NPU1()
    else:
        dev_ty = NPU2()

    A_taps = []
    B_taps = []
    C_taps = []

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    # L1 ObjectFifo element types are 2D in both modes — same as the pre-port
    # iron variant.  The JIT path's kernels.mm helper declares 1D arg types,
    # but BaseKernel.__call__ silently inserts a memref.collapse_shape to
    # bridge a contiguous N-D arg to a 1-D kernel signature, so designs can
    # keep their natural 2D L1 shape regardless of which mode is in use.
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    if for_jit:
        matmul_kernel = kernels.mm(
            dim_m=m,
            dim_k=k,
            dim_n=n,
            input_dtype=dtype_in,
            output_dtype=dtype_out,
            b_col_maj=bool(b_col_maj),
        )
        # Bind zero_* against the SAME .o that kernels.mm produces.  Calling
        # kernels.mm_zero() instead would compile mm.cc a second time with
        # the same -D{...}_ONLY flag, leaving both .o files exporting the
        # zero_* symbol and producing a duplicate-symbol link failure.
        zero_kernel = Kernel(
            f"zero_{dtype_out_str}",
            matmul_kernel.object_file_name,
            [C_l1_ty],
        )
    else:
        # Legacy: bind against the combined object file the Makefile builds.
        zero_kernel = Kernel(f"zero_{dtype_out_str}", f"mm_{m}x{k}x{n}.o", [C_l1_ty])
        matmul_kernel = Kernel(
            f"matmul_{dtype_in_str}_{dtype_out_str}",
            f"mm_{m}x{k}x{n}.o",
            [A_l1_ty, B_l1_ty, C_l1_ty],
        )

    tiles = [[(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    # ObjectFifos: L3↔L2 and L2↔L1 for A, B; L1↔L2↔L3 for C.
    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows
    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols
    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    # Input A
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
                # alternate columns in the full 4x8 NPU2 case
                tile=Tile(2 * i if n_aie_cols == 8 else i, 1),
            )
        )
        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = a_tmp_fifos[j]

    # Input B + Output C (per column)
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
                tile=Tile(col, 1),
            )
        )

        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)],
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
                tile=Tile(col, 1),
            )
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = c_tmp_fifos[j]

    # Per-core task: zero accumulator, then K/k iters of matmul-accumulate.
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

    workers = []
    for row in range(n_aie_rows):
        for col in range(n_aie_cols):
            tile_col, tile_row = core_tiles[row][col]
            workers.append(
                Worker(
                    core_fn,
                    [
                        A_l2l1_fifos[row].cons(),
                        B_l2l1_fifos[col].cons(),
                        C_l1l2_fifos[row][col].prod(),
                        zero_kernel,
                        matmul_kernel,
                    ],
                    tile=Tile(tile_col, tile_row),
                    stack_size=0xD00,
                )
            )

    # BD-budget split: at most 4 transfer-block rows per sync.
    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    # Tilers for A, B, C
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
        rt.start(*workers)

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
                        tile=Tile(col, 0),
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
                                tile=Tile(2 * col if n_aie_cols == 8 else col, 0),
                            )
                        rt.fill(
                            B_l3l2_fifos[col].prod(),
                            B,
                            tap=B_tiles[col],
                            task_group=tg,
                            tile=Tile(col, 0),
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

    return Program(dev_ty, rt).resolve_program()


@iron.jit
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
    emulate_bf16_mmul_with_bfp16: Compile[bool] = False,
):
    """JIT entry point: compiles the whole-array matmul on first call."""
    dev = iron.get_current_device()
    return _build_design(
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
        emulate_bf16_mmul_with_bfp16,
        for_jit=True,
    )


def _device_for_emit(dev_str, n_aie_cols):
    """Resolve the iron device object for the --print-mlir path."""
    if dev_str == "npu":
        if n_aie_cols == 1:
            return NPU1Col1()
        if n_aie_cols == 2:
            return NPU1Col2()
        if n_aie_cols == 4:
            return NPU1()
    return NPU2()


def _make_argparser():
    p = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication (Whole Array)",
        description=(
            "Default: emit MLIR for the legacy aiecc + test.cpp pipeline.  "
            "--jit instead JIT-compiles the design and runs it on the "
            "attached NPU, verifying the result against numpy."
        ),
    )
    p.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    p.add_argument("-M", type=int, default=512)
    p.add_argument("-K", type=int, default=512)
    p.add_argument("-N", type=int, default=512)
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=32)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    p.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=False)
    p.add_argument("--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16")
    p.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i16",
    )
    p.add_argument("--trace_size", type=int, default=0)
    p.add_argument(
        "--jit",
        action="store_true",
        help=(
            "JIT-compile the design and run on the attached NPU, verifying "
            "the result against numpy.  Without this flag, emit MLIR to "
            "stdout for the legacy aiecc + test.cpp pipeline (driven by "
            "makefile-common's mlir_target rule — the default so existing "
            "lit configs and the matmul sweep work unchanged)."
        ),
    )
    p.add_argument(
        "--generate-taps",
        action="store_true",
        help="Return TensorAccessPattern objects (used by the visualization notebook).",
    )
    p.add_argument("-w", "--warmup", type=int, default=2)
    p.add_argument("-i", "--iters", type=int, default=5)
    return p


def _numpy_reference(A_np, B_np, b_col_maj, dtype_out):
    """Compute the host-side reference C = A @ B (or A @ B.T if b_col_maj)."""
    B_logical = B_np.T if b_col_maj else B_np
    # Promote ints to int64 for accumulation, then cast back; bf16/f32 stay float.
    if np.issubdtype(A_np.dtype, np.integer):
        ref = (A_np.astype(np.int64) @ B_logical.astype(np.int64)).astype(dtype_out)
    else:
        ref = (A_np.astype(np.float32) @ B_logical.astype(np.float32)).astype(dtype_out)
    return ref


def _validate_shape_args(opts):
    """Surface common config errors as clean messages instead of stack traces.

    The design + TensorTiler2D have a number of divisibility preconditions
    that show up as raw AssertionError / ValueError otherwise.
    """
    n_aie_rows = 4
    if opts.M % (opts.m * n_aie_rows) != 0:
        sys.exit(
            f"-M {opts.M} must be a multiple of -m × n_aie_rows ({opts.m} × {n_aie_rows} = {opts.m * n_aie_rows})"
        )
    if opts.K % opts.k != 0:
        sys.exit(f"-K {opts.K} must be a multiple of -k {opts.k}")
    if opts.N % (opts.n * opts.n_aie_cols) != 0:
        sys.exit(
            f"-N {opts.N} must be a multiple of -n × --n-aie-cols ({opts.n} × {opts.n_aie_cols} = {opts.n * opts.n_aie_cols})"
        )
    # tb_max_n_rows = 4 (no c_col_maj yet); tb_n_rows = 2.  The C tiler
    # demands M//m//n_aie_rows be divisible by tb_n_rows.
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


def main():
    opts = _make_argparser().parse_args()
    _validate_shape_args(opts)

    if opts.generate_taps:
        # Pure design-introspection mode (no run, no print).
        return _build_design(
            _device_for_emit(opts.dev, opts.n_aie_cols),
            opts.M,
            opts.K,
            opts.N,
            opts.m,
            opts.k,
            opts.n,
            opts.n_aie_cols,
            opts.dtype_in,
            opts.dtype_out,
            opts.b_col_maj,
            opts.emulate_bf16_mmul_with_bfp16,
            for_jit=False,
            generate_taps=True,
        )

    if not opts.jit:
        # Default: legacy MLIR-emit path used by makefile-common's mlir_target
        # rule (consumed by aiecc + test.cpp).  Default so the existing 43
        # lit configs and the matmul sweep keep working unchanged.
        print(
            _build_design(
                _device_for_emit(opts.dev, opts.n_aie_cols),
                opts.M,
                opts.K,
                opts.N,
                opts.m,
                opts.k,
                opts.n,
                opts.n_aie_cols,
                opts.dtype_in,
                opts.dtype_out,
                opts.b_col_maj,
                opts.emulate_bf16_mmul_with_bfp16,
                for_jit=False,
            )
        )
        return

    # --jit: JIT host-run path: build inputs, run, verify.
    dtype_in = str_to_dtype(opts.dtype_in)
    dtype_out = str_to_dtype(opts.dtype_out)

    rng = np.random.default_rng(1726250518)  # match the C++ harness seed
    if np.issubdtype(dtype_in, np.integer):
        info = np.iinfo(dtype_in)
        A_np = rng.integers(
            info.min // 4, info.max // 4, size=(opts.M, opts.K), dtype=dtype_in
        )
        B_shape = (opts.N, opts.K) if opts.b_col_maj else (opts.K, opts.N)
        B_np = rng.integers(info.min // 4, info.max // 4, size=B_shape, dtype=dtype_in)
    else:
        A_np = rng.standard_normal((opts.M, opts.K)).astype(dtype_in)
        B_shape = (opts.N, opts.K) if opts.b_col_maj else (opts.K, opts.N)
        B_np = rng.standard_normal(B_shape).astype(dtype_in)
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
        emulate_bf16_mmul_with_bfp16=opts.emulate_bf16_mmul_with_bfp16,
        warmup=opts.warmup,
        iters=opts.iters,
    )

    actual = C_t.numpy().reshape(opts.M, opts.N)
    expected = _numpy_reference(A_np, B_np, opts.b_col_maj, dtype_out)

    if np.issubdtype(dtype_out, np.integer):
        ok = np.array_equal(actual, expected)
    else:
        ok = np.allclose(actual, expected, rtol=1e-2, atol=1e-2)

    if not ok:
        diffs = (
            np.argwhere(actual != expected)[:5]
            if np.issubdtype(dtype_out, np.integer)
            else None
        )
        sys.exit(f"FAIL! output does not match A @ B (first mismatches: {diffs})")

    print()
    print_benchmark(bench)
    macs = 2.0 * opts.M * opts.K * opts.N
    if bench.npu is not None:
        gflops = macs / (1000 * bench.npu.avg_us)
        print(f"NPU GFLOPS                    : {gflops:.2f}")
    print("PASS!")


if __name__ == "__main__":
    main()
