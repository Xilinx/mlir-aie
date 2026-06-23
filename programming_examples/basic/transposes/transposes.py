# transposes/transposes.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Four transpose strategies on a single AIE column, ``@iron.jit``-compiled.

All four strategies produce the same end result: a full ``M x N`` →
``N x M`` transpose.  Only the on-device mechanism differs, and each
mechanism has its own (dtype, size) support envelope:

  * ``dma``         — pure shim-DMA stride; no compute core.
                      ``ObjectFifo.forward()`` with a transpose TAP at
                      the input.  Shim DMA stride-1 must be ≥ 4 bytes,
                      so the element type must be 4 bytes
                      (``int32`` / ``uint32``).
  * ``dma_packet``  — same body as ``dma`` but lowered with
                      ``--packet-sw-objFifos`` so the data flows packet-
                      switched.  Same 4-byte-element constraint.
  * ``shuffle``     — per-tile VSHUFFLE on a compute core.  The kernel
                      (``aie_kernels/shuffle_16x16.cc``) is hand-written
                      for ``uint8`` ``16 x 16``; constrained to that
                      exact ``(dtype, M, K) = (uint8, 16, 16)``.
  * ``combined``    — hybrid: shim DMA does the outer-block reshuffle
                      (L3→L2→L1 TAP chain), and a VSHUFFLE kernel
                      (``transpose_4x4`` or ``transpose_8x8`` in
                      ``aie_kernels/transpose.cc``) transposes each
                      inner ``s x s`` sub-tile.  Supports
                      ``i8`` / ``i16`` / ``i32`` and any sizes with
                      ``m | M``, ``n | N``, ``s | m``, ``s | n``.

Each strategy raises ``ValueError`` if asked for a combo outside its
support envelope.

Standalone:    ``python3 transposes.py --strategy <s> [--M ... --K ... ...]``
Compile-only:  ``... --xclbin-path=PATH --insts-path=PATH``    (Makefile)
"""

import argparse
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import AnyComputeTile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

_KERNELS_DIR = Path(__file__).parent / "aie_kernels"
_SHUFFLE_SRC = str(_KERNELS_DIR / "shuffle_16x16.cc")
_COMBINED_SRC = str(_KERNELS_DIR / "transpose.cc")

_BYTES_TO_DTYPE = {1: np.uint8, 2: np.uint16, 4: np.uint32}
_COMBINED_DTYPE_MACRO = {1: "DTYPE_i8", 2: "DTYPE_i16", 4: "DTYPE_i32"}


# ---------------------------------------------------------------------------
# Strategy 1 / 2: pure shim-DMA stride.  4-byte element only.
# ---------------------------------------------------------------------------


@iron.jit
def _transpose_dma(
    A: In,
    C: Out,
    *,
    M: CompileTime[int] = 64,
    K: CompileTime[int] = 64,
    dtype_bytes: CompileTime[int] = 4,
):
    if dtype_bytes != 4:
        raise ValueError(
            f"--strategy=dma requires 4-byte elements (shim DMA stride-1 "
            f"must be ≥ 4 bytes); got dtype_bytes={dtype_bytes}."
        )
    dtype = _BYTES_TO_DTYPE[dtype_bytes]
    tensor_ty = np.ndarray[(M, K), np.dtype[dtype]]
    tap_in = TensorTiler2D.simple_tiler((M, K), tile_col_major=True)[0]
    of_in = ObjectFifo(tensor_ty)
    of_out = of_in.cons().forward(AnyComputeTile)
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a, c):
        rt.fill(of_in.prod(), a, tap_in)
        rt.drain(of_out.cons(), c, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit(aiecc_flags=["--packet-sw-objFifos"])
def _transpose_dma_packet(
    A: In,
    C: Out,
    *,
    M: CompileTime[int] = 64,
    K: CompileTime[int] = 64,
    dtype_bytes: CompileTime[int] = 4,
):
    if dtype_bytes != 4:
        raise ValueError(
            f"--strategy=dma_packet requires 4-byte elements (shim DMA "
            f"stride-1 must be ≥ 4 bytes); got dtype_bytes={dtype_bytes}."
        )
    dtype = _BYTES_TO_DTYPE[dtype_bytes]
    tensor_ty = np.ndarray[(M, K), np.dtype[dtype]]
    tap_in = TensorTiler2D.simple_tiler((M, K), tile_col_major=True)[0]
    of_in = ObjectFifo(tensor_ty, name="in")
    of_out = of_in.cons().forward()
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a, c):
        rt.fill(of_in.prod(), a, tap_in)
        rt.drain(of_out.cons(), c, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


# ---------------------------------------------------------------------------
# Strategy 3: VSHUFFLE-only, hand-coded uint8 16x16 kernel.
# ---------------------------------------------------------------------------


@iron.jit
def _transpose_shuffle(
    A: In,
    C: Out,
    *,
    M: CompileTime[int] = 16,
    K: CompileTime[int] = 16,
    dtype_bytes: CompileTime[int] = 1,
):
    if (dtype_bytes, M, K) != (1, 16, 16):
        raise ValueError(
            f"--strategy=shuffle is limited to (dtype_bytes, M, K) = (1, 16, 16) "
            f"because shuffle_16x16.cc is hand-coded for uint8 16x16; got "
            f"(dtype_bytes={dtype_bytes}, M={M}, K={K})."
        )
    tile_ty = np.ndarray[(M * K,), np.dtype[np.uint8]]

    kernel_func = ExternalFunction(
        "transpose_16x16",
        source_file=_SHUFFLE_SRC,
        arg_types=[tile_ty, tile_ty],
    )

    in_fifo = ObjectFifo(tile_ty, name="in_fifo")
    out_fifo = ObjectFifo(tile_ty, name="out_fifo")

    def core_fn(in_fifo, out_fifo, kernel_func):
        elem_in = in_fifo.acquire(1)
        elem_out = out_fifo.acquire(1)
        kernel_func(elem_in, elem_out)
        out_fifo.release(1)
        in_fifo.release(1)

    worker = Worker(core_fn, fn_args=[in_fifo.cons(), out_fifo.prod(), kernel_func])

    rt = Runtime()
    with rt.sequence(tile_ty, tile_ty) as (a, c):
        rt.start(worker)
        rt.fill(in_fifo.prod(), a)
        rt.drain(out_fifo.cons(), c, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


# ---------------------------------------------------------------------------
# Strategy 4: hybrid shim-DMA outer + VSHUFFLE inner.
# ---------------------------------------------------------------------------


@iron.jit
def _transpose_combined(
    A: In,
    C: Out,
    *,
    M: CompileTime[int] = 64,
    K: CompileTime[int] = 64,
    m: CompileTime[int] = 16,
    n: CompileTime[int] = 16,
    s: CompileTime[int] = 8,
    dtype_bytes: CompileTime[int] = 4,
):
    if dtype_bytes not in (1, 2, 4):
        raise ValueError(
            f"--strategy=combined supports dtype_bytes in {{1, 2, 4}}; "
            f"got {dtype_bytes}."
        )
    if s not in (4, 8):
        raise ValueError(f"--strategy=combined requires s in {{4, 8}}; got {s}.")
    if M % m or K % n or m % s or n % s:
        raise ValueError(
            f"combined requires m | M, n | K, s | m, s | n; got "
            f"M={M}, K={K}, m={m}, n={n}, s={s}."
        )
    if s == 8 and (m < 32 or n < 32):
        # The s=8 kernel uses an interleave_unzip(32) stage internally, so
        # it needs min(m, n) >= 32 — m,n > 8 compiles but silently produces
        # wrong output.  The matching static_assert lives in
        # aie_kernels/transpose.cc:transpose_8x8.
        raise ValueError("s=8 requires m, n >= 32 (kernel constraint)")

    dtype = _BYTES_TO_DTYPE[dtype_bytes]
    matrix_ty = np.ndarray[(M, K), np.dtype[dtype]]
    tile_ty = np.ndarray[(m, n), np.dtype[dtype]]

    kernel_func = ExternalFunction(
        f"transpose_{s}x{s}",
        source_file=_COMBINED_SRC,
        arg_types=[tile_ty, tile_ty],
        compile_flags=[
            f"-DDIM_m={m}",
            f"-DDIM_n={n}",
            f"-D{_COMBINED_DTYPE_MACRO[dtype_bytes]}",
        ],
    )

    tap_in_L3L2 = TensorAccessPattern(
        tensor_dims=(M, K),
        offset=0,
        sizes=[M // m, K // n, m, n],
        strides=[m * K, n, K, 1],
    )
    tap_in_L2L1 = TensorAccessPattern(
        tensor_dims=(M, K),
        offset=0,
        sizes=[m // s, s, n // s, s],
        strides=[s, m, s * m, 1],
    )
    tap_out_L1L3 = TensorAccessPattern(
        tensor_dims=(K, M),
        offset=0,
        sizes=[M // m, K // n, n, m],
        strides=[m, n * M, M, 1],
    )

    in_L3L2_fifo = ObjectFifo(tile_ty, name="in_L3L2_fifo")
    in_L2L1_fifo = in_L3L2_fifo.cons(
        dims_from_stream=tap_in_L2L1.transformation_dims
    ).forward(obj_type=tile_ty, name="in_L2L1_fifo")
    out_fifo = ObjectFifo(tile_ty, name="out_fifo")

    def core_fn(in_fifo, out_fifo, kernel_func):
        for _ in range_(K // n):
            for _ in range_(M // m):
                elem_in = in_fifo.acquire(1)
                elem_out = out_fifo.acquire(1)
                kernel_func(elem_in, elem_out)
                out_fifo.release(1)
                in_fifo.release(1)

    worker = Worker(
        core_fn, fn_args=[in_L2L1_fifo.cons(), out_fifo.prod(), kernel_func]
    )

    rt = Runtime()
    with rt.sequence(matrix_ty, matrix_ty) as (a, c):
        rt.start(worker)
        rt.fill(in_L3L2_fifo.prod(), a, tap_in_L3L2)
        rt.drain(out_fifo.cons(), c, tap_out_L1L3, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------


_STRATEGIES = {
    "dma": _transpose_dma,
    "dma_packet": _transpose_dma_packet,
    "shuffle": _transpose_shuffle,
    "combined": _transpose_combined,
}


# Per-strategy defaults — picked so `python3 transposes.py -s <s>` works
# out of the box without the caller having to know each strategy's support
# envelope.
_DEFAULTS = {
    "dma": dict(M=64, K=64, dtype_bytes=4),
    "dma_packet": dict(M=64, K=32, dtype_bytes=4),
    "shuffle": dict(M=16, K=16, dtype_bytes=1),
    # combined: m=n=32 / s=8 is the smallest empirically-working combo.
    # m=n=16 / s=8 *compiles* (passes the m>s, n>s assert) but the
    # underlying transpose_8x8 kernel's VECTOR_SIZE math breaks when the
    # outer tile equals the sub-tile size, producing the wrong block
    # interleave.  Original combined_transpose Makefile shipped m=64,n=32.
    "combined": dict(M=128, K=128, dtype_bytes=4, m=32, n=32, s=8),
}


def _compile_kwargs(opts):
    base = dict(M=opts.M, K=opts.K, dtype_bytes=opts.dtype_bytes)
    if opts.strategy == "combined":
        base.update(m=opts.m, n=opts.n, s=opts.s)
    return base


def _apply_defaults(opts):
    d = _DEFAULTS[opts.strategy]
    if opts.M is None:
        opts.M = d["M"]
    if opts.K is None:
        opts.K = d["K"]
    if opts.dtype_bytes is None:
        opts.dtype_bytes = d["dtype_bytes"]
    # combined has m/n/s too; pull them from the per-strategy defaults
    # only when the user didn't override on the CLI.  The argparse
    # defaults (-m 16 -n 16 --ss 8) are generic placeholders that don't
    # always satisfy the combined kernel's empirical size constraint, so
    # use the strategy-specific defaults when present.
    if opts.strategy == "combined":
        sentinel_defaults = dict(m=16, n=16, s=8)
        for k in ("m", "n", "s"):
            if getattr(opts, k) == sentinel_defaults[k] and k in d:
                setattr(opts, k, d[k])


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Transpose (four strategies)")
    add_compile_args(p)
    p.add_argument(
        "-s",
        "--strategy",
        choices=list(_STRATEGIES.keys()),
        required=True,
        help="which on-device transpose mechanism to use",
    )
    p.add_argument("-M", type=int, default=None, help="input rows")
    p.add_argument("-K", type=int, default=None, help="input cols (== N for output)")
    p.add_argument(
        "--dtype-bytes",
        type=int,
        choices=[1, 2, 4],
        default=None,
        help="element size in bytes; per-strategy support varies (see module docstring)",
    )
    p.add_argument("-m", type=int, default=16, help="outer tile rows (combined)")
    p.add_argument("-n", type=int, default=16, help="outer tile cols (combined)")
    p.add_argument(
        "--ss", dest="s", type=int, default=8, help="inner shuffle size (combined)"
    )
    return p


def _run_and_verify(opts):
    M, K = opts.M, opts.K
    dtype = _BYTES_TO_DTYPE[opts.dtype_bytes]
    rng = np.random.default_rng(0)
    info = np.iinfo(dtype)
    in_np = rng.integers(info.min, info.max + 1, size=(M, K), dtype=dtype)
    a_t = iron.tensor(in_np, dtype=dtype, device="npu")
    c_t = iron.zeros((K, M), dtype=dtype, device="npu")

    _STRATEGIES[opts.strategy](a_t, c_t, **_compile_kwargs(opts))

    expected = in_np.T
    actual = c_t.numpy().reshape(K, M)
    assert_pass(actual, expected, fail_msg="output does not match transpose(in)")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        _STRATEGIES[opts.strategy],
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        validate=_apply_defaults,
    )


if __name__ == "__main__":
    main()
