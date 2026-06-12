# verify.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Tolerance-based output verification helpers for examples and tests.

Mirrors the canonical ``test_utils::nearly_equal`` semantics used across the
C++ testbenches so Python migrations of those examples behave identically:

    |a - b|  <  max(atol, rtol * (|a| + |b|))

Defaults match the C++ default of ``rtol=0.128``, which is the widely-used
relative tolerance for bfloat16 / LUT-approximated kernels (exp, softmax,
gelu, silu, swiglu, ...).
"""

from __future__ import annotations

import sys

import numpy as np

from aie.utils.benchmark import print_benchmark

_DEFAULT_RTOL = 0.128


def nearly_equal(
    a, b, *, rtol: float = _DEFAULT_RTOL, atol: float | None = None
) -> np.ndarray:
    """Element-wise nearly-equal comparison.

    Returns a boolean ndarray of the broadcast shape; ``True`` where
    ``|a - b| < max(atol, rtol * (|a| + |b|))``.  Inputs are coerced to
    ``float32`` (sufficient headroom for bfloat16 work).  NaN inputs
    produce ``False`` (matching IEEE and the C++ semantics).

    Args:
        a, b: Array-likes to compare.
        rtol: Relative tolerance (default 0.128 — matches C++ test_utils).
        atol: Absolute floor.  Defaults to ``np.finfo(np.float32).tiny``.
    """
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    if atol is None:
        atol = float(np.finfo(np.float32).tiny)
    with np.errstate(over="ignore", invalid="ignore"):
        diff = np.abs(a32 - b32)
        norm = np.minimum(np.abs(a32) + np.abs(b32), np.finfo(np.float32).max)
        thresh = np.maximum(atol, rtol * norm)
    return (a32 == b32) | (diff < thresh)


def count_mismatches(
    actual,
    ref,
    *,
    rtol: float = _DEFAULT_RTOL,
    atol: float | None = None,
    stop_at_nonfinite: bool = True,
) -> tuple[int, int]:
    """Count tolerance violations between ``actual`` and ``ref``.

    Returns ``(errors, n_checked)`` where ``n_checked`` is the number of
    samples that were actually compared (less than ``len(ref)`` when
    ``stop_at_nonfinite`` halts on the first inf/nan from either side).

    With ``stop_at_nonfinite=True`` (default), this matches the canonical
    C++ verify pattern that ``break``\\s on the first inf/nan rather than
    treating the LUT's behaviour outside its defined input range as part
    of the contract.
    """
    a32 = np.asarray(actual, dtype=np.float32).ravel()
    r32 = np.asarray(ref, dtype=np.float32).ravel()
    if a32.shape != r32.shape:
        raise ValueError(
            f"actual and ref must have the same shape, got {a32.shape} vs {r32.shape}"
        )
    if stop_at_nonfinite:
        bad = ~(np.isfinite(a32) & np.isfinite(r32))
        stop = int(np.argmax(bad)) if bad.any() else len(a32)
    else:
        stop = len(a32)
    ok = nearly_equal(a32[:stop], r32[:stop], rtol=rtol, atol=atol)
    return int(np.size(ok) - np.count_nonzero(ok)), stop


def assert_pass(
    actual,
    expected,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    fail_msg: str | None = None,
    print_pass: bool = True,
) -> None:
    """Verify ``actual`` matches ``expected``; print ``PASS!`` on success.

    Args:
        actual, expected: Array-likes (numpy arrays, scalars, lists).
        rtol, atol: Tolerance bounds for the bf16/LUT-style comparator
            (see :func:`count_mismatches`).  When both are ``None``
            (the default), use ``np.array_equal`` for an exact compare
            — the right choice for integer and bit-exact pipelines.
            Pass ``rtol=`` (and/or ``atol=``) to opt into the
            tolerance comparator.
        fail_msg: Optional context appended to the ``FAIL!`` line that
            ``sys.exit()`` raises on mismatch.
        print_pass: When ``True`` (default), print ``PASS!`` on success.
            Set to ``False`` to do the verify check but defer the
            success banner — useful when you want to print benchmark
            stats first and then the ``PASS!`` line.

    Raises:
        SystemExit: On mismatch (via ``sys.exit``) — exits with the
            ``"FAIL!"`` message as the status string.
    """
    if rtol is None and atol is None:
        ok = bool(np.array_equal(actual, expected))
    else:
        errors, _ = count_mismatches(
            actual,
            expected,
            rtol=rtol if rtol is not None else _DEFAULT_RTOL,
            atol=atol,
        )
        ok = errors == 0
    if not ok:
        sys.exit("FAIL!" if fail_msg is None else f"FAIL! {fail_msg}")
    if print_pass:
        print("PASS!")


def assert_close_with_benchmark(
    actual,
    expected,
    *,
    bench,
    ops: float | None = None,
    gflops_fmt: str = ".2f",
    float_rtol: float = 0.05,
    float_atol: float = 0.5,
    fail_msg: str | None = None,
    mismatch_indices: bool = False,
) -> None:
    """Verify, print benchmark stats, optionally print GFLOPS, then ``PASS!``.

    Wraps the standard matmul/vector_scalar_mul tail in one call.  Picks
    the comparator based on ``expected``'s dtype: integer dtypes use the
    exact compare (``np.array_equal``), float dtypes use the tolerance
    compare with ``rtol=float_rtol`` / ``atol=float_atol``.

    Args:
        actual, expected: Array-likes; ``expected.dtype`` selects the
            comparator branch.
        bench: A :class:`~aie.utils.benchmark.BenchmarkResult` (typically
            from :func:`~aie.utils.benchmark.run_iters`).
        ops: Total scalar ops for the kernel (e.g. ``2 * M * K * N`` for
            matmul, ``2 * M * K`` for matvec).  When set and
            ``bench.npu`` is available, prints ``NPU GFLOPS`` using
            ``ops / (1000 * avg_us)``.
        gflops_fmt: Format spec for the GFLOPS number (default ``".2f"``;
            matrix_vector uses ``".4f"`` for finer resolution at low
            GFLOPS).
        float_rtol, float_atol: Tolerance bounds for the float branch.
            Defaults match the C++ matmul harness's get_*_tol.
        fail_msg: Optional context appended to the ``FAIL!`` line on
            mismatch.
        mismatch_indices: When True (and the integer branch detects a
            mismatch), append the first five mismatch ``np.argwhere``
            indices to the ``FAIL!`` line — useful for matmul-style
            debugging.  No-op for the float branch.

    Raises:
        SystemExit: On mismatch (via :func:`assert_pass`).
    """
    if np.issubdtype(np.asarray(expected).dtype, np.integer):
        if mismatch_indices and not bool(np.array_equal(actual, expected)):
            diffs = np.argwhere(np.asarray(actual) != np.asarray(expected))[:5]
            base = "output mismatch" if fail_msg is None else fail_msg
            sys.exit(f"FAIL! {base} (first mismatches: {diffs.tolist()})")
        assert_pass(actual, expected, fail_msg=fail_msg, print_pass=False)
    else:
        assert_pass(
            actual,
            expected,
            rtol=float_rtol,
            atol=float_atol,
            fail_msg=fail_msg,
            print_pass=False,
        )

    print()
    print_benchmark(bench)
    if ops is not None and bench.npu is not None:
        gflops = ops / (1000 * bench.npu.avg_us)
        print(f"NPU GFLOPS                    : {gflops:{gflops_fmt}}")
    print("PASS!")
