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

import numpy as np

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
