# test_verify.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# RUN: %pytest %s
"""Unit tests for aie.utils.verify (no NPU required)."""

import numpy as np
import pytest

from aie.utils.verify import count_mismatches, nearly_equal

# ---------------------------------------------------------------------------
# nearly_equal
# ---------------------------------------------------------------------------


def test_identical_inputs_are_nearly_equal():
    assert nearly_equal([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]).all()


def test_within_default_rtol():
    # 10% diff → within 12.8% default rtol → True
    assert nearly_equal(1.0, 1.1).item()


def test_outside_default_rtol():
    # 100% diff → outside 12.8% default rtol → False
    assert not nearly_equal(1.0, 2.0).item()


def test_nan_compares_false():
    assert not nearly_equal(np.nan, np.nan).item()
    assert not nearly_equal(np.nan, 0.0).item()


def test_inf_equal_inf_passes():
    # IEEE: inf == inf is True; matches C++ test_utils::nearly_equal short-circuit.
    assert nearly_equal(np.inf, np.inf).item()


def test_inf_vs_finite_compares_false():
    assert not nearly_equal(np.inf, 1.0).item()


def test_custom_rtol():
    # Default would reject 100% diff; rtol=2.0 accepts up to ~200% (relative norm)
    assert nearly_equal(1.0, 2.0, rtol=2.0).item()


def test_custom_atol_floor_passes_near_zero():
    # 0.05 absolute diff: relative tol on |0|+|0.05| = 0.0064 → fails by rtol;
    # but atol=0.1 puts the floor above the diff → passes.
    assert nearly_equal(0.0, 0.05, atol=0.1).item()


def test_returns_ndarray_of_broadcast_shape():
    out = nearly_equal([1.0, 2.0, 3.0], 1.5)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)


# ---------------------------------------------------------------------------
# count_mismatches
# ---------------------------------------------------------------------------


def test_no_mismatches():
    e, n = count_mismatches([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert (e, n) == (0, 3)


def test_counts_violations():
    e, n = count_mismatches([1.0, 100.0], [1.0, 1.0])
    assert (e, n) == (1, 2)


def test_stops_at_first_nan_in_ref():
    e, n = count_mismatches([1.0, 2.0, 999.0, 999.0], [1.0, 2.0, np.nan, 0.0])
    assert (e, n) == (0, 2)


def test_stops_at_first_nan_in_actual():
    e, n = count_mismatches([1.0, 2.0, np.nan, 999.0], [1.0, 2.0, 3.0, 0.0])
    assert (e, n) == (0, 2)


def test_stops_at_first_inf():
    e, n = count_mismatches([1.0, np.inf, 999.0], [1.0, 2.0, 3.0])
    assert (e, n) == (0, 1)


def test_stop_at_nonfinite_disabled_counts_all():
    e, n = count_mismatches(
        [1.0, np.nan, 999.0], [1.0, 2.0, 3.0], stop_at_nonfinite=False
    )
    # Sample 0 ok, sample 1 NaN → not nearly_equal → error, sample 2 → error
    assert (e, n) == (2, 3)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        count_mismatches([1.0, 2.0], [1.0, 2.0, 3.0])


def test_works_on_2d_arrays_via_ravel():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    e, n = count_mismatches(a, a)
    assert (e, n) == (0, 4)
