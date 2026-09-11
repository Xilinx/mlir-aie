# test_verify.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

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


# ---------------------------------------------------------------------------
# Tolerance / compare: the kernel-owned contract
# ---------------------------------------------------------------------------

from aie.utils.verify import Tolerance, bf16_ulp_distance, compare  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

_REF = np.array([1.0, 2.5, -3.0, 1e-3, np.nan, np.inf, -np.inf, 0.0], np.float32)


def _bump(a, idx, n=1):
    """Move element ``idx`` of a bf16 array ``n`` bit patterns away from zero."""
    b = a.copy()
    b.view(np.uint16)[idx] += n
    return b


def test_tolerance_kind_is_derived_from_the_fields():
    assert Tolerance.exact().kind == "exact"
    assert Tolerance.bf16_ulps(2).kind == "ulps"
    assert Tolerance.relative(0.01).kind == "relative"
    assert Tolerance(atol=0.5).kind == "relative"


def test_default_tolerance_is_exact_for_ints_and_canonical_rtol_for_floats():
    assert Tolerance.default_for(np.int32).kind == "exact"
    assert Tolerance.default_for(np.bool_).kind == "exact"
    f = Tolerance.default_for(bfloat16)
    assert f.kind == "relative" and f.rtol == 0.128


def test_compare_defaults_to_the_output_dtype_contract():
    ints = np.array([1, 2, 3], np.int32)
    assert compare(ints, ints).ok
    assert not compare(ints, ints + 1).ok
    x = np.array([1.0, 2.0], np.float32)
    assert compare((x * 1.1).astype(bfloat16), x).ok  # 10% inside rtol=0.128


def test_bf16_exact_roundtrip_passes():
    assert compare(_REF.astype(bfloat16), _REF, Tolerance.bf16_ulps(0)).ok


def test_bf16_one_ulp_boundary():
    a = _REF.astype(bfloat16)
    assert compare(_bump(a, 1, 1), _REF, Tolerance.bf16_ulps(1)).ok
    assert not compare(_bump(a, 1, 1), _REF, Tolerance.bf16_ulps(0)).ok
    v = compare(_bump(a, 1, 2), _REF, Tolerance.bf16_ulps(1))
    assert not v.ok and v.max_ulp_err == 2 and v.first_bad_index == 1


def test_bf16_negative_ulp_direction():
    a = _REF.astype(bfloat16)
    # index 2 is -3.0; +1 in bits moves away from zero, still 1 ULP.
    assert compare(_bump(a, 2, 1), _REF, Tolerance.bf16_ulps(1)).ok
    # Signed zeros are the same point on the ULP scale.
    zeros = (np.array([-0.0], bfloat16), np.array([0.0], bfloat16))
    assert bf16_ulp_distance(*zeros)[0] == 0


def test_ulps_tolerance_rejects_non_bf16_output():
    with pytest.raises(ValueError, match="bfloat16"):
        compare(np.zeros(2, np.float32), np.zeros(2), Tolerance.bf16_ulps(1))


@pytest.mark.parametrize(
    "tol", [Tolerance.exact(), Tolerance.bf16_ulps(64), Tolerance.relative(10.0, 1e9)]
)
def test_nonfinite_must_match_under_every_kind(tol):
    a = _REF.astype(bfloat16)
    dropped_nan = a.copy()
    dropped_nan[4] = 0
    assert not compare(dropped_nan, _REF, tol).ok, "NaN silently dropped"
    wrong_inf_sign = a.copy()
    wrong_inf_sign[5] = -np.inf
    assert not compare(wrong_inf_sign, _REF, tol).ok, "inf sign ignored"
    assert compare(a, _REF, tol).ok, "matching non-finite values must pass"


def test_nonfinite_mismatch_ignores_max_mismatch_frac():
    # A generous budget is for how close finite values came, not for whether
    # NaN/Inf were reproduced at all -- a single non-finite mismatch must fail
    # a verdict even when the budget alone would forgive it.
    r = _REF.astype(bfloat16).copy()
    r[4] = np.nan
    a = r.copy()
    a[4] = 0
    assert not compare(a, r, Tolerance.bf16_ulps(1, max_mismatch_frac=1.0)).ok


def test_exact_kind_casts_the_reference_to_the_output_dtype():
    # relu-style: reference computed in f32 but the kernel emits bf16.
    ref = np.array([1.00390625, 2.0], np.float32)  # 1 + 2^-8, not bf16-representable
    assert compare(ref.astype(bfloat16), ref, Tolerance.exact()).ok


def test_relative_uses_the_canonical_nearly_equal_formula():
    r = np.array([1.0, 1000.0], np.float32)
    a = r * np.float32(1 + 5e-6)
    assert compare(a, r, Tolerance.relative(rtol=1e-5)).ok
    assert not compare(a, r, Tolerance.relative(rtol=1e-6, atol=0.0)).ok


def test_mismatch_budget():
    r = np.zeros(10_000, np.float32)
    a = r.astype(bfloat16)
    a[0] = 1.0
    assert not compare(a, r, Tolerance.bf16_ulps(1)).ok
    assert compare(a, r, Tolerance.bf16_ulps(1, max_mismatch_frac=1e-4)).ok


def test_shape_mismatch_fails_with_detail():
    v = compare(np.zeros(3, np.int32), np.zeros(4, np.int64), Tolerance.exact())
    assert not v.ok and "shape mismatch" in v.detail


def test_verdict_detail_names_the_evidence():
    tol = Tolerance.relative(0.01, note="measured on npu2, 2026-09")
    v = compare(np.array([2.0], np.float32), np.array([1.0], np.float32), tol)
    assert not v.ok and "measured on npu2" in v.detail


def test_integers_honour_an_lsb_slack_only_under_a_relative_tolerance():
    ref = np.array([10, 20, 30, 255], dtype=np.uint8)
    got = np.array([11, 19, 30, 254], dtype=np.uint8)
    assert not compare(got, ref, Tolerance.exact())
    v = compare(got, ref, Tolerance.lsb(1))
    assert v and v.max_abs_err == 1
    assert not compare(got, ref, Tolerance.lsb(0))
    assert Tolerance.lsb(2).kind == "relative"
    # The slack is absolute: a wide rtol does not creep in for integers.
    assert not compare(
        np.array([200], np.uint8), np.array([100], np.uint8), Tolerance.lsb(1)
    )


def test_integer_overflow_of_the_reference_follows_the_declared_semantics():
    ref = np.array([100, 40000, -40000, 7], dtype=np.int64)
    # wrap: the plain cast, two's complement.
    wrapped = ref.astype(np.int16)
    assert compare(wrapped, ref, Tolerance.exact(), overflow="wrap")
    # saturate: the kernel clamps.
    sat = np.array([100, 32767, -32768, 7], dtype=np.int16)
    assert compare(sat, ref, Tolerance.exact(), overflow="saturate")
    assert not compare(wrapped, ref, Tolerance.exact(), overflow="saturate")
    # undefined: an overflowing reference is not graded at all.
    v = compare(sat, ref, Tolerance.exact(), overflow="undefined")
    assert not v and v.n_mismatch == 2 and "overflows int16" in v.detail
    # ... but a reference that fits is graded normally.
    fits = np.array([1, 2, 3, 4], dtype=np.int64)
    assert compare(fits.astype(np.int16), fits, Tolerance.exact(), overflow="undefined")
    with pytest.raises(ValueError, match="overflow must be"):
        compare(sat, ref, Tolerance.exact(), overflow="clamp")


def test_default_tolerance_is_dtype_aware():
    assert Tolerance.default_for(np.float32).rtol == 1e-4
    assert Tolerance.default_for(np.float16).rtol == 1e-2
    assert Tolerance.default_for(bfloat16).rtol == 0.128
    assert Tolerance.default_for(np.int32).kind == "exact"


def test_compare_can_flush_subnormals():
    from aie.utils.verify import Tolerance, compare

    tiny = np.float32(1e-40)  # subnormal in float32
    got = np.array([0.0, 1.0, tiny], np.float32)
    ref = np.array([tiny, 1.0, 0.0], np.float32)
    assert not compare(got, ref, Tolerance.exact())
    assert compare(got, ref, Tolerance.exact(), subnormals="flush")
    with pytest.raises(ValueError):
        compare(got, ref, Tolerance.exact(), subnormals="sometimes")
