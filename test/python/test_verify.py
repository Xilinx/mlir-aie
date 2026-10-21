# test_verify.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit tests for aie.utils.verify (no NPU required)."""

import numpy as np
import pytest
from aie.utils.verify import count_mismatches, nearly_equal, poisoned

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


def test_ulps_reference_is_rounded_directly_not_through_float32():
    """Rounding the float64 reference through float32 first picks the wrong
    neighbour on this tie: 1 + 2**-8 + 2**-30 is correctly rounded to
    1.0078125, but casting it to float32 then bfloat16 lands on 1.0.
    """
    ref = np.array([1 + 2**-8 + 2**-30], np.float64)
    correct = np.array([1.0078125], bfloat16)
    assert compare(correct, ref, Tolerance.bf16_ulps(0)).ok


def test_ulps_atol_floor_admits_a_flushed_subnormal():
    """A subnormal the device flushed to zero meets the floor, not the ulps."""
    smallest_normal = 2.0**-126
    ref = np.array([smallest_normal / 8, 1.0], np.float32)
    got = np.array([0.0, 1.0], bfloat16)  # the subnormal came back flushed

    assert not compare(got, ref, Tolerance.bf16_ulps(1)).ok
    assert compare(got, ref, Tolerance.bf16_ulps(1, atol=smallest_normal)).ok
    # The floor is not a blanket pass: a normal value still owes its ulp.
    assert not compare(
        np.array([0.0, 2.0], bfloat16),
        ref,
        Tolerance.bf16_ulps(1, atol=smallest_normal),
    ).ok


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("range_frac", [None, 2.0**-127])
def test_ulps_atol_floor_excludes_the_smallest_normal(sign, range_frac):
    smallest_normal = 2.0**-126
    ref = np.array([sign * smallest_normal, 1.0], np.float32)
    got = np.array([0.0, 1.0], bfloat16)
    tol = Tolerance(ulps=1, atol=smallest_normal, range_frac=range_frac)
    assert not compare(got, ref, tol).ok
    ref[0] = sign * (smallest_normal - 2.0**-133)
    assert compare(got, ref, tol).ok


def test_ulps_range_floor_remains_inclusive_with_atol():
    ref = np.array([256.0, 0.0], np.float32)
    got = np.array([256.0, 1.0], bfloat16)
    assert compare(got, ref, Tolerance(ulps=0, atol=1.0, range_frac=1 / 256)).ok
    assert not compare(got, ref, Tolerance.bf16_ulps(0, atol=1.0)).ok


def test_range_frac_admits_an_output_its_own_terms_cancelled():
    """The case the floor exists for: a dot product that cancelled to near zero.

    Its error is set by the operands, not by the sum, so an elementwise
    relative bound reads it as entirely wrong while its neighbours pass.
    """
    ref = np.array([1000.0, -1000.0, 0.01], np.float32)
    got = np.array([1000.0, -1000.0, 0.31], np.float32)  # 0.3 absolute, everywhere

    assert not compare(got, ref, Tolerance.relative(0.05)).ok
    # 0.3 is 3e-4 of the 1000 range -- the same absolute error the two large
    # outputs carry and are forgiven for.
    assert compare(got, ref, Tolerance.relative(0.05, range_frac=1e-3)).ok


def test_range_frac_follows_the_range_where_a_fixed_atol_cannot():
    """One fraction holds at both scales; one atol can only suit one of them."""
    tol = Tolerance.relative(0.0, range_frac=1e-3)
    for scale in (34.0, 3.4e9):
        ref = np.array([scale, 0.0], np.float32)
        assert compare(np.array([scale, 5e-4 * scale], np.float32), ref, tol).ok
        assert not compare(np.array([scale, 2e-3 * scale], np.float32), ref, tol).ok


@pytest.mark.parametrize("dtype", [np.int32, np.float32, bfloat16])
def test_range_frac_scales_each_call_independently(dtype):
    ref = np.array([[1024, 0], [4, 0], [0, 0]], dtype)
    got = np.array([[1024, 1], [4, 1], [0, 1]], dtype)
    tol = Tolerance(
        ulps=0 if dtype == bfloat16 else None, rtol=0.0, range_frac=1 / 1024
    )
    assert compare(got, ref, tol).ok
    verdict = compare(got, ref, tol, range_axis=1)
    assert not verdict.ok
    assert verdict.n_checked == 6
    assert verdict.n_mismatch == 2
    assert verdict.first_bad_index == 3
    got[1:, 1] = 0
    assert compare(got, ref, tol, range_axis=1).ok


def test_range_frac_per_call_handles_nonfinite_and_empty_references():
    tol = Tolerance.relative(0.0, range_frac=1 / 1024)
    ref = np.array([[np.inf, np.nan], [4, 0]], np.float32)
    assert compare(ref, ref, tol, range_axis=1).ok
    got = ref.copy()
    got[1, 1] = 1
    verdict = compare(got, ref, tol, range_axis=1)
    assert not verdict.ok
    assert "max_abs_err/max|expected|=0.25" in verdict.detail
    for shape in ((0, 2), (2, 0)):
        empty = np.empty(shape, np.float32)
        assert compare(empty, empty, tol, range_axis=1).ok


def test_range_frac_scales_to_the_reference_not_the_output():
    """A kernel returning something large must not thereby widen its own bound.

    One wild element is inside the mismatch budget, so what decides the verdict
    is the second: under a bound scaled to ``got`` its 0.5 would sit far below
    the floor and the run would pass on the strength of the wild element alone.
    """
    ref = np.array([1.0, 1.0, 0.0], np.float32)
    got = np.array([1e6, 1.0, 0.5], np.float32)
    tol = Tolerance.relative(0.0, range_frac=1e-3, max_mismatch_frac=1 / 3)

    assert compare(got[1:], ref[1:], tol).ok is False  # 0.5 misses on its own
    assert not compare(got, ref, tol).ok


def test_range_frac_reports_the_fraction_it_measured():
    ref = np.array([100.0, 0.0], np.float32)
    got = np.array([100.0, 5.0], np.float32)
    detail = compare(got, ref, Tolerance.relative(0.0, range_frac=1e-3)).detail
    assert "max_abs_err/max|expected|=0.05" in detail
    assert "range_frac=0.001" in detail


def test_range_frac_applies_to_ulps_and_integer_kinds():
    ref = np.array([256.0, 0.0], np.float32)
    got = np.array([256.0, 1.0], bfloat16)  # 1.0 is many ulps from 0
    assert not compare(got, ref, Tolerance.bf16_ulps(1)).ok
    assert compare(got, ref, Tolerance(ulps=1, range_frac=0.01)).ok

    ints = (np.array([256, 2], np.int32), np.array([256, 0], np.int32))
    assert not compare(*ints, Tolerance.relative(0.0, 1.0)).ok
    assert compare(*ints, Tolerance.relative(0.0, range_frac=0.01)).ok


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_range_frac_is_inclusive_and_accepts_equal_zeros(dtype):
    ref = np.array([1000, 0, 0], dtype)
    got = np.array([1000, 1, 0], dtype)
    assert compare(got, ref, Tolerance.relative(0.0, range_frac=0.001)).ok


def test_range_frac_needs_a_tolerance_to_be_a_floor_under():
    with pytest.raises(ValueError, match="admits nothing"):
        Tolerance(range_frac=1e-3)
    with pytest.raises(ValueError, match="must be positive"):
        Tolerance(rtol=0.1, range_frac=0.0)


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


def test_a_bound_tolerance_holds_each_element_to_its_own_bound():
    tol = Tolerance.bounded(lambda x: np.abs(x) / 8, note="half-width band")
    assert tol.kind == "bound"
    ref = np.array([1.0, 8.0, -4.0, np.inf], np.float32).astype(bfloat16)
    x = np.array([0.0, 8.0, 16.0, 1.0])
    bound = tol.bound(x)
    got = np.array([1.0, 9.0, -6.0, np.inf], np.float32).astype(bfloat16)
    assert compare(got[1:], ref[1:], tol, bound=bound[1:])
    # The first element's input admits nothing; its neighbours' slack does
    # not carry over.
    v = compare(got + np.array([0.5, 0, 0, 0], np.float32), ref, tol, bound=bound)
    assert not v and v.n_mismatch == 1 and "abs_err/bound=inf" in v.detail
    v = compare(ref.copy(), ref, tol, bound=bound)
    assert v
    v = compare(got[:3] * 2, ref[:3], tol, bound=bound[:3])
    assert not v and "half-width band" in v.detail
    # Non-finite values still have to match, whatever the bound says.
    assert not compare(np.zeros(1, bfloat16), ref[3:], tol, bound=np.array([np.inf]))


def test_a_bound_tolerance_needs_its_bound_and_stands_alone():
    tol = Tolerance.bounded(np.abs)
    with pytest.raises(ValueError, match="bound evaluated"):
        compare(np.zeros(2, np.float32), np.zeros(2, np.float32), tol)
    with pytest.raises(ValueError, match="floating-point"):
        compare(np.zeros(2, np.int32), np.zeros(2, np.int32), tol, bound=1)
    with pytest.raises(ValueError, match="whole comparison"):
        Tolerance(bound=np.abs, atol=1e-3)


def test_integers_honor_an_lsb_slack_only_under_a_relative_tolerance():
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


def test_the_reference_models_overflow_and_compare_reports_when_it_does_not():
    """``compare`` measures; the reference is what says the kernel wraps or clips."""
    ref = np.array([100, 40000, -40000, 7], dtype=np.int64)
    # A wrapping kernel's reference casts; a saturating one's clips. Either
    # matches its own device output, and neither needs a flag here.
    assert compare(ref.astype(np.int16), ref.astype(np.int16), Tolerance.exact())
    sat = np.clip(ref, -32768, 32767).astype(np.int16)
    assert compare(sat, sat, Tolerance.exact())

    # A reference left in its wider type does not model either, so a failure
    # says so rather than blaming the kernel.
    v = compare(sat, ref, Tolerance.exact())
    assert not v and "overflows int16" in v.detail
    # A reference that fits is graded normally, with no such note.
    fits = np.array([1, 2, 3, 4], dtype=np.int64)
    v = compare(fits.astype(np.int16), fits, Tolerance.exact())
    assert v and "overflows" not in (v.detail or "")


def test_default_tolerance_is_dtype_aware():
    assert Tolerance.default_for(np.float32).rtol == 1e-4
    assert Tolerance.default_for(np.float16).rtol == 1e-2
    assert Tolerance.default_for(bfloat16).rtol == 0.128
    assert Tolerance.default_for(np.int32).kind == "exact"


def test_a_flushing_kernel_flushes_in_its_reference():
    """Denormal flushing belongs to the reference, not to a ``compare`` flag."""
    tiny = np.float32(1e-40)  # subnormal in float32
    got = np.array([0.0, 1.0, tiny], np.float32)
    ref = np.array([tiny, 1.0, 0.0], np.float32)
    assert not compare(got, ref, Tolerance.exact())

    def flush(x):
        return np.where(np.abs(x) < np.finfo(np.float32).tiny, np.float32(0), x)

    assert compare(flush(got), flush(ref), Tolerance.exact())


@pytest.mark.parametrize("dtype", [np.int16, np.uint8, np.float32, bfloat16])
def test_poisoned_gives_n_elements_of_a_value_no_kernel_writes(dtype):
    """An unwritten output must not pass against a reference of zeros."""
    buf = poisoned(4, dtype)
    assert buf.size == 4 and buf.dtype == np.dtype(dtype)
    assert (buf.view(np.uint8) == 0x55).all()
    assert not compare(buf, np.zeros(4, dtype), Tolerance.exact())
