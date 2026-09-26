# test_accuracy.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""aie.utils.accuracy: rounding, ulp distance and error stats (no NPU)."""

import math

import numpy as np
import pytest
from aie.utils import accuracy
from aie.utils.accuracy import all_bf16, error_stats, round_to, ulp_distance, ulp_size
from aie.utils.verify import bf16_ulp_distance
from ml_dtypes import bfloat16


def _round_8_bits(x):
    """bf16 rounding of normal float64 values: 8 significant bits, ties even."""
    m, e = np.frexp(x)
    return np.ldexp(np.rint(np.ldexp(m, 8)), e - 8)


def test_round_to_is_correct_where_the_cast_rounds_twice():
    # Via float32 the sticky bit is lost: 1 + 2**-8 is a tie, which goes even.
    x = np.array([1 + 2**-8 + 2**-30])
    assert x.astype(bfloat16)[0] == 1.0
    assert round_to(x, bfloat16)[0] == 1.0078125


def test_round_to_matches_integer_rounding_on_normals():
    rng = np.random.default_rng(0)
    base = all_bf16(nan=False, inf=False, subnormal=False).astype(np.float64)
    base = base[(base != 0) & (np.abs(base) < 1e38)]
    # Midpoints, and a hair either side of them, are where rounding goes wrong.
    mid = base * (1 + 2**-9)
    x = np.concatenate(
        [mid, np.nextafter(mid, 0), np.nextafter(mid, np.inf), base, mid * 1.0000001]
    )
    x = np.concatenate([x, rng.uniform(-1e30, 1e30, 100_000)])
    got = round_to(x, bfloat16).astype(np.float64)
    np.testing.assert_array_equal(got, _round_8_bits(x))


def test_round_to_float32_agrees_with_numpy():
    # numpy's float64 -> float32 cast is correctly rounded (one rounding).
    x = np.random.default_rng(1).standard_normal(100_000) * 10.0 ** np.arange(
        -3, 7
    ).repeat(10_000)
    np.testing.assert_array_equal(round_to(x, np.float32), x.astype(np.float32))


def test_round_to_overflow_underflow_and_signed_zero():
    bmax = float(accuracy._finfo(bfloat16).max)
    half = 2.0**119  # half an ulp at the top binade
    x = np.array([bmax + half * 0.99, bmax + half, -(bmax + half), -1e-60, 1e-60])
    r = round_to(x, bfloat16).astype(np.float64)
    assert r[0] == bmax  # below the midpoint stays finite
    assert r[1] == np.inf and r[2] == -np.inf  # a tie goes to even: infinity
    assert r[3] == 0 and np.signbit(r[3]) and r[4] == 0 and not np.signbit(r[4])


def test_round_to_passes_nonfinite_through():
    r = round_to(np.array([np.inf, -np.inf, np.nan]), bfloat16).astype(np.float64)
    assert r[0] == np.inf and r[1] == -np.inf and np.isnan(r[2])


def test_ulp_distance_counts_representable_values():
    one = np.array([1.0], bfloat16)
    assert ulp_distance(one, np.array([1.0078125]), bfloat16)[0] == 1
    # Across zero: -min_sub, 0, +min_sub.
    tiny = float(accuracy._finfo(bfloat16).smallest_subnormal)
    assert ulp_distance(np.array([-tiny]), np.array([tiny]), bfloat16)[0] == 2
    assert ulp_distance(np.array([-0.0]), np.array([0.0]), bfloat16)[0] == 0
    bmax = float(accuracy._finfo(bfloat16).max)
    assert ulp_distance(np.array([bmax]), np.array([np.inf]), bfloat16)[0] == 1
    f = np.float32(1.0)
    assert (
        ulp_distance(np.array([f]), np.array([np.nextafter(f, 2)]), np.float32)[0] == 1
    )


def test_bf16_ulp_distance_is_the_bf16_case():
    a, b = all_bf16(nan=False), all_bf16(nan=False)[::-1]
    np.testing.assert_array_equal(bf16_ulp_distance(a, b), ulp_distance(a, b))


def test_ulp_size():
    assert ulp_size(np.array([1.0]), bfloat16)[0] == 2.0**-7
    assert ulp_size(np.array([1.5]), np.float32)[0] == 2.0**-23
    assert ulp_size(np.array([0.0]), bfloat16)[0] == 2.0**-133  # subnormal step


def test_all_bf16_classes():
    x = all_bf16()
    assert x.size == 65536
    assert np.array_equal(x.view(np.uint16), np.arange(65536, dtype=np.uint16))
    x64 = all_bf16(nan=False).astype(np.float64)
    assert x64.size == 65536 - 254 and not np.isnan(x64).any()
    assert all_bf16(nan=False, inf=False).size == 65536 - 256
    y = all_bf16(nan=False, inf=False, subnormal=False).astype(np.float64)
    assert y.size == 65536 - 256 - 254
    assert (y == 0).sum() == 2


def test_error_stats_counts_and_locates_the_worst():
    ref = np.array([1.0, 2.0, 3.0, 1 + 2**-8 + 2**-30, 100.0])
    got = np.array([1.0, 2.015625, 3.0, 1.0, 100.0], bfloat16)
    s = error_stats(got, ref)
    assert s.dtype == "bfloat16" and s.n == 5
    assert s.not_correctly_rounded == 2
    assert s.max_ulp == 1 and s.ulp_histogram == {0: 3, 1: 2}
    # Both miss by one ulp; the tie goes to the larger error against the
    # unrounded value: 2.015625 is a whole ulp (2**-6) off, 1.0 is just over half.
    assert s.worst_index == 1 and s.max_ulp_error == pytest.approx(1.0)
    assert s.max_abs == pytest.approx(2**-6) and s.max_abs_index == 1
    assert s.mean_ulp == pytest.approx(0.4)
    assert "2 not correctly rounded" in s.summary()


def test_error_stats_nonfinite():
    ref = np.array([np.inf, 1e300, np.nan, 1.0, np.nan])
    got = np.array([np.inf, np.inf, np.nan, np.nan, 1.0], bfloat16)
    s = error_stats(got, ref)
    # 1e300 overflows bfloat16: infinity is its correctly rounded value.
    assert s.nan_mismatch == 2 and s.not_correctly_rounded == 2
    assert s.max_ulp == 0 and s.max_abs == 0 and s.max_rel == 0


def test_error_stats_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="3 results for 2"):
        error_stats(np.zeros(3, np.float32), np.zeros(2))


def test_numpy_bf16_exp_against_float64():
    """Real numbers: exp over every finite bf16, via float32, vs float64."""
    x = all_bf16(nan=False, inf=False)
    with np.errstate(over="ignore"):
        ref = np.exp(x.astype(np.float64))
        got = np.exp(x.astype(np.float32)).astype(bfloat16)
    s = error_stats(got, ref)
    assert s.n == x.size
    # float32 exp is within a couple of float32 ulps; rounding that to bf16
    # can only miss the correctly rounded bf16 by one, on a near-tie.
    assert s.max_ulp <= 1 and s.max_ulp_error < 0.51
    assert s.not_correctly_rounded < 10
    # The correctly rounded answer scores zero, by construction.
    assert error_stats(round_to(ref, bfloat16), ref).not_correctly_rounded == 0
    # A kernel that truncates instead is caught, about half the time where
    # the result is neither 1 nor saturated.
    inner = (np.abs(x.astype(np.float64)) > 2**-6) & (np.abs(x) < 80)
    ref = ref[inner]
    trunc = (
        (ref.astype(np.float32).view(np.uint32) & 0xFFFF0000)
        .view(np.float32)
        .astype(bfloat16)
    )
    t = error_stats(trunc, ref)
    assert t.max_ulp == 1 and 0.3 < t.not_correctly_rounded / t.n < 0.7
    assert t.worst_index is not None and t.max_ulp_error > 0.9
    assert math.isfinite(t.mean_ulp)


def test_float32_stats():
    x = np.linspace(-10, 10, 10_001)
    ref = np.tanh(x)
    exact = error_stats(ref.astype(np.float32), ref)
    assert exact.not_correctly_rounded == 0 and exact.max_ulp_error <= 0.5
    off = error_stats(np.nextafter(ref.astype(np.float32), np.float32(np.inf)), ref)
    assert off.max_ulp == 1 and off.not_correctly_rounded == ref.size
