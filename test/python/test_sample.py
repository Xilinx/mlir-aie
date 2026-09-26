# test_sample.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""sample_select / sample_combine factories and the sampling reference (no NPU required).

The reference is checked against computations that share none of its code:
math.exp for exp64, a stable sort for the top-k candidates, a scan for the
argmax and exact rational prefix sums for the draw.
"""

import math
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from aie.iron import In, InOut, Out, kernels
from aie.iron.kernels import sample
from ml_dtypes import bfloat16

_ROOT = Path(__file__).resolve().parents[2]


def _ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in float64 steps between same-signed, non-NaN a and b."""
    return np.abs(a.view(np.int64) - b.view(np.int64))


def _math_exp(x: np.ndarray) -> np.ndarray:
    def one(v):
        try:
            return math.exp(v)
        except OverflowError:
            return math.inf

    return np.array([one(float(v)) for v in x], dtype=np.float64)


def _all_bf16() -> np.ndarray:
    """Every bf16 value but NaN."""
    bits = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    values = bits.view(bfloat16)
    return values[~np.isnan(values.astype(np.float32))]


def _logits(rng, n, *, ties=0, scale=3.0):
    logits = rng.normal(0, scale, n).astype(bfloat16)
    if ties:
        logits[rng.choice(n, ties, replace=False)] = logits.max()
    return logits


# --- factory arguments -----------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, 1.0, 64.0, "64", True, np.bool_(True)])
@pytest.mark.parametrize(
    "factory,name",
    [
        (kernels.sample_select, "slice_size"),
        (kernels.sample_select, "chunk"),
        (kernels.sample_select, "k_max"),
        (kernels.sample_combine, "slice_size"),
        (kernels.sample_combine, "columns"),
        (kernels.sample_combine, "k_max"),
    ],
)
def test_factories_reject_non_positive_integers(factory, name, bad):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        factory(**{name: bad})


@pytest.mark.parametrize(
    "factory,kwargs,match",
    [
        (kernels.sample_select, dict(slice_size=32, chunk=32, k_max=33), "k_max"),
        (kernels.sample_combine, dict(slice_size=32, k_max=33), "k_max"),
        (kernels.sample_select, dict(slice_size=1024, chunk=256, k_max=129), "stack"),
        (kernels.sample_combine, dict(slice_size=1024, k_max=129), "stack"),
        (kernels.sample_select, dict(slice_size=1 << 24, chunk=1 << 12), "2\\*\\*24"),
        (kernels.sample_combine, dict(slice_size=1 << 24, columns=1), "2\\*\\*24"),
        (kernels.sample_select, dict(slice_size=1024, chunk=384), "divide"),
        (kernels.sample_select, dict(slice_size=1023, chunk=341), "even"),
        (kernels.sample_combine, dict(slice_size=1 << 23, columns=256), "int32"),
    ],
)
def test_factories_reject_bad_geometry(factory, kwargs, match):
    with pytest.raises(ValueError, match=match):
        factory(**kwargs)


def test_factories_accept_their_edges(npu2_device):
    kernels.sample_select(slice_size=64, chunk=64, k_max=64)
    kernels.sample_select(slice_size=(1 << 24) - 2, chunk=2, k_max=1)
    kernels.sample_select(slice_size=np.int64(1030), chunk=np.int32(206), k_max=8)
    kernels.sample_combine(slice_size=(1 << 23) - 1, columns=256, k_max=1)
    kernels.sample_select(slice_size=1024, chunk=256, k_max=sample.K_MAX_LIMIT)
    kernels.sample_combine(slice_size=1024, k_max=sample.K_MAX_LIMIT)


# --- geometry --------------------------------------------------------------


@pytest.mark.parametrize("slice_size", [1, 31, 32, 33, 1024, 1030, 32064])
@pytest.mark.parametrize("k_max", [1, 8, 64])
def test_summary_words(slice_size, k_max):
    header, entries = 8, 2 * k_max
    bitmap = math.ceil(slice_size / 32)
    assert sample.summary_words(slice_size, k_max) == header + entries + bitmap


def test_select_streams():
    assert sample.select_streams(1024, 1024) == 1
    assert sample.select_streams(1024, 512) == sample.SELECT_PASSES
    assert sample.select_streams(32064, 5344) == 2


def test_constants_match_the_c_sources():
    source = _ROOT / "aie_kernels" / "sample"
    select = (source / "sample_select.cc").read_text()
    header = (source / "sample.h").read_text()

    def define(name):
        return int(re.search(rf"#define {name} (\d+)", select).group(1))

    def constant(name):
        return int(re.search(rf"constexpr int32_t {name} = (\d+);", select).group(1))

    def enum(name):
        return int(re.search(rf"\b{name} = (\d+)", header).group(1))

    assert define("SAMPLE_SELECT_STATE_WORDS") == sample.SELECT_STATE_WORDS
    assert define("SAMPLE_SELECT_PASSES") == sample.SELECT_PASSES
    # sample_select's static_assert: no k_max the factories take fails in C.
    assert sample.K_MAX_LIMIT + constant("LANES") <= constant("CANDIDATES")
    assert enum("SAMPLE_HEADER") == sample.SUMMARY_HEADER
    rows = re.findall(r"\bSAMPLE_ROW_\w+ = (\d+)", header)
    assert sorted(map(int, rows)) == list(range(sample.ROW_WORDS))


# --- draw rows -------------------------------------------------------------


def test_draw_row_layout():
    n53 = (0x1ABCDE << 32) | 0x89ABCDEF
    row = sample.draw_row(0.7, 40, n53)
    assert row.dtype == np.int32 and row.shape == (sample.ROW_WORDS,)
    words = row.view(np.uint32)
    assert words[0] == np.float32(0.7).view(np.uint32)
    assert words[1] == 40
    assert words[2] == 0x89ABCDEF
    assert words[3] == 0x1ABCDE
    assert sample.draw_row(-0.0, 1, 0).view(np.uint32)[0] == 0x80000000
    assert sample.draw_row(1.0, 1, (1 << 53) - 1).view(np.uint32)[3] == 0x1FFFFF


@pytest.mark.parametrize("n53", [-1, 1 << 53, 1 << 64])
def test_draw_row_rejects_n53_outside_53_bits(n53):
    with pytest.raises(ValueError, match="n53"):
        sample.draw_row(1.0, 1, n53)


# --- exp64 -----------------------------------------------------------------


def test_exp64_ref_within_one_ulp_of_math_exp():
    rng = np.random.default_rng(64)
    x = np.concatenate(
        [
            rng.uniform(-745.2, 709.8, 100_000),
            rng.uniform(-1, 1, 20_000),
            rng.uniform(-(2.0**-50), 2.0**-50, 1_000),
            # The subnormal results and the edges of both special ranges.
            rng.uniform(-745.2, -708.3, 20_000),
            rng.uniform(510, 514, 1_000),
            rng.uniform(-514, -510, 1_000),
            rng.uniform(709.7, 709.8, 1_000),
            rng.uniform(-745.2, -745.1, 1_000),
            # Every sign of every binade.
            np.ldexp(1.0, np.arange(-60, 10)),
            -np.ldexp(1.0, np.arange(-60, 10)),
        ]
    )
    got, want = sample.exp64_ref(x), _math_exp(x)
    assert _ulps(got, want).max() <= 1


def test_exp64_ref_special_values():
    x = np.array([0.0, -0.0, -np.inf, np.inf, 710.0, 1e308, -746.0, -1e308, 2.0**-60])
    want = np.array([1.0, 1.0, 0.0, np.inf, np.inf, np.inf, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(sample.exp64_ref(x), want)
    assert np.isnan(sample.exp64_ref(np.array([np.nan]))).all()
    assert sample.exp64_ref(np.zeros((2, 3))).shape == (2, 3)


def test_exp64_table_is_what_its_generator_writes():
    pytest.importorskip("mpmath")
    subprocess.run(
        [sys.executable, str(_ROOT / "utils" / "generate_exp64_table.py"), "--check"],
        check=True,
    )


# --- keys and temperatures -------------------------------------------------


def test_order_keys_follow_numeric_order():
    values = _all_bf16()
    order = np.argsort(values.astype(np.float32), kind="stable")
    ordered = values[order].astype(np.float32)
    keys = sample.order_keys(values[order]).astype(np.int64)
    assert np.all(np.diff(keys) >= 0)
    # Equal keys exactly where the values are equal: only -0 and +0.
    np.testing.assert_array_equal(np.diff(keys) == 0, np.diff(ordered) == 0)
    assert np.count_nonzero(np.diff(keys) == 0) == 1


@pytest.mark.parametrize("temperature", [1.0, 0.7, 0.5, 2.0, 1e-3])
def test_check_order_preserving_accepts(temperature):
    sample.check_order_preserving(temperature)


@pytest.mark.parametrize("temperature", [1e30, 3e38])
def test_check_order_preserving_rejects_merging_temperatures(temperature):
    with pytest.raises(ValueError, match="maps two bf16 logits"):
        sample.check_order_preserving(temperature)


@pytest.mark.parametrize("temperature", [0.0, -0.0, -1.0, np.inf, np.nan])
def test_check_order_preserving_rejects_non_positive(temperature):
    with pytest.raises(ValueError, match="finite and positive"):
        sample.check_order_preserving(temperature)


# --- the reference ---------------------------------------------------------


def _first_argmax(logits) -> int:
    values = [float(v) for v in np.asarray(logits, dtype=bfloat16).astype(np.float32)]
    best = 0
    for i, v in enumerate(values):
        if v > values[best]:
            best = i
    return best


def _top_k(logits, top_k) -> np.ndarray:
    values = np.asarray(logits, dtype=bfloat16).astype(np.float32)
    k = min(top_k, values.size)
    tau = np.sort(values, kind="stable")[::-1][k - 1]
    return np.flatnonzero(values >= tau)


def _exact_draw(candidates, weights, n53) -> int:
    """First candidate whose exact prefix sum exceeds u times the exact total."""
    prefix = [Fraction(0)]
    for w in weights:
        prefix.append(prefix[-1] + Fraction(float(w)))
    target = Fraction(n53, 1 << 53) * prefix[-1]
    for c, p in zip(candidates, prefix[1:]):
        if p > target:
            return int(c)
    return int(candidates[-1])


@pytest.mark.parametrize("temperature", [0.0, -0.0])
def test_temperature_zero_takes_the_first_argmax(temperature):
    rng = np.random.default_rng(0)
    for _ in range(20):
        logits = _logits(rng, 257, ties=int(rng.integers(0, 4)))
        want = _first_argmax(logits)
        assert sample.sample_ref(logits, temperature, 1, 0) == want
        assert sample.sample_ref(logits, temperature, 64, (1 << 53) - 1) == want


def test_temperature_zero_edge_rows():
    # +0 and -0 are one value: the first of them wins.
    signed = np.array([-1.0, -0.0, 0.0, -2.0], dtype=bfloat16)
    assert sample.sample_ref(signed, 0.0, 1, 0) == 1
    assert sample.sample_ref(np.full(9, -np.inf, dtype=bfloat16), 0.0, 1, 0) == 0
    assert sample.sample_ref(np.full(9, 3.0, dtype=bfloat16), 0.0, 1, 0) == 0


@pytest.mark.parametrize("top_k", [1, 2, 7, 64, 300, 1000])
def test_candidates_are_the_top_k_with_ties(top_k):
    rng = np.random.default_rng(top_k)
    for ties in (0, 3, 40):
        logits = _logits(rng, 300, ties=ties)
        # Ties below the top too, so the k-th value often has company.
        logits[rng.choice(300, 30, replace=False)] = logits[0]
        candidates, weights = sample.sample_weights(logits, 1.0, top_k)
        np.testing.assert_array_equal(candidates, _top_k(logits, top_k))
        assert candidates.size >= min(top_k, logits.size)
        assert weights.shape == candidates.shape


def test_candidates_of_signed_zeros_and_minus_infinity():
    logits = np.array([-np.inf, 0.0, -0.0, -np.inf, -1.0], dtype=bfloat16)
    np.testing.assert_array_equal(sample.sample_weights(logits, 1.0, 1)[0], [1, 2])
    np.testing.assert_array_equal(sample.sample_weights(logits, 1.0, 3)[0], [1, 2, 4])
    candidates, weights = sample.sample_weights(logits, 1.0, 4)
    np.testing.assert_array_equal(candidates, np.arange(5))
    np.testing.assert_array_equal(weights[:4], [0.0, 1.0, 1.0, 0.0])
    assert _ulps(weights[4:], np.array([math.exp(-1)])).max() <= 1


@pytest.mark.parametrize("temperature", [1.0, 0.7, 2.0, 0.05])
def test_weights_are_the_float32_softmax_numerators(temperature):
    rng = np.random.default_rng(int(temperature * 100))
    logits = _logits(rng, 500, ties=2)
    candidates, weights = sample.sample_weights(logits, temperature, 50)
    t = np.float32(temperature)
    values = logits.astype(np.float32)
    x = [np.float32(values[c] / t) - np.float32(values.max() / t) for c in candidates]
    want = _math_exp(np.array(x, dtype=np.float32).astype(np.float64))
    assert _ulps(weights, want).max() <= 1
    assert (
        weights.max() == 1.0 and weights[values[candidates] == values.max()].min() == 1
    )


@pytest.mark.parametrize(
    "temperature,top_k,scale", [(1.0, 50, 3.0), (0.7, 64, 1.0), (2.0, 8, 10.0)]
)
def test_draws_match_exact_prefix_sums(temperature, top_k, scale):
    rng = np.random.default_rng(top_k)
    for _ in range(10):
        logits = _logits(rng, 1000, ties=3, scale=scale)
        candidates, weights = sample.sample_weights(logits, temperature, top_k)
        for n53 in rng.integers(0, 1 << 53, 20, dtype=np.int64):
            want = _exact_draw(candidates, weights, int(n53))
            assert sample.sample_ref(logits, temperature, top_k, int(n53)) == want


def test_draw_ends_take_the_first_and_last_candidate():
    rng = np.random.default_rng(5)
    logits = _logits(rng, 256, ties=2)
    candidates, weights = sample.sample_weights(logits, 1.0, 16)
    assert (weights > 0).all()
    assert sample.sample_ref(logits, 1.0, 16, 0) == candidates[0]
    assert sample.sample_ref(logits, 1.0, 16, (1 << 53) - 1) == candidates[-1]
    # top-k 1 with no tie: the argmax, whatever the draw.
    logits[_first_argmax(logits)] += bfloat16(1.0)
    for n53 in (0, 1 << 52, (1 << 53) - 1):
        assert sample.sample_ref(logits, 1.0, 1, n53) == _first_argmax(logits)


@pytest.mark.parametrize("k", [2, 3, 4, 5, 7, 64])
def test_draws_on_the_boundaries_between_equal_weights(k):
    # k equal weights: candidate j owns u in [j / k, (j + 1) / k), so the
    # draw at the first n53 of a share, and the one before it, pin the rule
    # to "first prefix strictly above u * total".
    logits = np.zeros(k + 3, dtype=bfloat16)
    logits[k:] = -1.0
    for j in range(1, k):
        first = -(-j * (1 << 53) // k)  # ceil(j * 2**53 / k)
        assert sample.sample_ref(logits, 1.0, k, first) == j
        assert sample.sample_ref(logits, 1.0, k, first - 1) == j - 1


def test_draw_skips_zero_weights():
    # exp(-200 / 0.1) underflows to 0: neither end of the draw may land there.
    logits = np.array([-200.0, 0.0, -200.0, 0.0, -200.0], dtype=bfloat16)
    assert sample.sample_ref(logits, 0.1, 5, 0) == 1
    assert sample.sample_ref(logits, 0.1, 5, (1 << 53) - 1) == 3


def test_draw_keeps_weights_below_a_double_doubles_reach():
    # At u = 1/2 the exact prefix passes u * S at the e^-100 (index 2); a
    # double-double sum of 1 + e^-40 + e^-100 drops it, and took index 3.
    logits = np.array([0, -40, -100, 0, -40], dtype=bfloat16)
    assert sample.sample_ref(logits, 1.0, 5, 1 << 52) == 2
    assert sample.sample_ref(logits, 1.0, 5, (1 << 52) + 1) == 3


def test_draws_on_the_boundaries_of_every_binade():
    # Weights from 1 down to subnormal float64 and 0, each drawn at the first
    # n53 past its prefix and the one before.
    rng = np.random.default_rng(1074)
    x = np.concatenate([np.geomspace(1e-3, 700, 60), [712, 730, 744, 750]])
    logits = rng.permutation(-x).astype(bfloat16)
    logits[rng.integers(64)] = 0
    candidates, weights = sample.sample_weights(logits, 1.0, 64)
    assert weights.min() == 0 and 0 < weights[weights > 0].min() < 2.0**-1022
    prefix = np.cumsum([Fraction(float(w)) for w in weights])
    for p in prefix[:-1]:
        first = -(-p * (1 << 53) // prefix[-1])  # ceil
        for n53 in (int(first) - 1, int(first)):
            want = _exact_draw(candidates, weights, n53)
            assert sample.sample_ref(logits, 1.0, 64, n53) == want


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_reference_rejects_nan_and_plus_infinity(bad):
    logits = np.zeros(8, dtype=bfloat16)
    logits[3] = bad
    for temperature in (0.0, 1.0):
        with pytest.raises(ValueError, match="NaN or \\+inf"):
            sample.sample_ref(logits, temperature, 4, 0)


def test_reference_rejects_a_non_finite_scaled_maximum():
    logits = np.array([3e38, 0.0], dtype=bfloat16)
    with pytest.raises(ValueError, match="not finite"):
        sample.sample_weights(logits, 1e-3, 1)


# --- factory metadata ------------------------------------------------------


def test_select_factory_metadata(npu2_device):
    fn = kernels.sample_select(slice_size=2048, chunk=512, k_max=32)
    assert Path(fn._source_file).name == "sample_select.cc"
    assert Path(fn._source_file).parent.name == "sample"
    for flag in (
        "-DSAMPLE_SLICE=2048",
        "-DSAMPLE_CHUNK=512",
        "-DSAMPLE_K_MAX=32",
        "-ffp-contract=off",
        "-fno-fast-math",
    ):
        assert flag in fn._compile_flags
    assert fn.arg_shape(0) == (512,) and fn.arg_dtype(0) == bfloat16
    assert fn.arg_shape(1) == (sample.ROW_WORDS,)
    assert fn.arg_shape(2) == (sample.SELECT_STATE_WORDS,)
    assert fn.arg_shape(3) == (sample.summary_words(2048, 32),)
    assert all(fn.arg_dtype(i) == np.int32 for i in (1, 2, 3))
    assert fn.contract.roles == (In, In, InOut, InOut)
    assert fn == kernels.sample_select(slice_size=2048, chunk=512, k_max=32)


def test_combine_factory_metadata(npu2_device):
    fn = kernels.sample_combine(columns=3, slice_size=2048, k_max=32)
    assert Path(fn._source_file).name == "sample_combine.cc"
    for flag in (
        "-DSAMPLE_SLICE=2048",
        "-DSAMPLE_COLUMNS=3",
        "-DSAMPLE_K_MAX=32",
        "-ffp-contract=off",
        "-fno-fast-math",
    ):
        assert flag in fn._compile_flags
    assert fn.arg_shape(0) == (3 * sample.summary_words(2048, 32),)
    assert fn.arg_shape(1) == (sample.ROW_WORDS,)
    assert fn.arg_shape(2) == fn.arg_shape(3) == (1,)
    assert all(fn.arg_dtype(i) == np.int32 for i in range(4))
    assert fn.contract.roles == (In, In, Out, Out)
    assert fn.contract.stack_bytes == 4096
