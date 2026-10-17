# test_activation_refs.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""The activation references are the correctly rounded float64 functions.

A reference computed in float32 is itself wrong where float32 is: exp(-x)
overflows from x = -88.7 down, so sigmoid and silu came out 0 for inputs
whose true result is a nonzero bf16, and an error measured against them was
partly the reference's.
"""

import math

import ml_dtypes
import numpy as np
import pytest
from aie.iron import kernels
from aie.iron.kernels.activation import soft_cap_bound
from aie.utils.accuracy import all_bf16, round_to
from ml_dtypes import bfloat16

_TRUE = {
    "tanh_ref": np.tanh,
    "sigmoid_ref": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "silu_ref": lambda x: x / (1.0 + np.exp(-x)),
    # x sigmoid(2u): 0.5 x (1 + tanh(u)) cancels in float64 from x = -5 down.
    "gelu_ref": lambda x: x
    / (1.0 + np.exp(-2.0 * math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))),
    # The kernel clamps before its table lookup. float32 gave 936 at
    # x = 6.84375, where exp is 938.00005 and rounds up to 940.
    "bf16_exp_ref": lambda x: np.exp(np.clip(x, -88.0, 88.0)),
}


_MAX = float(ml_dtypes.finfo(bfloat16).max)
_ENDS = np.array([-np.inf, -_MAX, _MAX, np.inf], bfloat16)
# silu and gelu at -inf are -inf / inf and -inf * 0 as written, so NaN, where
# the limit is -0.
_LIMITS = {
    "tanh_ref": [-1, -1, 1, 1],
    "sigmoid_ref": [0, 0, 1, 1],
    "silu_ref": [-0.0, -0.0, _MAX, np.inf],
    "gelu_ref": [-0.0, -0.0, _MAX, np.inf],
}


@pytest.mark.parametrize("name", sorted(_TRUE))
def test_unary_ref_is_correctly_rounded_on_every_bf16(name):
    x = all_bf16(nan=False, inf=False)
    with np.errstate(over="ignore", invalid="ignore"):
        want = round_to(_TRUE[name](x.astype(np.float64)), bfloat16)
    got = getattr(kernels, name)(x)
    assert got.dtype == bfloat16
    bad = got.view(np.uint16) != want.view(np.uint16)
    assert not bad.any(), (
        f"{name}: {bad.sum()} inputs, first x={float(x[bad][0])} "
        f"got {float(got[bad][0])} want {float(want[bad][0])}"
    )


@pytest.mark.parametrize("name", sorted(_LIMITS))
def test_unary_ref_takes_the_limit_at_the_ends(name):
    with np.errstate(invalid="raise", divide="raise"):
        got = getattr(kernels, name)(_ENDS)
    want = np.array(_LIMITS[name], bfloat16)
    assert (got.view(np.uint16) == want.view(np.uint16)).all(), f"{name}: {got}"


def test_swiglu_ref_is_zero_where_silu_is():
    # x * w1 overflows to inf in the last two, and silu is -0 in all but the
    # middle two.
    x = np.array([-np.inf, -_MAX, _MAX, np.inf, 1e20, 1], bfloat16)
    w1 = np.array([1, 1, 1, 1, 1e20, np.inf], bfloat16)
    w2 = np.array([1, 1, 1, 1, -1e20, -np.inf], bfloat16)
    with np.errstate(invalid="raise", divide="raise"):
        got = kernels.swiglu_ref(x, w1, w2)
    want = np.array([0.0, 0.0, np.inf, np.inf, -0.0, -0.0], bfloat16)
    assert (got.view(np.uint16) == want.view(np.uint16)).all(), got


@pytest.mark.parametrize("name", ["sigmoid_ref", "silu_ref"])
def test_ref_keeps_the_tail_float32_loses(name):
    x = np.array([-89.0, -90.0, -92.0], bfloat16)
    xf = x.astype(np.float32)
    with np.errstate(over="ignore"):
        f32 = (xf if name == "silu_ref" else 1.0) / (1.0 + np.exp(-xf))
    assert (f32.astype(bfloat16) == 0).all()
    assert (getattr(kernels, name)(x).astype(np.float64) != 0).all()


def test_swiglu_ref_rounds_the_products_then_is_float64():
    rng = np.random.default_rng(0)
    x, w1, w2 = (
        (rng.standard_normal(1 << 14) * s).astype(bfloat16) for s in (4, 4, 30)
    )
    xw1 = round_to(x.astype(np.float64) * w1.astype(np.float64), bfloat16)
    xw2 = round_to(x.astype(np.float64) * w2.astype(np.float64), bfloat16)
    a, b = xw1.astype(np.float64), xw2.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        want = round_to(a * (b / (1.0 + np.exp(-b))), bfloat16)
    got = kernels.swiglu_ref(x, w1, w2)
    assert (got.view(np.uint16) == want.view(np.uint16)).all()


def test_softmax_ref_is_correctly_rounded_per_tile():
    # float32 rounded 4 of these 65536 outputs the wrong way.
    x = np.random.default_rng(2).standard_normal((64, 1024)).astype(bfloat16)
    xf = x.astype(np.float64)
    e = np.exp(xf - xf.max(axis=1, keepdims=True))
    want = round_to(e / e.sum(axis=1, keepdims=True), bfloat16)
    got = kernels.softmax_ref(x.reshape(-1)).reshape(x.shape)
    assert got.dtype == bfloat16
    bad = got.view(np.uint16) != want.view(np.uint16)
    assert not bad.any(), f"softmax_ref: {bad.sum()} outputs not correctly rounded"


def test_softmax_ref_float_path_caps_adds_and_masks():
    # The logits are x + addend, capped to cap * tanh(. / cap), plus the
    # mask, in float64; the result rounds once, to float32.
    rng = np.random.default_rng(3)
    x = rng.uniform(-400, 400, (2, 32)).astype(np.float32)
    addend = rng.uniform(-8, 8, (2, 32)).astype(np.float32)
    mask = np.zeros((2, 32), bfloat16)
    mask[0, 20:] = ml_dtypes.finfo(bfloat16).min
    mask[1, 25:] = -np.inf
    z = 50.0 * np.tanh((x.astype(np.float64) + addend) / 50.0)
    z = z + mask.astype(np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    want = round_to(e / e.sum(axis=1, keepdims=True), np.float32)
    got = kernels.softmax_ref(x, addend, mask, tile_size=32, soft_cap=50.0)
    assert got.dtype == np.float32
    assert (got == want).all()
    assert (got[0, 20:] == 0).all() and (got[1, 25:] == 0).all()


def test_softmax_ref_soft_cap_saturates():
    # tanh saturates: 1e4 and -1e4 are logits cap and -cap.
    x = np.full(32, -1e4, np.float32)
    x[0] = 1e4
    got = kernels.softmax_ref(x, tile_size=32, soft_cap=30.0).astype(np.float64)
    assert np.isclose(got[0] / got[1], np.exp(60.0), rtol=1e-6)


@pytest.mark.parametrize("cap", [0.5, 30.0, 50.0])
def test_soft_cap_bound_is_one_ulp_only_near_ties(cap):
    x = all_bf16(nan=False, inf=False)
    ref = kernels.soft_cap_ref(x, cap=cap).astype(np.float64)
    bound = soft_cap_bound(x, cap=cap)
    slack = bound > 2.0**-126
    assert (
        bound[slack] <= np.exp2(np.floor(np.log2(np.abs(ref[slack]))) - 7) + 2.0**-126
    ).all()
    assert slack.mean() < 0.02
    assert kernels.soft_cap_ref(np.array([np.inf, -np.inf]), cap=cap).tolist() == [
        cap,
        -cap,
    ]
