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
from aie.utils.accuracy import all_bf16, round_to
from ml_dtypes import bfloat16

_TRUE = {
    "tanh_ref": np.tanh,
    "sigmoid_ref": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "silu_ref": lambda x: x / (1.0 + np.exp(-x)),
    "gelu_ref": lambda x: 0.5
    * x
    * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))),
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


def test_sigmoid_table_keeps_the_tail_the_tanh_path_rounds_to_zero():
    # AIE2P's sigmoid table against aie2's 0.5 * (1 + tanh(x/2)), which
    # rounds tanh to bf16 first: from x = -7.5 to -6.9 that is -1, and the
    # sigmoid 0. Both are 0 below -7.5, the table's flat end segment.
    x = all_bf16(nan=False)
    x = x[(x >= -8) & (x < -2)]
    true = 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
    tail = x >= -7.5
    table = kernels.sigmoid_table_ref(x).astype(np.float64)
    assert (table[~tail] == 0).all()
    # At worst 7.05% off, at x = -7.
    assert (np.abs(table[tail] - true[tail]) <= 0.071 * true[tail]).all()
    old = kernels.sigmoid_lut_ref(x).astype(np.float64)
    assert (old[tail & (x <= -6.9)] == 0).all()


def test_silu_table_keeps_the_tail_the_tanh_path_rounds_to_zero():
    # silu on AIE2P's sigmoid table, against aie2's tanh-path model.
    x = all_bf16(nan=False)
    x = x[(x >= -8) & (x < -2)]
    xf = x.astype(np.float64)
    true = xf / (1.0 + np.exp(-xf))
    tail = x >= -7.5
    table = kernels.silu_table_ref(x).astype(np.float64)
    assert (table[~tail] == 0).all()
    # At worst 7.17% off, at x = -7.
    assert (np.abs(table[tail] - true[tail]) <= 0.072 * np.abs(true[tail])).all()
    old = kernels.silu_lut_ref(x).astype(np.float64)
    assert (old[tail & (x <= -6.9)] == 0).all()
    assert kernels.silu_table_ref(np.array([-np.inf], bfloat16))[0] == 0


def test_lut_models_flush_subnormal_products():
    # The accumulator flushes a subnormal product to zero before storing it.
    # Without that, silu_lut_ref gave 506 subnormals over every bf16 where
    # npu2 gave 0.
    x = all_bf16()
    one = np.ones_like(x)
    with np.errstate(over="ignore", invalid="ignore"):
        outs = {
            "silu_lut_ref(x)": kernels.silu_lut_ref(x),
            "swiglu_lut_ref(x, 1, 1)": kernels.swiglu_lut_ref(x, one, one),
            "swiglu_lut_ref(1, 1, x)": kernels.swiglu_lut_ref(one, one, x),
        }
    for name, got in outs.items():
        a = np.abs(got.astype(np.float32))
        sub = (a > 0) & (a < 2.0**-126)
        assert not sub.any(), f"{name}: {sub.sum()} subnormal outputs"
    # x * w2 = 2**-128 flushes, so silu is 0 and the gate zeroes the output
    # however large x * w1 is.
    x, w1, w2 = (np.array([v], bfloat16) for v in (2.0**-64, 2.0**100, 2.0**-64))
    assert kernels.swiglu_lut_ref(x, w1, w2)[0] == 0


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
