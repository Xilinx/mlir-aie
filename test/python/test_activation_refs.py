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


@pytest.mark.parametrize("name", sorted(_TRUE))
def test_unary_ref_is_correctly_rounded_on_every_bf16(name):
    x = all_bf16(nan=False)
    with np.errstate(over="ignore", invalid="ignore"):
        want = round_to(_TRUE[name](x.astype(np.float64)), bfloat16)
    got = getattr(kernels, name)(x)
    assert got.dtype == bfloat16
    bad = got.view(np.uint16) != want.view(np.uint16)
    assert not bad.any(), (
        f"{name}: {bad.sum()} inputs, first x={float(x[bad][0])} "
        f"got {float(got[bad][0])} want {float(want[bad][0])}"
    )


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
