# test_aie2p_emulation.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""aie.utils.aie2p_emulation: AIE2P arithmetic on the host (no NPU)."""

import numpy as np
import pytest
from aie.iron.kernels import flm_gemma4
from aie.utils import aie2p_emulation as emu
from ml_dtypes import bfloat16

_EDGES = np.array(
    [0.0, -0.0, np.inf, -np.inf, 1.0, -1.0, 1e-40, -1e-40, 3.4e38, -3.4e38],
    np.float32,
)


def _floats(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) * 2.0 ** rng.integers(-60, 60, n)
    return np.concatenate([_EDGES, x.astype(np.float32)])


def _ulps(a, b):
    return np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32))


def test_bf16_floor_rounds_toward_minus_infinity():
    x = _floats()
    bits = emu.bf16_floor_bits(x)
    y = emu.bf16_floor(x)
    assert bits.dtype == np.uint16 and y.dtype == np.float32
    np.testing.assert_array_equal(y.view(np.uint32), bits.astype(np.uint32) << 16)
    finite = np.isfinite(y)
    assert np.all(y[finite] <= x[finite])
    above = np.nextafter(y.astype(bfloat16), bfloat16(np.inf)).astype(np.float32)
    assert np.all(above[finite] > x[finite])


def test_round_bf16_rounds_through_fp32():
    x = np.concatenate([_floats().astype(np.float64), [1 + 2.0**-30, 1 - 2.0**-30]])
    with np.errstate(over="ignore"):
        x32 = x.astype(np.float32)
    np.testing.assert_array_equal(emu.f32(x), x32)
    np.testing.assert_array_equal(
        emu.round_bf16(x, "conv_even"), x32.astype(bfloat16).astype(np.float32)
    )
    np.testing.assert_array_equal(emu.round_bf16(x), emu.bf16_floor(x32))
    # 1 - 2**-30 rounds to 1 in fp32, so its floor is 1, not the bf16 below.
    assert emu.round_bf16(1 - 2.0**-30) == 1.0
    with pytest.raises(ValueError, match="rounding"):
        emu.round_bf16(1.0, "rne")


def test_fmul_is_within_two_ulps_of_ieee():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(20000).astype(np.float32)
    b = rng.standard_normal(20000).astype(np.float32)
    p = emu.fmul(a, b)
    assert p.dtype == np.float32
    d = _ulps(p, a * b)
    assert d.max() <= 2
    assert np.mean(d == 0) > 0.5
    # Products of bf16 values are exact in every limb.
    ab = a.astype(bfloat16).astype(np.float32)
    bb = b.astype(bfloat16).astype(np.float32)
    np.testing.assert_array_equal(emu.fmul(ab, bb), ab * bb)


def test_tree_sum_adds_lanes_then_halves():
    x = np.array([2.0**24, 1.0, -(2.0**24), 1.0])
    # Lane 0 holds 2**24 - 2**24 = 0 and lane 1 holds 2: the sum is exact.
    assert emu.tree_sum(x, 2) == 2.0
    # One lane adds in order: 2**24 + 1 rounds to 2**24.
    assert emu.tree_sum(x, 1) == 1.0
    ints = np.random.default_rng(2).integers(-100, 100, (5, 256))
    np.testing.assert_array_equal(emu.tree_sum(ints, 32), ints.sum(axis=1))


def test_fast_rsqrt_after_two_newton_steps():
    s = np.random.default_rng(3).uniform(1e-3, 1e3, 20000).astype(np.float32)
    r = emu.fast_rsqrt(s)
    assert r.dtype == np.float32
    assert np.max(np.abs(r * np.sqrt(s.astype(np.float64)) - 1)) < 1e-5


def test_inv_bf16_relative_error():
    assert emu.INV_MANTISSA.shape == (128,) and emu.INV_MANTISSA[0] == 0
    # Every positive bf16 from 2**-31 to 2**32.
    x = (np.arange(0x3000, 0x4F80, dtype=np.uint32) << 16).view(np.float32)
    inv = emu.inv_bf16(x)
    assert np.max(np.abs(inv.astype(np.float64) * x - 1)) <= 0.004
    np.testing.assert_array_equal(inv, inv.astype(bfloat16).astype(np.float32))
    assert emu.inv_bf16(2.0) == 0.5


def test_gelu_bf16_follows_tanh_gelu_within_the_kernel_bound():
    assert emu.gelu_lut_segments().shape == (64, 2)
    x = np.linspace(-6, 6, 24001)
    y = emu.gelu_bf16(x)
    np.testing.assert_array_equal(y, y.astype(bfloat16).astype(np.float32))
    err = np.abs(y - flm_gemma4._gelu_tanh(x))
    assert np.all(err <= flm_gemma4._gelu_lut_bound(x))
