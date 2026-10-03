# aie2p_emulation.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""AIE2P arithmetic in numpy, for references that match the device bit for bit.

The functions take and return float arrays unless their docstrings say
otherwise. A bf16 result is a float32 array of values that bf16 represents.

- A core rounds toward minus infinity when it converts a float to bf16,
  unless the kernel sets the rounding mode register. ``rounding="conv_even"``
  models a kernel that selects round to nearest even. It also models
  host-side constants.
- AIE2P has no fp32 multiplier. ``fmul`` models the product of bf16 limbs that
  the cores compute.
- ``inv_bf16`` and ``gelu_bf16`` model the table lookups ``getInvBf16`` and
  ``getGeluBf16`` of aie_runtime_lib for AIE2P.
  [`bf16_exp_lut_ref`][iron.kernels.activation.bf16_exp_lut_ref] models the
  exponential table, and [`aie.utils.bfp.quantize`][utils.bfp.quantize] the
  bfp16ebs8 conversion.
"""

import re
from functools import cache
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.utils import config

_ROUNDINGS = ("floor", "conv_even")


def _from_bits(bits):
    """bf16 bit patterns to the float32 values they encode."""
    return (np.asarray(bits).astype(np.uint32) << 16).view(np.float32)


def bf16_floor_bits(x):
    """float32 ``x`` rounded to bf16 toward minus infinity, as uint16 bit patterns."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    inexact = (u & 0xFFFF) != 0
    negative = (u >> 31) != 0
    return ((u >> 16) + (inexact & negative)).astype(np.uint16)


def bf16_floor(x):
    """float32 ``x`` rounded to bf16 toward minus infinity, as float32 values."""
    return _from_bits(bf16_floor_bits(x))


def f32(x):
    """``x`` rounded to fp32 once, as an AIE fp32 accumulator rounds a sum.

    ``f32`` converts ``x`` to float64 first, so a float64 sum of exact
    products rounds once.
    """
    return np.asarray(x, np.float64).astype(np.float32)


def round_bf16_bits(x, rounding="floor"):
    """``x`` rounded to fp32, then to bf16 bit patterns (uint16).

    ``rounding`` is ``"floor"`` (toward minus infinity, a core's default) or
    ``"conv_even"`` (round to nearest even). The first rounding models the fp32
    register that holds a value before a core narrows it.
    """
    if rounding not in _ROUNDINGS:
        raise ValueError(f"rounding must be 'floor' or 'conv_even', got {rounding!r}")
    if rounding == "conv_even":
        return f32(x).astype(bfloat16).view(np.uint16)
    return bf16_floor_bits(f32(x))


def round_bf16(x, rounding="floor"):
    """``x`` rounded to fp32, then to bf16, as float32 values.

    See [`round_bf16_bits`][utils.aie2p_emulation.round_bf16_bits] for
    ``rounding``.
    """
    return _from_bits(round_bf16_bits(x, rounding))


def fmul(a, b):
    """The fp32 product of ``a`` and ``b`` as AIE2P computes it.

    AIE2P splits each operand into three bf16 limbs and adds the nine limb
    products in fp32. Each addition rounds, so the result can differ from the
    IEEE product by up to two ulps.
    """
    a, b = np.broadcast_arrays(np.asarray(a, np.float64), np.asarray(b, np.float64))

    def limbs(v):
        v = f32(v)
        l0 = round_bf16(v, "conv_even")
        r = f32(v - l0)
        l1 = round_bf16(r, "conv_even")
        return [l0, l1, round_bf16(f32(r - l1), "conv_even")]

    A, B = limbs(a), limbs(b)
    acc = None
    for i, j in (
        (0, 0),
        (0, 1),
        (1, 0),
        (0, 2),
        (1, 1),
        (2, 0),
        (1, 2),
        (2, 1),
        (2, 2),
    ):
        p = A[i].astype(np.float64) * B[j]
        acc = f32(p) if acc is None else f32(acc + p)
    return acc


def tree_sum(x, lanes):
    """The fp32 sum over the last axis of ``x``, in an accumulator of ``lanes`` lanes.

    Each lane keeps a running sum of every ``lanes``-th value. A pairwise
    halving of the lanes then adds the lane sums. The last axis must be a
    multiple of ``lanes``, and ``lanes`` a power of two.
    """
    x = np.asarray(x, np.float64)
    xs = x.reshape(x.shape[:-1] + (-1, lanes))
    acc = f32(xs[..., 0, :])
    for i in range(1, xs.shape[-2]):
        acc = f32(acc + xs[..., i, :])
    while acc.shape[-1] > 1:
        h = acc.shape[-1] // 2
        acc = f32(acc[..., :h] + acc[..., h:])
    return acc[..., 0]


def fast_rsqrt(s):
    """``1 / sqrt(s)`` in fp32: the ``0x5f3759df`` seed, then two Newton steps in ``fmul``."""
    s = f32(s)
    half = fmul(s, np.float32(0.5))
    y = (
        (np.uint32(0x5F3759DF) - (s.view(np.uint32) >> 1))
        .astype(np.uint32)
        .view(np.float32)
    )
    for _ in range(2):
        y = fmul(y, f32(np.float32(1.5) - fmul(fmul(half, y), y)))
    return y


# getInvBf16's table: the 7-bit mantissa of 1 / (1 + m / 128) for each m.
INV_MANTISSA = np.round(256 / (1 + np.arange(128) / 128)).astype(np.uint32) & 0x7F


def inv_bf16(x):
    """``getInvBf16``: ``1 / x`` as bf16, from the exponent and ``INV_MANTISSA``.

    ``x`` must be positive. The function rounds ``x`` to the nearest bf16
    first. The relative error is up to 0.4% for a bf16 ``x`` and up to 0.6%
    for an fp32 ``x``.
    """
    bits = f32(x).view(np.uint32).astype(np.uint64) + 0x8000
    exponent = (bits & 0x7F800000) >> 23
    mantissa = (bits & 0x007FFFFF) >> 16
    inv_exp = (mantissa == 0).astype(np.uint64) + (253 - exponent)
    return _from_bits(
        (((inv_exp << 7) + INV_MANTISSA[mantissa]) & 0xFFFF).astype(np.uint16)
    )


@cache
def gelu_lut_segments():
    """``getGeluBf16``'s table: 64 (slope, offset) rows, one per 1/8 of [-4, 4).

    The function reads ``gelu_lut_ab`` from the aie_runtime_lib sources that the
    kernels compile against. No simple fit of GELU reproduces the table. The
    source holds each run of four segments twice, for the gather read.
    """
    path = Path(config.aie_runtime_lib_dir()) / "AIE2P" / "lut_based_ops.cpp"
    body = re.search(r"gelu_lut_ab\[\d+\]\s*=\s*\{([^}]*)\}", path.read_text())[1]
    values = [float(v.strip().rstrip("f")) for v in body.split(",") if v.strip()]
    return np.array(values, np.float32).reshape(-1, 2, 8)[:, 0].reshape(-1, 2)


def gelu_bf16(x):
    """``getGeluBf16``: GELU of ``x`` from ``gelu_lut_segments``, as bf16.

    The device reads the slope as bf16 and the offset as fp32. Inputs outside
    [-4, 4) take the end segments.
    """
    x = np.asarray(x, np.float64)
    k = np.clip(np.floor(x * 128).astype(np.int64), -512, 511) >> 4
    pair = gelu_lut_segments()[k + 32]
    slope = (pair[..., 0].view(np.uint32) & 0xFFFF0000).view(np.float32)
    return round_bf16(f32(slope.astype(np.float64) * x + pair[..., 1]))
