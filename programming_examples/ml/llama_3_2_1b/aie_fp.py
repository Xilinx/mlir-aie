#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""fp32 arithmetic as the AIE2P kernels execute it, for the numpy mirrors.

Peano lowers every scalar float add or subtract to one lane of the
accumulator adder, and `aie::add` uses the same adder. It is not IEEE
round-to-nearest: it first rounds the smaller-magnitude operand onto the
larger operand's ULP grid, then rounds the sum. Multiplies are IEEE, so only
adds need mirroring.
"""

import numpy as np


def aie_add(u, v):
    """Adds like the AIE2P accumulator adder.

    Args:
        `u`: fp32 scalar or array.
        `v`: fp32 scalar or array, broadcastable with `u`.

    Returns:
        The fp32 sum, elementwise.
    """
    u = np.asarray(u, np.float32).astype(np.float64)
    v = np.asarray(v, np.float32).astype(np.float64)
    u_bigger = np.abs(u) >= np.abs(v)
    big = np.where(u_bigger, u, v)
    small = np.where(u_bigger, v, u)
    ulp = np.spacing(np.abs(big).astype(np.float32)).astype(np.float64)
    return (big + np.round(small / ulp) * ulp).astype(np.float32)


def sw_recip(a):
    """Mirrors the kernels' `sw_recip`: magic-constant guess plus 4 Newton steps.

    Args:
        `a`: fp32 scalar.

    Returns:
        The fp32 reciprocal the device computes.
    """
    a = np.float32(a)
    x = (np.int32(0x7EF477D5) - a.view(np.int32)).view(np.float32)
    for _ in range(4):
        x = np.float32(x * aie_add(np.float32(2.0), -np.float32(a * x)))
    return x


def sw_invsqrt(a):
    """Mirrors the kernels' `sw_invsqrt`: Quake-III guess plus 2 Newton steps.

    Args:
        `a`: fp32 scalar.

    Returns:
        The fp32 inverse square root the device computes.
    """
    a = np.float32(a)
    x = (np.int32(0x5F3759DF) - (a.view(np.int32) >> np.int32(1))).view(np.float32)
    for _ in range(2):
        t = np.float32(np.float32(np.float32(np.float32(0.5) * a) * x) * x)
        x = np.float32(x * aie_add(np.float32(1.5), -t))
    return x


def round_to_i8(v):
    """Mirrors the kernels' `round_to_i8`: truncate `v ± 0.5`, saturate.

    Args:
        `v`: fp32 array.

    Returns:
        The int8 array the device computes.
    """
    v = np.asarray(v, np.float32)
    half = np.where(v >= 0, np.float32(0.5), np.float32(-0.5))
    return np.clip(np.trunc(aie_add(v, half)), -128, 127).astype(np.int8)
