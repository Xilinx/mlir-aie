# aie_fp32.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""AIE2P scalar fp32 arithmetic in numpy, for bit-exact oracles of kernels.

Peano lowers every scalar float add or subtract to one lane of the
accumulator adder, and `aie::add` uses the same adder. It is not IEEE
round-to-nearest: it first rounds the smaller-magnitude operand onto the
larger operand's ulp grid, then rounds the sum. Multiplies are IEEE, so only
adds need modelling.

Both roundings are to nearest, ties to even, as under
`aie::rounding_mode::conv_even`; on npu2 the core's reset mode adds the same.
Subnormals are out of scope: the core flushes a subnormal sum to zero, and
this model does not.
"""

import numpy as np
from numpy.typing import ArrayLike


def aie_add(u: ArrayLike, v: ArrayLike) -> np.ndarray:
    """Add as the AIE2P accumulator adder does.

    Args:
        u: fp32 scalar or array.
        v: fp32 scalar or array, broadcastable with `u`.

    Returns:
        The fp32 sum, elementwise.
    """
    u = np.asarray(u, np.float32).astype(np.float64)
    v = np.asarray(v, np.float32).astype(np.float64)
    u_bigger = np.abs(u) >= np.abs(v)
    big = np.where(u_bigger, u, v)
    small = np.where(u_bigger, v, u)
    # From the exponent: np.spacing on a float32 overflows at its maximum.
    _, exponent = np.frexp(np.abs(big))
    ulp = np.ldexp(1.0, np.maximum(exponent - 24, -149))
    return (big + np.round(small / ulp) * ulp).astype(np.float32)
