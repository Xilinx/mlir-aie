# test_aie_fp32.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""aie.utils.aie_fp32's model of the AIE2P scalar fp32 adder."""

import numpy as np
from aie.utils.aie_fp32 import aie_add

MAX = np.finfo(np.float32).max
TINY = np.float32(2.0**-149)


def test_the_largest_float_adds_without_overflowing_its_ulp():
    assert aie_add(MAX, 0) == MAX
    assert aie_add(MAX, -MAX) == 0
    assert aie_add(-MAX, np.float32(1.0)) == -MAX


def test_subnormals_round_onto_the_smallest_ulp():
    assert aie_add(TINY, 0) == TINY
    assert aie_add(TINY, TINY) == np.float32(2.0**-148)


def test_the_smaller_operand_rounds_onto_the_larger_ones_grid_first():
    # 1 + 2**-24 is a tie in IEEE (rounds to 1); here 2**-24 rounds onto
    # 1's grid (ulp 2**-23) first, to 0 (half to even), so the sum is 1.
    assert aie_add(np.float32(1.0), np.float32(2.0**-24)) == np.float32(1.0)
    # 3 * 2**-25 rounds onto that grid as 2**-23, which IEEE would not do.
    assert aie_add(np.float32(1.0), np.float32(3 * 2.0**-25)) == np.float32(
        1.0 + 2.0**-23
    )
