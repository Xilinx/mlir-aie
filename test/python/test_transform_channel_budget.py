# test_transform_channel_budget.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""num_channels is bounded by the shim, and a binary transform has two inputs.

One worker per (column, channel) fills one fifo per input from the shim and
drains one, so a column needs ``num_inputs * num_channels`` MM2S channels
and it has two. ``transform_parallel_binary(..., num_channels=2)`` asks for
four, and used to build MLIR that failed to place thirteen stages later:

    error: no ShimNOCTile on the device has 0 input/1 output DMA channel(s)
    free: all 8 ShimNOCTile(s) are at 0/16 input, 16/16 output channels used

No device needed: this is the declaration-time check, not the placement.
"""

import pytest

from aie.iron.algorithms._transform import (
    _SHIM_CHANNELS_PER_DIRECTION,
    _check_num_channels,
)


def test_two_channels_fit_one_input():
    _check_num_channels(2, num_inputs=1)
    _check_num_channels(1, num_inputs=1)


def test_a_binary_transform_has_room_for_one_channel():
    _check_num_channels(1, num_inputs=2)
    with pytest.raises(ValueError, match="needs 4 shim MM2S channels"):
        _check_num_channels(2, num_inputs=2)


def test_the_message_names_the_channel_count_that_fits():
    with pytest.raises(ValueError, match=r"Use num_channels=1\."):
        _check_num_channels(2, num_inputs=2)


def test_the_input_count_defaults_to_one():
    # Callers that predate the argument keep the unary budget.
    _check_num_channels(_SHIM_CHANNELS_PER_DIRECTION)


@pytest.mark.parametrize("bad", [0, 3, -1])
def test_only_one_or_two_channels_exist(bad):
    with pytest.raises(ValueError, match="num_channels must be 1 or 2"):
        _check_num_channels(bad)
