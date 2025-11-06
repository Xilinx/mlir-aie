# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import pytest
from aie.iron.device import NPU2, Tile


def test_device_common_mem():
    dev = NPU2()
    cs = dev.get_compute_tiles()

    # single tile -- trivially correct for compute
    assert dev.has_common_mem([cs[0]])

    # Always false for shim and mem even when paired with compute tiles
    assert not dev.has_common_mem([dev.get_shim_tiles()[0]])
    assert not dev.has_common_mem([dev.get_mem_tiles()[0]])
    assert not dev.has_common_mem([cs[0], dev.get_shim_tiles()[0]])
    assert not dev.has_common_mem([cs[0], dev.get_mem_tiles()[0]])
    assert not dev.has_common_mem([dev.get_shim_tiles()[0], dev.get_shim_tiles()[1]])
    assert not dev.has_common_mem([dev.get_mem_tiles()[0], dev.get_mem_tiles()[1]])

    # correct with two neighbors, both directions
    assert cs[0].is_neighbor(cs[1])
    assert dev.has_common_mem(cs[:2])
    assert dev.has_common_mem([Tile(0, 3), Tile(1, 3)])
    assert dev.has_common_mem([Tile(0, 3), Tile(0, 4)])

    # correct with 5 files
    assert dev.has_common_mem(
        [Tile(0, 4), Tile(1, 4), Tile(2, 4), Tile(1, 3), Tile(1, 5)]
    )

    # check a few incorrect cases
    assert not dev.has_common_mem([Tile(0, 3), Tile(2, 3)])
    assert not dev.has_common_mem([Tile(0, 3), Tile(0, 5)])
