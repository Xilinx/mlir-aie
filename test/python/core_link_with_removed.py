# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Verify that Core() raises TypeError when link_with is passed.
# link_with must be set on external_func() declarations instead.

# RUN: %python %s

import pytest
from aie.dialects.aie import AIEDevice, Core, Device, end, tile
from aie.ir import Block, InsertionPoint
from aie.extras.context import mlir_mod_ctx


def _make_core_with_link_with():
    with mlir_mod_ctx():
        dev = Device(AIEDevice.npu1_1col)
        dev_block = Block.create_at_start(dev.body_region)
        with InsertionPoint(dev_block):
            t = tile(col=0, row=2)
            Core(t, link_with="test.o")


# Core(link_with=...) must raise TypeError with a message directing users
# to external_func().
with pytest.raises(TypeError, match="link_with"):
    _make_core_with_link_with()

print("PASS: Core(link_with=...) correctly raises TypeError")
