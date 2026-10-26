# test_device_identity.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""A device's name, its constructor and repr, and its shim DMA budget."""

import pytest

from aie.dialects._aie_enum_gen import AIEArch
from aie.dialects.aie import AIEDevice
from aie.iron.device import (
    NPU1,
    NPU1Col1,
    NPU2,
    NPU2Col1,
    Device,
    NamedDevice,
    from_name,
)


@pytest.mark.parametrize(
    "device, name, arch",
    [
        (NPU1(), "npu1", AIEArch.AIE2),
        (NPU1Col1(), "npu1_1col", AIEArch.AIE2),
        (NPU2(), "npu2", AIEArch.AIE2p),
        (NPU2Col1(), "npu2_1col", AIEArch.AIE2p),
        (from_name("npu2", n_cols=4), "npu2_4col", AIEArch.AIE2p),
    ],
)
def test_a_device_names_itself_and_its_arch_is_the_family(device, name, arch):
    assert device.name == name == device.resolve().name
    assert device.arch is arch


def test_a_device_reprs_as_its_constructor():
    assert isinstance(from_name("npu2", n_cols=4), NamedDevice)
    assert repr(NPU2()) == "NPU2()"
    assert repr(from_name("npu2", n_cols=4)) == "NPU2Col4()"
    assert repr(Device(AIEDevice.npu2)) == "Device(AIEDevice.npu2)"


@pytest.mark.parametrize(
    "device, channels",
    [(NPU2(), 16), (from_name("npu2", n_cols=4), 8), (NPU1Col1(), 2)],
)
def test_shim_dma_channels_are_two_per_shim_tile_each_way(device, channels):
    assert device.shim_dma_channels_in == channels
    assert device.shim_dma_channels_out == channels
