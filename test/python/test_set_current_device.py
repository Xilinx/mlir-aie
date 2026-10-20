# test_set_current_device.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""`set_current_device` selects at once and, used with `with`, restores on exit."""

import pytest

import aie.iron as iron
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.utils import get_current_device
from aie.utils.hostruntime import DeviceSelection, set_current_device


@pytest.fixture(autouse=True)
def unbound():
    with set_current_device(None):
        yield


def test_plain_call_selects_and_stays():
    npu1 = NPU1Col1()
    selection = set_current_device(npu1)
    assert isinstance(selection, DeviceSelection)
    assert selection.previous is None
    assert get_current_device(probe_runtime=False) is npu1


def test_with_selects_and_restores():
    npu1 = NPU1Col1()
    set_current_device(npu1)
    npu2 = NPU2Col1()
    with set_current_device(npu2) as selection:
        assert get_current_device(probe_runtime=False) is npu2
        assert selection.previous is npu1
    assert get_current_device(probe_runtime=False) is npu1


def test_nested_restores_each_level():
    npu1, npu2 = NPU1Col1(), NPU2Col1()
    with set_current_device(npu1):
        with set_current_device(npu2):
            with set_current_device(None):
                assert get_current_device(probe_runtime=False) is None
            assert get_current_device(probe_runtime=False) is npu2
        assert get_current_device(probe_runtime=False) is npu1
    assert get_current_device(probe_runtime=False) is None


def test_restores_when_the_block_raises():
    npu1 = NPU1Col1()
    set_current_device(npu1)
    with pytest.raises(ZeroDivisionError):
        with set_current_device(NPU2Col1()):
            1 / 0
    assert get_current_device(probe_runtime=False) is npu1


def test_none_clears_and_restores():
    npu2 = NPU2Col1()
    set_current_device(npu2)
    with set_current_device(None):
        assert get_current_device(probe_runtime=False) is None
    assert get_current_device(probe_runtime=False) is npu2


def test_iron_reexports_the_restoring_call():
    npu2 = NPU2Col1()
    with iron.set_current_device(npu2):
        assert iron.get_current_device(probe_runtime=False) is npu2
    assert get_current_device(probe_runtime=False) is None
