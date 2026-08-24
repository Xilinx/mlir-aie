# test_xrt_device_handle.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""How many times allocating tensors opens the device, with the open substituted.

Both halves are counted rather than asserted about indirectly: the open is what
fails transiently, and closing it is what makes the next allocation reopen it.
Substituting ``pyxrt.device`` and the XRT allocation leaves the real
``XRTTensor.__init__`` under test, so a call site that goes back to opening its
own handle is caught here rather than as a flake on a device runner.
"""

import numpy as np
import pytest

pytest.importorskip("pyxrt")

from aie.utils.hostruntime.xrtruntime import device as device_module  # noqa: E402
from aie.utils.hostruntime.xrtruntime import tensor as xrt_tensor_module  # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor  # noqa: E402


class FakeBo:
    """Stands in for a pyxrt buffer object over a bytearray."""

    def __init__(self, nbytes):
        self.storage = bytearray(nbytes)
        self.size = nbytes

    def map(self):
        return memoryview(self.storage)

    def sync(self, direction):
        pass


@pytest.fixture
def opens(monkeypatch):
    """Count device opens, and start from a process that has not opened one."""
    monkeypatch.setattr(device_module, "_DEVICES", {})
    monkeypatch.setattr(xrt_tensor_module.xrt, "bo", lambda *args: FakeBo(args[1]))

    calls = []

    def fake_device(index):
        calls.append(index)
        return object()

    monkeypatch.setattr(device_module.pyxrt, "device", fake_device)
    return calls


def test_allocating_many_tensors_opens_the_device_once(opens):
    """The reason this test exists.

    Every ``pyxrt.device()`` is a real open, and the handle closes the device
    when it is dropped, so a handle per tensor closes and reopens the device as
    tensors come and go. That reopen is what returned ENODEV mid-suite.
    """
    tensors = [XRTTensor((64,), dtype=np.uint8) for _ in range(4)]

    assert opens == [0]
    assert len({id(t.xrt_device) for t in tensors}) == 1


def test_a_supplied_device_is_used_as_given(opens):
    """A caller holding its own handle keeps it, and no open happens."""
    supplied = object()
    tensor = XRTTensor((64,), dtype=np.uint8, xrt_device=supplied)

    assert opens == []
    assert tensor.xrt_device is supplied


def test_a_view_shares_its_parents_handle(opens):
    parent = XRTTensor((128,), dtype=np.uint8)
    view = parent.subview(64, (64,))

    assert opens == [0]
    assert view.xrt_device is parent.xrt_device


def test_a_failed_open_is_retried(monkeypatch, opens):
    monkeypatch.setattr(device_module.time, "sleep", lambda _: None)
    handle = object()
    attempts = []

    def flaky(index):
        attempts.append(index)
        if len(attempts) < 3:
            raise RuntimeError("No such device with index '0'")
        return handle

    monkeypatch.setattr(device_module.pyxrt, "device", flaky)

    assert device_module.acquire_device() is handle
    assert len(attempts) == 3


def test_an_open_that_never_succeeds_propagates(monkeypatch, opens):
    monkeypatch.setattr(device_module.time, "sleep", lambda _: None)
    attempts = []

    def never(index):
        attempts.append(index)
        raise RuntimeError("No such device with index '0'")

    monkeypatch.setattr(device_module.pyxrt, "device", never)

    with pytest.raises(RuntimeError):
        device_module.acquire_device()
    assert len(attempts) == device_module._MAX_RETRIES


def test_a_failed_open_is_not_cached(monkeypatch, opens):
    """A device that comes back is usable, rather than poisoned for the process."""
    monkeypatch.setattr(device_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        device_module.pyxrt,
        "device",
        lambda index: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError):
        device_module.acquire_device()

    handle = object()
    monkeypatch.setattr(device_module.pyxrt, "device", lambda index: handle)
    assert device_module.acquire_device() is handle
