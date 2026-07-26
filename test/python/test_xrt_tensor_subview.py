# test_xrt_tensor_subview.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s

"""How XRTTensor derives a sub-buffer, without needing a device.

The on-device tests in ``npu-xrt/`` prove the result is right on hardware, but
they only run where hardware exists. This pins the derivation itself -- which
buffer a view is carved from and at what offset -- on any machine, so a lane
without an NPU still catches a backend that ignores the offset or nests from the
wrong buffer. Only the XRT allocation call is substituted; the code under test is
the real ``XRTTensor._subview``.
"""

import numpy as np
import pytest

pytest.importorskip("pyxrt")

from aie.utils.hostruntime.tensor_class import NpuTensor  # noqa: E402
from aie.utils.hostruntime.xrtruntime import tensor as xrt_tensor_module  # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor  # noqa: E402

GRANULE = 64


class FakeBo:
    """Stands in for a pyxrt buffer object over a bytearray."""

    def __init__(self, storage, size, offset, parent=None):
        self.storage = storage
        self.size = size
        self.offset = offset
        self.parent = parent

    def map(self):
        return memoryview(self.storage)[self.offset : self.offset + self.size]


@pytest.fixture
def fake_xrt(monkeypatch):
    """Record every sub-buffer derivation instead of allocating one."""
    calls = []

    def fake_bo(parent, size, offset):
        calls.append({"parent": parent, "size": size, "offset": offset})
        return FakeBo(parent.storage, size, offset, parent=parent)

    monkeypatch.setattr(xrt_tensor_module.xrt, "bo", fake_bo)
    return calls


def make_root(nbytes, dtype=np.uint8):
    """An XRTTensor of ``nbytes`` bytes over fake storage, without a device."""
    shape = (nbytes // np.dtype(dtype).itemsize,)
    root = XRTTensor.__new__(XRTTensor)
    NpuTensor.__init__(root, shape, dtype=dtype, device="npu")
    root.xrt_device = object()
    root._shape = shape
    root._bo = FakeBo(bytearray(nbytes), nbytes, 0)
    root._data = np.frombuffer(root._bo.map(), dtype=dtype).reshape(shape)
    return root


def test_subview_derives_from_the_parent_at_a_byte_offset(fake_xrt):
    root = make_root(4 * GRANULE)
    view = root.subview(GRANULE, (GRANULE,))

    assert len(fake_xrt) == 1
    assert fake_xrt[0]["parent"] is root._bo
    assert fake_xrt[0]["offset"] == GRANULE
    assert fake_xrt[0]["size"] == GRANULE
    assert view.storage_offset == GRANULE
    assert view.base is root
    assert view.xrt_device is root.xrt_device


def test_nested_subview_derives_from_the_root_at_an_accumulated_offset(fake_xrt):
    """The reason this test exists.

    A sub-buffer of a sub-buffer does not compose: XRT takes the host pointer
    from the parent (picking up its offset) but the device address from the
    underlying allocation (picking up only the innermost offset), so nesting
    that way leaves the host and the device addressing different regions. Every
    view is therefore derived from the buffer that owns the storage.
    """
    root = make_root(8 * GRANULE)
    outer = root.subview(2 * GRANULE, (4 * GRANULE,))
    inner = outer.subview(GRANULE, (GRANULE,))

    assert fake_xrt[-1]["parent"] is root._bo  # not outer._bo
    assert fake_xrt[-1]["offset"] == 3 * GRANULE  # 2 + 1 granules, from the root
    assert inner.storage_offset == 3 * GRANULE
    assert inner.base is root


def test_nested_subview_addresses_the_same_bytes_the_host_sees(fake_xrt):
    """Host view and derived offset must agree, which is what nesting broke."""
    root = make_root(8 * GRANULE)
    inner = root.subview(2 * GRANULE, (4 * GRANULE,)).subview(GRANULE, (GRANULE,))

    inner.data[:] = 0xC7
    written = np.nonzero(np.frombuffer(root._bo.storage, dtype=np.uint8))[0]
    assert written.min() == inner.storage_offset
    assert written.max() == inner.storage_offset + GRANULE - 1


def test_subview_keeps_its_parent(fake_xrt):
    root = make_root(2 * GRANULE)
    view = root.subview(GRANULE, (GRANULE,))
    assert view._storage is root


def test_subview_carries_the_requested_dtype(fake_xrt):
    root = make_root(4 * GRANULE, dtype=np.uint32)
    view = root.subview(GRANULE // 4, (GRANULE,), dtype=np.uint8)
    # offset counts the parent's elements: (GRANULE // 4) * 4 bytes
    assert fake_xrt[-1]["offset"] == GRANULE
    assert view.dtype == np.dtype(np.uint8)
    assert view.data.dtype == np.dtype(np.uint8)


def test_subview_enforces_alignment_on_this_backend_too(fake_xrt):
    root = make_root(4 * GRANULE)
    with pytest.raises(ValueError, match="coherence granule"):
        root.subview(1, (GRANULE,))
    assert fake_xrt == []  # rejected before any buffer was derived
