# test_xrt_tensor_subview.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""How XRTTensor derives a sub-buffer, with the allocation substituted.

``test_tensor_subview_device.py`` proves the result is right on hardware. This
pins the derivation itself -- which buffer a view is carved from and at what
offset -- by substituting only the XRT allocation call, so a wrong offset or a
nest from the wrong buffer is caught as a wrong argument rather than as wrong
bytes. The code under test is the real ``XRTTensor._subview``.
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

    def sync(self, direction):
        pass


@pytest.fixture
def fake_xrt(monkeypatch):
    """Record every sub-buffer derivation instead of allocating one."""
    calls = []

    def fake_bo(*args):
        if len(args) == 4:  # allocation: (device, nbytes, flags, group_id)
            return FakeBo(bytearray(args[1]), args[1], 0)
        parent, size, offset = args  # derivation from an existing allocation
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
    root._buffer = xrt_tensor_module.XRTBuffer(root.xrt_device, nbytes, None, 0, "npu")
    root._offset_bytes = 0
    root._bo = root._buffer.binding_handle(0, nbytes)
    root._data = root._buffer.host_bytes.view(dtype).reshape(shape)
    return root


def test_subview_derives_from_the_parent_at_a_byte_offset(fake_xrt):
    root = make_root(4 * GRANULE)
    view = root.subview(GRANULE, (GRANULE,))

    assert len(fake_xrt) == 1
    assert fake_xrt[0]["parent"] is root._buffer._bo
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

    assert fake_xrt[-1]["parent"] is root._buffer._bo  # not the intermediate view
    assert fake_xrt[-1]["offset"] == 3 * GRANULE  # 2 + 1 granules, from the root
    assert inner.storage_offset == 3 * GRANULE
    assert inner.base is root


def test_nested_subview_addresses_the_same_bytes_the_host_sees(fake_xrt):
    """Host view and derived offset must agree, which is what nesting broke."""
    root = make_root(8 * GRANULE)
    inner = root.subview(2 * GRANULE, (4 * GRANULE,)).subview(GRANULE, (GRANULE,))

    inner.data[:] = 0xC7
    written = np.nonzero(root._buffer.host_bytes)[0]
    assert written.min() == inner.storage_offset
    assert written.max() == inner.storage_offset + GRANULE - 1


def test_subview_keeps_its_parent(fake_xrt):
    root = make_root(2 * GRANULE)
    view = root.subview(GRANULE, (GRANULE,))
    assert view._storage is root


def test_subview_carries_the_requested_dtype(fake_xrt):
    root = make_root(4 * GRANULE, dtype=np.uint32)
    view = root.subview(GRANULE, (GRANULE,), dtype=np.uint8)
    # offset is bytes, so the parent's uint32 does not scale it
    assert fake_xrt[-1]["offset"] == GRANULE
    assert view.dtype == np.dtype(np.uint8)
    assert view.data.dtype == np.dtype(np.uint8)


def test_subview_enforces_alignment_on_this_backend_too(fake_xrt):
    root = make_root(4 * GRANULE)
    with pytest.raises(ValueError, match="coherence granule"):
        root.subview(1, (GRANULE,))
    assert fake_xrt == []  # rejected before any buffer was derived


def test_a_transfer_covers_only_the_ranges_that_were_written(fake_xrt, monkeypatch):
    """Sending the whole extent is correct but wasteful, so nothing else catches it.

    A buffer is reconciled a range at a time precisely so that writing a few
    windows of a large arena does not maintain the whole arena. That is a
    property no functional assertion can see, since transferring more than was
    written still produces the right answer, so it is pinned here.
    """
    sent = []
    original = xrt_tensor_module.XRTBuffer.sync_to_device

    def recording(self, offset, nbytes):
        sent.append((offset, nbytes))
        return original(self, offset, nbytes)

    monkeypatch.setattr(xrt_tensor_module.XRTBuffer, "sync_to_device", recording)

    root = make_root(16 * GRANULE)
    root.to("npu")
    sent.clear()

    for start in (2 * GRANULE, 9 * GRANULE):
        with root.subview(start, (GRANULE,)).mutate() as buf:
            buf[:] = 0x5C

    root.to("npu")

    assert sorted(sent) == [(2 * GRANULE, GRANULE), (9 * GRANULE, GRANULE)], (
        "a reconcile should cover the written windows and nothing else, but "
        f"covered {sorted(sent)}"
    )
    assert sum(n for _, n in sent) == 2 * GRANULE
