# test_coherence_is_per_region.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""Coherence describes a region of memory, not the object naming it.

Every assertion here is about the same underlying bytes being described
consistently, however many tensors happen to point at them. They fail on a
design that stores the coherence state per object, and hold on one that stores
it per region.
"""

import numpy as np
import pytest

from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

GRANULE = 64
_N = GRANULE  # bytes


@pytest.fixture
def parent():
    buffer = XRTTensor((4 * GRANULE,), dtype=np.uint8, device="npu")
    buffer.fill_(0)
    buffer.device = "npu"
    return buffer


def test_two_views_of_one_region_agree(parent):
    """Reconciling bytes through one view reconciles them, full stop.

    Two tensors over the same bytes are two names for one fact. If they can
    disagree, one of them is lying, and nothing in the design says which.
    """
    first = parent.subview(0, (_N,))
    second = parent.subview(0, (_N,))
    assert np.shares_memory(first.data, second.data)

    first.to("cpu")

    assert second.device == first.device


def test_reconciling_a_view_reconciles_it_for_the_parent(parent):
    """The parent contains the view's bytes; it cannot believe otherwise."""
    view = parent.subview(0, (_N,))

    view.to("cpu")

    assert parent.device == "cpu" or view.device == parent.device


def test_a_host_write_through_a_view_is_visible_to_the_buffer(parent):
    """The pattern the whole arena design depends on.

    A caller writes into one window of an arena and then hands the arena to a
    kernel. The write went through the view; the dispatch syncs the parent. If
    the buffer does not know a host write happened inside it, the sync is a
    no-op and the device reads whatever was there before.
    """
    view = parent.subview(0, (_N,))
    with view.mutate() as buf:  # host write, through the view
        buf[:] = 0xAB

    # What a runtime does before dispatch.
    assert parent.device == "cpu", (
        "the parent still believes the device owns bytes the host just wrote, "
        "so its pre-dispatch sync will do nothing"
    )


def test_sibling_windows_hold_independent_state(parent):
    """Different regions may legitimately differ; that must stay expressible."""
    left = parent.subview(0, (_N,))
    right = parent.subview(_N, (_N,))

    left.to("cpu")

    # Distinct regions, so no constraint that they agree -- only that each is
    # described correctly.
    assert left.device == "cpu"
    assert right.device == "npu"
