# test_tensor_subview_device.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device behaviour of ``Tensor.subview``.

A sub-region view is only useful if it is *independent*: the device can write
one view while the host owns the bytes around it, and synchronizing that view
disturbs neither its own result nor its neighbours.

That independence is not free. Host and device are reconciled a cache line at a
time, so two views sharing a line are not separable -- synchronizing one acts on
the other's bytes, and a neighbour's stale host copy can be written back over
what the device just produced. ``subview`` rejects layouts that would share a
line (see ``test_tensor_subview.py``); these tests check that the layouts it
does allow really are independent on hardware, which is the property that
rejection is protecting.

The neighbour test is the regression test for that guard: it fails on a build
where the alignment rule is dropped.
"""

import gc

import numpy as np
import pytest

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.utils.hostruntime.tensor_class import COHERENCE_GRANULE
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

_TILE = 16
_N = 256  # int32 elements the kernel transforms
_ADD = 7
_ELEM = np.dtype(np.int32).itemsize
_PAD = COHERENCE_GRANULE // _ELEM  # one granule, in int32 elements
# subview() offsets are bytes, while shapes stay in elements, so the same
# padding needs both spellings.
_PAD_BYTES = COHERENCE_GRANULE
_N_BYTES = _N * _ELEM


def _add_const_design(input_buf: In, output_buf: Out, N: CompileTime[int]):
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]

    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out):
        for _ in range_(N // _TILE):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(_TILE):
                elem_out[i] = elem_in[i] + _ADD
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])

    def sequence(a, b, in_h, out_h):
        in_h.fill(a)
        out_h.drain(b, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@iron.jit(N=_N)
def add_const(input_buf: In, output_buf: Out, *, N: CompileTime[int]):
    return _add_const_design(input_buf, output_buf, N=N)


@pytest.fixture(scope="module")
def source():
    tensor = XRTTensor((_N,), dtype=np.int32, device="npu")
    tensor[:] = np.arange(_N, dtype=np.int32)  # a mediated write: resyncs
    return tensor


@pytest.fixture(scope="module")
def expected(source):
    return np.arange(_N, dtype=np.int32) + _ADD


def test_device_writes_through_subview(source, expected):
    """A view is a real destination: the device writes the parent's storage."""
    parent = XRTTensor((_PAD + _N + _PAD,), dtype=np.int32, device="npu")
    parent.fill_(0)
    view = parent.subview(_PAD_BYTES, (_N,))

    add_const(source, view)
    view.to("cpu")

    np.testing.assert_array_equal(view.numpy(), expected)
    # The write landed inside the parent, at the view's offset.
    parent.device = "npu"
    parent.to("cpu")
    np.testing.assert_array_equal(parent.numpy()[_PAD : _PAD + _N], expected)


def test_subview_result_matches_a_standalone_buffer(source, expected):
    """Being a view changes nothing about the result."""
    standalone = XRTTensor((_N,), dtype=np.int32, device="npu")
    add_const(source, standalone)
    standalone.to("cpu")

    parent = XRTTensor((_PAD + _N + _PAD,), dtype=np.int32, device="npu")
    parent.fill_(0)
    view = parent.subview(_PAD_BYTES, (_N,))
    add_const(source, view)
    view.to("cpu")

    np.testing.assert_array_equal(view.numpy(), standalone.numpy())


def test_subview_sync_does_not_disturb_neighbours(source, expected):
    """The guard's regression test: neighbours and view survive each other.

    The bytes on either side of the view are written by the host and left
    unsynchronized, so they are dirty in the host's cache while the device
    produces the view's contents. Synchronizing the view must not write those
    dirty bytes back over the device's output, and must not discard them.

    A layout that shared a cache line would make this racy rather than wrong
    every time: the view reads back its pre-run contents only when the shared
    line is still dirty at the moment of the sync, so unrelated host activity
    that happens to evict it hides the damage. That is the argument for
    rejecting such layouts outright rather than testing for them -- an
    intermittent silent corruption is not something a test suite can be relied
    on to catch.
    """
    parent = XRTTensor((_PAD + _N + _PAD,), dtype=np.int32, device="npu")
    parent.fill_(0)
    view = parent.subview(_PAD_BYTES, (_N,))
    left = parent.subview(0, (_PAD,))
    right = parent.subview(_PAD_BYTES + _N_BYTES, (_PAD,))

    # Host-owned bytes either side, deliberately never synced to the device.
    left_pattern = np.full(_PAD, 0x0A0A0A0A, dtype=np.int32)
    right_pattern = np.full(_PAD, 0x0B0B0B0B, dtype=np.int32)
    left.data[:] = left_pattern
    right.data[:] = right_pattern

    add_const(source, view)
    view.to("cpu")

    np.testing.assert_array_equal(view.numpy(), expected)
    np.testing.assert_array_equal(left.data, left_pattern)
    np.testing.assert_array_equal(right.data, right_pattern)


def test_view_outlives_the_caller_s_reference_to_its_parent(source, expected):
    """The view owns the storage's lifetime, not the caller's variable.

    On this backend the parent holds the XRT buffer object the view's sub-buffer
    is derived from, so a view that failed to keep it alive would be reading
    freed device memory rather than merely getting a wrong answer.
    """
    parent = XRTTensor((_PAD + _N + _PAD,), dtype=np.int32, device="npu")
    parent.fill_(0)
    view = parent.subview(_PAD_BYTES, (_N,))
    add_const(source, view)
    view.to("cpu")

    assert view.base is parent
    del parent
    gc.collect()

    np.testing.assert_array_equal(view.numpy(), expected)


def test_nested_subviews_address_the_root_buffer(source, expected):
    """A view of a view must resolve against the original allocation."""
    parent = XRTTensor((2 * _PAD + 2 * _N,), dtype=np.int32, device="npu")
    parent.fill_(0)
    outer = parent.subview(_PAD_BYTES, (_N + _PAD,))
    inner = outer.subview(0, (_N,))
    assert inner.base is parent

    add_const(source, inner)
    inner.to("cpu")

    np.testing.assert_array_equal(inner.numpy(), expected)
    parent.device = "npu"
    parent.to("cpu")
    # The write landed at the root offset the nesting implies, not at 0.
    np.testing.assert_array_equal(parent.numpy()[_PAD : _PAD + _N], expected)
    np.testing.assert_array_equal(parent.numpy()[:_PAD], np.zeros(_PAD, np.int32))


def test_adjacent_subviews_are_independent(source, expected):
    """Two views of one buffer, synced separately, do not interfere."""
    parent = XRTTensor((2 * _N,), dtype=np.int32, device="npu")
    parent.fill_(0)
    first = parent.subview(0, (_N,))
    second = parent.subview(_N_BYTES, (_N,))

    add_const(source, first)
    first.to("cpu")
    add_const(source, second)
    second.to("cpu")

    np.testing.assert_array_equal(first.numpy(), expected)
    np.testing.assert_array_equal(second.numpy(), expected)
