# test_tensor_subview.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s

import numpy as np
import pytest

from aie.utils.hostruntime.tensor_class import COHERENCE_GRANULE, CPUOnlyTensor

# Element counts that make the byte arithmetic below granule-aligned.
_G = COHERENCE_GRANULE
_F32_PER_GRANULE = _G // 4


def test_subview_shares_storage():
    parent = CPUOnlyTensor((4, _F32_PER_GRANULE), dtype=np.float32)
    view = parent.subview(_F32_PER_GRANULE, (_F32_PER_GRANULE,))  # row 1
    assert view.shape == (_F32_PER_GRANULE,)
    view.data[:] = 3.0
    # The view aliases the parent's storage, so the write lands in the parent,
    # and sibling regions are untouched.
    assert np.allclose(parent.data[1], 3.0)
    assert np.allclose(parent.data[0], 0.0)


def test_subview_dtype_reinterpret():
    parent = CPUOnlyTensor((_G,), dtype=np.uint8)
    view = parent.subview(0, (_G // 4,), dtype=np.uint32)  # same bytes, as u32
    assert view.shape == (_G // 4,)
    assert view.nbytes == parent.nbytes


def test_subview_nested():
    parent = CPUOnlyTensor((4 * _F32_PER_GRANULE,), dtype=np.float32)
    outer = parent.subview(_F32_PER_GRANULE, (2 * _F32_PER_GRANULE,))
    inner = outer.subview(_F32_PER_GRANULE, (_F32_PER_GRANULE,))
    inner.data[:] = 7.0
    # Offsets compose: inner is row 2 of the parent, and rows 0/1/3 are intact.
    assert np.allclose(parent.data[2 * _F32_PER_GRANULE : 3 * _F32_PER_GRANULE], 7.0)
    assert np.allclose(parent.data[: 2 * _F32_PER_GRANULE], 0.0)
    assert np.allclose(parent.data[3 * _F32_PER_GRANULE :], 0.0)


def test_subview_bounds_checked():
    parent = CPUOnlyTensor((4 * _F32_PER_GRANULE,), dtype=np.float32)
    with pytest.raises(ValueError, match="out of bounds"):
        parent.subview(_F32_PER_GRANULE, (4 * _F32_PER_GRANULE,))
    with pytest.raises(ValueError, match="out of bounds"):
        parent.subview(-_F32_PER_GRANULE, (_F32_PER_GRANULE,))


def test_subview_rejects_unaligned_offset():
    """A view starting mid-line shares that line with its predecessor."""
    parent = CPUOnlyTensor((4 * _G,), dtype=np.uint8)
    with pytest.raises(ValueError, match="coherence granule"):
        parent.subview(1, (_G,))
    with pytest.raises(ValueError, match="coherence granule"):
        parent.subview(_G - 1, (_G,))


def test_subview_rejects_unaligned_length():
    """A view ending mid-line shares that line with its successor."""
    parent = CPUOnlyTensor((4 * _G,), dtype=np.uint8)
    with pytest.raises(ValueError, match="coherence granule"):
        parent.subview(0, (_G + 1,))
    with pytest.raises(ValueError, match="coherence granule"):
        parent.subview(0, (_G // 2,))


def test_subview_allows_unaligned_tail_at_end_of_buffer():
    """The final view's last line is shared with no sibling, so it is allowed.

    This also keeps a whole-buffer view legal when the buffer's own size is not
    a multiple of the granule.
    """
    parent = CPUOnlyTensor((_G + 8,), dtype=np.uint8)
    whole = parent.subview(0, (_G + 8,))
    assert whole.nbytes == parent.nbytes
    tail = parent.subview(_G, (8,))
    tail.data[:] = 5
    assert np.array_equal(parent.data[_G:], np.full(8, 5, dtype=np.uint8))


def test_subview_tiling_is_expressible():
    """The intended use -- carving a buffer into independently synced windows."""
    n_windows, window_bytes = 4, 2 * _G
    parent = CPUOnlyTensor((n_windows * window_bytes,), dtype=np.uint8)
    windows = [
        parent.subview(i * window_bytes, (window_bytes,)) for i in range(n_windows)
    ]
    for i, w in enumerate(windows):
        w.data[:] = i + 1
    for i in range(n_windows):
        expected = np.full(window_bytes, i + 1, dtype=np.uint8)
        start = i * window_bytes
        assert np.array_equal(parent.data[start : start + window_bytes], expected)


def test_coherence_granule_is_sane():
    """Whatever we detect, never trust it below a cache line."""
    assert COHERENCE_GRANULE >= 64
    assert COHERENCE_GRANULE & (COHERENCE_GRANULE - 1) == 0  # power of two
