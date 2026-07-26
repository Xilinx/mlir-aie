# test_tensor_subview.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s

import gc

import numpy as np
import pytest

from aie.utils.hostruntime import tensor_class
from aie.utils.hostruntime.tensor_class import (
    COHERENCE_GRANULE,
    CPUOnlyTensor,
    NpuTensor,
    _detect_coherence_granule,
)

# 64 is the cache line on every architecture this runs on, and the floor the
# detector is not allowed to go below. Tests that only ever derive their sizes
# from the detected value cannot notice the detector itself regressing, so the
# alignment behaviour is pinned against this literal and only the cases that are
# genuinely about the detected value use COHERENCE_GRANULE.
GRANULE = 64
_F32_PER_GRANULE = GRANULE // 4


# ---------------------------------------------------------------------------
# Granule detection
# ---------------------------------------------------------------------------


def test_detected_granule_matches_the_literal_this_suite_assumes():
    assert COHERENCE_GRANULE == GRANULE


def test_granule_is_floored_when_the_host_reports_something_smaller(monkeypatch):
    """A small or absent report must make the check stricter, never weaker."""
    monkeypatch.setattr(tensor_class.os, "sysconf", lambda _: 8)
    assert _detect_coherence_granule(default=64) == 64


def test_granule_uses_a_larger_reported_line_size(monkeypatch):
    monkeypatch.setattr(tensor_class.os, "sysconf", lambda _: 128)
    assert _detect_coherence_granule(default=64) == 128


def test_granule_is_rounded_up_to_a_power_of_two(monkeypatch):
    """Alignment arithmetic assumes it; nothing in the reporting path promises it."""
    monkeypatch.setattr(tensor_class.os, "sysconf", lambda _: 96)
    assert _detect_coherence_granule(default=64) == 128


def test_granule_falls_back_when_no_source_reports(monkeypatch):
    """Windows has no os.sysconf at all; the floor is the answer there."""

    def no_sysconf(_):
        raise AttributeError("no sysconf on this platform")

    monkeypatch.setattr(tensor_class.os, "sysconf", no_sysconf)
    monkeypatch.setattr(tensor_class, "_read_sysfs_line_size", lambda: None)
    assert _detect_coherence_granule(default=64) == 64


def test_granule_falls_back_to_sysfs_when_sysconf_is_unavailable(monkeypatch):
    def no_sysconf(_):
        raise ValueError("unrecognized configuration name")

    monkeypatch.setattr(tensor_class.os, "sysconf", no_sysconf)
    monkeypatch.setattr(tensor_class, "_read_sysfs_line_size", lambda: 128)
    assert _detect_coherence_granule(default=64) == 128


def _write_cache_index(cache_dir, index, level, kind, line_size):
    entry = cache_dir / f"index{index}"
    entry.mkdir()
    (entry / "level").write_text(f"{level}\n")
    (entry / "type").write_text(f"{kind}\n")
    (entry / "coherency_line_size").write_text(f"{line_size}\n")


def test_sysfs_scan_selects_the_l1_data_cache_not_the_first_entry(tmp_path):
    """Cache indices are not ordered by level or type.

    A machine that enumerates the instruction cache first would otherwise
    report its line size, which has nothing to do with data coherence.
    """
    _write_cache_index(tmp_path, 0, level=1, kind="Instruction", line_size=32)
    _write_cache_index(tmp_path, 1, level=1, kind="Data", line_size=128)
    _write_cache_index(tmp_path, 2, level=2, kind="Unified", line_size=256)
    assert tensor_class._read_sysfs_line_size(cache_dir=str(tmp_path)) == 128


def test_sysfs_scan_accepts_a_unified_l1(tmp_path):
    _write_cache_index(tmp_path, 0, level=1, kind="Unified", line_size=128)
    assert tensor_class._read_sysfs_line_size(cache_dir=str(tmp_path)) == 128


def test_sysfs_scan_reports_nothing_when_there_is_no_l1_data_cache(tmp_path):
    _write_cache_index(tmp_path, 0, level=2, kind="Unified", line_size=128)
    assert tensor_class._read_sysfs_line_size(cache_dir=str(tmp_path)) is None


def test_enforced_granule_follows_the_module_value(monkeypatch):
    """The class must not freeze the value at import time.

    If it did, the alignment rule could not be corrected or tested without
    reimporting, and a backend could not raise it.
    """
    monkeypatch.setattr(tensor_class, "COHERENCE_GRANULE", 128)
    assert NpuTensor._resolve_coherence_granule() == 128
    parent = CPUOnlyTensor((512,), dtype=np.uint8)
    with pytest.raises(ValueError, match="coherence granule"):
        parent.subview(64, (64,))  # legal at 64, not at 128


def test_a_backend_may_require_a_coarser_granule():
    class CoarseTensor(CPUOnlyTensor):
        _coherence_granule = 256

    parent = CoarseTensor((1024,), dtype=np.uint8)
    with pytest.raises(ValueError, match="256-byte coherence granule"):
        parent.subview(64, (64,))
    assert parent.subview(256, (256,)).shape == (256,)


# ---------------------------------------------------------------------------
# Aliasing: a view must be a view, in both directions
# ---------------------------------------------------------------------------


def test_subview_shares_storage():
    parent = CPUOnlyTensor((4, _F32_PER_GRANULE), dtype=np.float32)
    view = parent.subview(_F32_PER_GRANULE, (_F32_PER_GRANULE,))  # row 1
    assert view.shape == (_F32_PER_GRANULE,)
    view.data[:] = 3.0
    assert np.allclose(parent.data[1], 3.0)
    assert np.allclose(parent.data[0], 0.0)


def test_parent_writes_are_visible_through_the_view():
    """The other direction of the aliasing contract."""
    parent = CPUOnlyTensor((2 * GRANULE,), dtype=np.uint8)
    view = parent.subview(GRANULE, (GRANULE,))
    parent.data[GRANULE:] = 0x5A
    assert np.array_equal(view.data, np.full(GRANULE, 0x5A, dtype=np.uint8))


def test_subview_does_not_copy():
    """A copy would satisfy every read-only assertion but silently break writes."""
    parent = CPUOnlyTensor((2 * GRANULE,), dtype=np.uint8)
    view = parent.subview(0, (GRANULE,))
    assert np.shares_memory(view.data, parent.data)


def test_subview_nested():
    parent = CPUOnlyTensor((4 * _F32_PER_GRANULE,), dtype=np.float32)
    outer = parent.subview(_F32_PER_GRANULE, (2 * _F32_PER_GRANULE,))
    inner = outer.subview(_F32_PER_GRANULE, (_F32_PER_GRANULE,))
    inner.data[:] = 7.0
    assert np.allclose(parent.data[2 * _F32_PER_GRANULE : 3 * _F32_PER_GRANULE], 7.0)
    assert np.allclose(parent.data[: 2 * _F32_PER_GRANULE], 0.0)
    assert np.allclose(parent.data[3 * _F32_PER_GRANULE :], 0.0)


def test_subview_tiling_is_expressible():
    """The intended use: carving a buffer into independently synced windows."""
    n_windows, window_bytes = 4, 2 * GRANULE
    parent = CPUOnlyTensor((n_windows * window_bytes,), dtype=np.uint8)
    windows = [
        parent.subview(i * window_bytes, (window_bytes,)) for i in range(n_windows)
    ]
    for i, window in enumerate(windows):
        window.data[:] = i + 1
    for i in range(n_windows):
        start = i * window_bytes
        expected = np.full(window_bytes, i + 1, dtype=np.uint8)
        assert np.array_equal(parent.data[start : start + window_bytes], expected)


# ---------------------------------------------------------------------------
# Ownership and lifetime
# ---------------------------------------------------------------------------


def test_base_reports_the_owning_buffer():
    parent = CPUOnlyTensor((4 * GRANULE,), dtype=np.uint8)
    view = parent.subview(GRANULE, (GRANULE,))
    assert parent.base is None
    assert not parent.is_view
    assert view.base is parent
    assert view.is_view


def test_storage_offset_is_measured_from_the_owner():
    """Nested offsets accumulate, so a view always locates itself in the owner.

    The backends need this to address the underlying allocation directly rather
    than through the intermediate view.
    """
    parent = CPUOnlyTensor((8 * GRANULE,), dtype=np.uint8)
    outer = parent.subview(2 * GRANULE, (4 * GRANULE,))
    inner = outer.subview(GRANULE, (GRANULE,))
    assert parent.storage_offset == 0
    assert outer.storage_offset == 2 * GRANULE
    assert inner.storage_offset == 3 * GRANULE  # not GRANULE
    # And it agrees with where a write actually lands.
    inner.data[:] = 0x33
    assert np.nonzero(parent.data != 0)[0].min() == inner.storage_offset


def test_base_collapses_a_chain_of_views():
    """Like numpy: the base of a view of a view is the real owner."""
    parent = CPUOnlyTensor((8 * GRANULE,), dtype=np.uint8)
    outer = parent.subview(GRANULE, (4 * GRANULE,))
    inner = outer.subview(GRANULE, (GRANULE,))
    assert inner.base is parent


def test_view_keeps_its_parent_alive():
    """Dropping the caller's reference to the parent must not free the storage."""
    parent = CPUOnlyTensor((2 * GRANULE,), dtype=np.uint8)
    parent.data[:] = 0x11
    view = parent.subview(GRANULE, (GRANULE,))
    del parent
    gc.collect()
    assert view.base is not None  # the view still holds the owner
    assert np.array_equal(view.data, np.full(GRANULE, 0x11, dtype=np.uint8))


def test_view_has_the_backend_type_of_its_parent():
    """A subclass must not be silently downcast to the backend it derives from."""

    class DerivedTensor(CPUOnlyTensor):
        pass

    parent = DerivedTensor((2 * GRANULE,), dtype=np.uint8)
    assert type(parent.subview(0, (GRANULE,))) is DerivedTensor


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


def test_subview_dtype_reinterpret():
    parent = CPUOnlyTensor((GRANULE,), dtype=np.uint8)
    view = parent.subview(0, (GRANULE // 4,), dtype=np.uint32)
    assert view.shape == (GRANULE // 4,)
    assert view.nbytes == parent.nbytes
    assert view.dtype == np.dtype(np.uint32)
    assert view.data.dtype == np.dtype(np.uint32)


def test_subview_dtype_reinterpret_of_the_same_width():
    """nbytes cannot distinguish these, so assert the dtype itself."""
    parent = CPUOnlyTensor((GRANULE // 4,), dtype=np.uint32)
    view = parent.subview(0, (GRANULE // 4,), dtype=np.float32)
    assert view.dtype == np.dtype(np.float32)
    assert view.data.dtype == np.dtype(np.float32)


def test_offset_counts_elements_of_the_parent_dtype():
    """The one case where the offset unit is observable.

    offset is in the parent's elements while shape is in the view's, so a
    reinterpreting subview at a nonzero offset is the only call that can tell
    the two units apart. At offset 0 every possible unit agrees.
    """
    parent = CPUOnlyTensor((GRANULE,), dtype=np.uint32)  # 4 * GRANULE bytes
    view = parent.subview(GRANULE // 4, (GRANULE,), dtype=np.uint8)
    view.data[:] = 0xEE
    written = np.nonzero(parent.data.view(np.uint8) != 0)[0]
    assert written.min() == GRANULE  # (GRANULE // 4) elements * 4 bytes
    assert written.max() == 2 * GRANULE - 1


def test_subview_accepts_a_multidimensional_shape():
    parent = CPUOnlyTensor((4 * GRANULE,), dtype=np.uint8)
    view = parent.subview(0, (2, GRANULE))
    assert view.shape == (2, GRANULE)
    view.data[1, :] = 9
    assert np.array_equal(parent.data[GRANULE : 2 * GRANULE], np.full(GRANULE, 9))
    assert np.array_equal(parent.data[:GRANULE], np.zeros(GRANULE, dtype=np.uint8))


def test_subview_accepts_a_bare_int_as_a_length():
    parent = CPUOnlyTensor((2 * GRANULE,), dtype=np.uint8)
    assert parent.subview(0, GRANULE).shape == (GRANULE,)


def test_subview_rejects_a_negative_extent():
    """A negative extent understates nbytes, so every size check would pass."""
    parent = CPUOnlyTensor((4 * GRANULE,), dtype=np.uint8)
    with pytest.raises(ValueError, match="must not be negative"):
        parent.subview(0, (-GRANULE,))


def test_subview_rejects_an_overflowing_shape():
    """Extents whose product wraps int64 must not defeat the bounds check."""
    parent = CPUOnlyTensor((4 * GRANULE,), dtype=np.uint8)
    with pytest.raises(ValueError, match="out of bounds"):
        parent.subview(0, (1099511627779, 2049606955414760192))


def test_subview_rejects_a_non_integer_offset():
    parent = CPUOnlyTensor((4 * GRANULE,), dtype=np.uint8)
    with pytest.raises(TypeError, match="must be an integer"):
        parent.subview(1.5, (GRANULE,))
    with pytest.raises(TypeError, match="must be an integer"):
        parent.subview(-0.5, (GRANULE,))  # would truncate to a legal 0


def test_subview_rejects_a_non_integer_extent():
    parent = CPUOnlyTensor((4 * GRANULE,), dtype=np.uint8)
    with pytest.raises(TypeError, match="sequence of integers"):
        parent.subview(0, (GRANULE, "x"))


def test_subview_accepts_an_empty_region():
    parent = CPUOnlyTensor((GRANULE,), dtype=np.uint8)
    assert parent.subview(0, (0,)).shape == (0,)


# ---------------------------------------------------------------------------
# Bounds and alignment
# ---------------------------------------------------------------------------


def test_subview_bounds_checked():
    parent = CPUOnlyTensor((4 * _F32_PER_GRANULE,), dtype=np.float32)
    with pytest.raises(ValueError, match="out of bounds"):
        parent.subview(_F32_PER_GRANULE, (4 * _F32_PER_GRANULE,))
    with pytest.raises(ValueError, match="out of bounds"):
        parent.subview(-_F32_PER_GRANULE, (_F32_PER_GRANULE,))
    # The boundary case, on a single-byte dtype so that offset -1 is -1 byte:
    # a wider dtype scales it away from the boundary and stops testing it.
    single_byte = CPUOnlyTensor((4 * GRANULE,), dtype=np.uint8)
    with pytest.raises(ValueError, match="out of bounds"):
        single_byte.subview(-1, (GRANULE,))


def test_subview_rejects_unaligned_offset():
    """A view starting mid-line shares that line with its predecessor."""
    parent = CPUOnlyTensor((4 * GRANULE,), dtype=np.uint8)
    for offset in (1, 4, 32, GRANULE - 1):
        with pytest.raises(ValueError, match="coherence granule"):
            parent.subview(offset, (GRANULE,))


def test_subview_rejects_unaligned_length():
    """A view ending mid-line shares that line with its successor."""
    parent = CPUOnlyTensor((4 * GRANULE,), dtype=np.uint8)
    for length in (1, GRANULE // 2, GRANULE + 1):
        with pytest.raises(ValueError, match="coherence granule"):
            parent.subview(0, (length,))


def test_subview_allows_unaligned_tail_at_end_of_buffer():
    """The final view's last line is shared with no sibling, so it is allowed.

    This also keeps a whole-buffer view legal when the buffer's own size is not
    a multiple of the granule.
    """
    parent = CPUOnlyTensor((GRANULE + 8,), dtype=np.uint8)
    assert parent.subview(0, (GRANULE + 8,)).nbytes == parent.nbytes
    tail = parent.subview(GRANULE, (8,))
    tail.data[:] = 5
    assert np.array_equal(parent.data[GRANULE:], np.full(8, 5, dtype=np.uint8))


def test_alignment_composes_through_nested_views():
    """A legal view of a legal view must still be aligned in the root buffer."""
    parent = CPUOnlyTensor((8 * GRANULE,), dtype=np.uint8)
    outer = parent.subview(2 * GRANULE, (4 * GRANULE,))
    inner = outer.subview(GRANULE, (GRANULE,))
    inner.data[:] = 0x77
    written = np.nonzero(parent.data != 0)[0]
    assert written.min() % GRANULE == 0
    assert written.min() == 3 * GRANULE
    assert written.max() == 4 * GRANULE - 1


# ---------------------------------------------------------------------------
# Backend contract
# ---------------------------------------------------------------------------


def test_backend_without_subview_says_so():
    """The capability answer must not arrive dressed as an alignment complaint."""

    class NoSubviewTensor(NpuTensor):
        @property
        def data(self):
            return np.zeros(4, dtype=np.uint8)

        @property
        def shape(self):
            return (4,)

        def _sync_to_device(self):
            pass

        def _sync_from_device(self):
            pass

    tensor = NoSubviewTensor((4,), dtype=np.uint8)
    with pytest.raises(NotImplementedError, match="NoSubviewTensor"):
        tensor.subview(1, (3,))


def test_the_backend_hook_receives_a_byte_offset_and_the_view_dtype():
    """Pin what subview() forwards, independently of any backend's storage."""
    seen = {}

    class RecordingTensor(CPUOnlyTensor):
        def _subview(self, offset_bytes, shape, dtype):
            seen.update(offset_bytes=offset_bytes, shape=shape, dtype=dtype)
            return super()._subview(offset_bytes, shape, dtype)

    parent = RecordingTensor((4 * GRANULE,), dtype=np.uint32)
    parent.subview(GRANULE // 4, (GRANULE,), dtype=np.uint8)
    assert seen["offset_bytes"] == GRANULE  # elements * parent itemsize
    assert seen["shape"] == (GRANULE,)
    assert seen["dtype"] == np.dtype(np.uint8)


# ---------------------------------------------------------------------------
# The buffer holds the invariant, not just the view constructor
# ---------------------------------------------------------------------------


def test_buffer_refuses_a_state_boundary_inside_a_granule():
    """Reaching past subview() does not get you a map that cannot be maintained.

    subview() rejects a misaligned region up front, but it is not the only way
    to record state, and a future path that marks a sub-range written would not
    go through it. The rule belongs where the state is.
    """
    buffer = tensor_class.CPUBuffer(4 * GRANULE, "npu", GRANULE)
    with pytest.raises(ValueError, match="coherence granule"):
        buffer.coherence.set(0, GRANULE + 8, "cpu")


def test_buffer_allows_an_aligned_state_boundary():
    buffer = tensor_class.CPUBuffer(4 * GRANULE, "npu", GRANULE)
    buffer.coherence.set(0, GRANULE, "cpu")
    assert buffer.coherence.get(0, GRANULE) == "cpu"
    assert buffer.coherence.get(GRANULE, 2 * GRANULE) == "npu"


def test_buffer_allows_a_final_region_to_end_anywhere():
    """Nothing follows the last region, so its end shares a granule with nobody."""
    buffer = tensor_class.CPUBuffer(GRANULE + 8, "npu", GRANULE)
    buffer.coherence.set(GRANULE, GRANULE + 8, "cpu")
    assert buffer.coherence.get(GRANULE, GRANULE + 8) == "cpu"


def test_a_backend_granule_governs_its_buffer():
    buffer = tensor_class.CPUBuffer(8 * GRANULE, "npu", 4 * GRANULE)
    with pytest.raises(ValueError, match="256-byte coherence granule"):
        buffer.coherence.set(0, GRANULE, "cpu")
    buffer.coherence.set(0, 4 * GRANULE, "cpu")  # on its own granule, fine
