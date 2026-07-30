# test_tensor_subview.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s

import gc
import os

import numpy as np
import pytest

from aie.utils.hostruntime import coherence, tensor_class
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
    monkeypatch.setattr(coherence, "_read_sysfs_line_size", lambda: 8)
    assert _detect_coherence_granule(default=64) == 64


def test_granule_uses_a_larger_reported_line_size(monkeypatch):
    monkeypatch.setattr(coherence, "_read_sysfs_line_size", lambda: 128)
    assert _detect_coherence_granule(default=64) == 128


def test_granule_is_rounded_up_to_a_power_of_two(monkeypatch):
    """Alignment arithmetic assumes it; nothing in the reporting path promises it."""
    monkeypatch.setattr(coherence, "_read_sysfs_line_size", lambda: 96)
    assert _detect_coherence_granule(default=64) == 128


def test_granule_falls_back_when_no_source_reports(monkeypatch):
    """Windows has no sysfs cache directory; the floor is the answer there."""
    monkeypatch.setattr(coherence, "_read_sysfs_line_size", lambda: None)
    assert _detect_coherence_granule(default=64) == 64


def test_sysconf_cannot_report_the_line_size_on_this_interpreter():
    """The detector reads sysfs rather than asking sysconf, and this is why.

    ``_SC_LEVEL1_DCACHE_LINESIZE`` is a glibc extension, not POSIX, and CPython
    does not carry it in ``os.sysconf_names``. Asking for it by name therefore
    raises on every platform: ValueError where sysconf exists, AttributeError on
    Windows where it does not. If a future interpreter did grow the name, this
    is the test that says the detector could start preferring it.
    """
    assert "SC_LEVEL1_DCACHE_LINESIZE" not in getattr(os, "sysconf_names", {})


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
    assert coherence._read_sysfs_line_size(cache_dir=str(tmp_path)) == 128


def test_sysfs_scan_accepts_a_unified_l1(tmp_path):
    _write_cache_index(tmp_path, 0, level=1, kind="Unified", line_size=128)
    assert coherence._read_sysfs_line_size(cache_dir=str(tmp_path)) == 128


def test_sysfs_scan_reports_nothing_when_there_is_no_l1_data_cache(tmp_path):
    _write_cache_index(tmp_path, 0, level=2, kind="Unified", line_size=128)
    assert coherence._read_sysfs_line_size(cache_dir=str(tmp_path)) is None


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
    view = parent.subview(GRANULE, (_F32_PER_GRANULE,))  # row 1, one granule in
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
    outer = parent.subview(GRANULE, (2 * _F32_PER_GRANULE,))
    inner = outer.subview(GRANULE, (_F32_PER_GRANULE,))
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


def test_offset_counts_bytes_whatever_the_dtypes_are():
    """The one case where the offset unit is observable.

    A reinterpreting subview at a nonzero offset is the only call that can tell
    the candidate units apart, since at offset 0 all of them agree. Bytes is the
    unit the buffer is measured in: it has no dtype, the alignment rule is in
    bytes, and storage_offset reports bytes.
    """
    parent = CPUOnlyTensor((GRANULE,), dtype=np.uint32)  # 4 * GRANULE bytes
    view = parent.subview(GRANULE, (GRANULE,), dtype=np.uint8)
    view.data[:] = 0xEE
    written = np.nonzero(parent.data.view(np.uint8) != 0)[0]
    assert written.min() == GRANULE
    assert written.max() == 2 * GRANULE - 1


def test_offset_does_not_scale_with_the_parent_dtype():
    """Same byte offset, same place, whatever the parent is typed as.

    Under the old parent-dtype rule these two landed 4x apart.
    """
    as_bytes = CPUOnlyTensor((4 * GRANULE,), dtype=np.uint8)
    as_words = CPUOnlyTensor((GRANULE,), dtype=np.uint32)
    assert as_bytes.subview(GRANULE, (GRANULE,)).storage_offset == GRANULE
    assert as_words.subview(GRANULE, (GRANULE // 4,)).storage_offset == GRANULE


def test_offset_agrees_with_storage_offset():
    """The argument in and the value reported back are the same number."""
    parent = CPUOnlyTensor((GRANULE,), dtype=np.uint32)
    assert parent.subview(2 * GRANULE, (GRANULE,), dtype=np.uint8).storage_offset == (
        2 * GRANULE
    )


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
        parent.subview(GRANULE, (4 * _F32_PER_GRANULE,))
    with pytest.raises(ValueError, match="out of bounds"):
        parent.subview(-GRANULE, (_F32_PER_GRANULE,))
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
    parent.subview(GRANULE, (GRANULE,), dtype=np.uint8)
    assert seen["offset_bytes"] == GRANULE  # bytes, unscaled by either dtype
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
    buffer = _host_storage(4 * GRANULE, "npu", GRANULE)
    with pytest.raises(ValueError, match="coherence granule"):
        buffer.coherence.set(0, GRANULE + 8, "cpu")


def test_buffer_allows_an_aligned_state_boundary():
    buffer = _host_storage(4 * GRANULE, "npu", GRANULE)
    buffer.coherence.set(0, GRANULE, "cpu")
    assert buffer.coherence.get(0, GRANULE) == "cpu"
    assert buffer.coherence.get(GRANULE, 2 * GRANULE) == "npu"


def test_buffer_allows_a_final_region_to_end_anywhere():
    """Nothing follows the last region, so its end shares a granule with nobody."""
    buffer = _host_storage(GRANULE + 8, "npu", GRANULE)
    buffer.coherence.set(GRANULE, GRANULE + 8, "cpu")
    assert buffer.coherence.get(GRANULE, GRANULE + 8) == "cpu"


def test_a_backend_granule_governs_its_buffer():
    buffer = _host_storage(8 * GRANULE, "npu", 4 * GRANULE)
    with pytest.raises(ValueError, match="256-byte coherence granule"):
        buffer.coherence.set(0, GRANULE, "cpu")
    buffer.coherence.set(0, 4 * GRANULE, "cpu")  # on its own granule, fine


def test_overwrite_records_the_write_without_reading_first():
    """The saving is the reconcile that mutate() does on entry."""
    parent = CPUOnlyTensor((4 * GRANULE,), dtype=np.uint8)
    parent.device = "cpu"
    with parent.overwrite() as buf:
        buf[:] = 0x31
    assert parent.device == "cpu"
    assert np.array_equal(parent.data, np.full(4 * GRANULE, 0x31, dtype=np.uint8))


def test_overwrite_retires_its_borrow_like_mutate():
    parent = CPUOnlyTensor((GRANULE,), dtype=np.uint8)
    with parent.overwrite() as buf:
        buf[:] = 1
    with pytest.raises(ValueError):
        buf[0] = 2


def test_mutate_still_reconciles_before_handing_over():
    """A partial update has to see what is already there."""
    parent = CPUOnlyTensor((2 * GRANULE,), dtype=np.uint8)
    with parent.mutate() as buf:
        buf[:] = 0x77
    with parent.mutate() as buf:
        buf[:GRANULE] = 0x88  # only half; the rest must survive
    assert np.array_equal(parent.data[GRANULE:], np.full(GRANULE, 0x77, dtype=np.uint8))


def test_regions_written_forwards_collapse_back_into_one():
    """Coalescing is what keeps a reconcile from costing a transfer per window.

    Without it the map stays fragmented for the life of the buffer: every read
    of it walks spans, and handing the buffer to the device sends each window
    separately instead of the run they now form.
    """
    buffer = _host_storage(4 * GRANULE, "npu", GRANULE)
    for i in range(4):
        buffer.coherence.set(i * GRANULE, (i + 1) * GRANULE, "cpu")
    assert buffer.coherence.uniform == "cpu"
    assert buffer.coherence.ranges(0, 4 * GRANULE, "cpu") == [(0, 4 * GRANULE)]


def test_regions_written_backwards_collapse_too():
    """The same, merging onto the region that follows rather than precedes."""
    buffer = _host_storage(4 * GRANULE, "npu", GRANULE)
    for i in reversed(range(4)):
        buffer.coherence.set(i * GRANULE, (i + 1) * GRANULE, "cpu")
    assert buffer.coherence.uniform == "cpu"
    assert buffer.coherence.ranges(0, 4 * GRANULE, "cpu") == [(0, 4 * GRANULE)]


def test_a_written_window_is_one_range_not_several():
    """Two windows either side of an untouched one stay two ranges, not four."""
    buffer = _host_storage(8 * GRANULE, "npu", GRANULE)
    buffer.coherence.set(0, 2 * GRANULE, "cpu")
    buffer.coherence.set(4 * GRANULE, 6 * GRANULE, "cpu")
    assert buffer.coherence.ranges(0, 8 * GRANULE, "cpu") == [
        (0, 2 * GRANULE),
        (4 * GRANULE, 6 * GRANULE),
    ]


# ---------------------------------------------------------------------------
# The whole-tensor methods, over a buffer whose regions disagree
# ---------------------------------------------------------------------------


def _host_storage(nbytes, device, granule=None):
    """A host-only storage, the spelling several tests below want."""
    return tensor_class.Storage(
        tensor_class.HostOnlyTransport(nbytes), nbytes, device, granule
    )


class TwoMemoryTransport(tensor_class.Transport):
    """Host and device bytes as separate arrays, so a missed reconcile shows.

    ``CPUOnlyTensor`` cannot demonstrate any of this: its transfers are no-ops
    over one array, so a skipped reconcile is indistinguishable from a correct
    one. Here the two sides are different memory and only a transfer moves a
    byte between them, which is what makes an omitted transfer observable.
    """

    def __init__(self, nbytes):
        self._host = np.zeros(nbytes, dtype=np.uint8)
        self.device_bytes = np.zeros(nbytes, dtype=np.uint8)

    @property
    def host_bytes(self):
        return self._host

    def to_device(self, offset, nbytes):
        end = offset + nbytes
        self.device_bytes[offset:end] = self._host[offset:end]

    def from_device(self, offset, nbytes):
        end = offset + nbytes
        self._host[offset:end] = self.device_bytes[offset:end]


class TwoMemoryTensor(NpuTensor):
    """A minimal backend over :class:`TwoMemoryTransport`."""

    DEVICES = ["cpu", "npu"]
    DEFAULT_DEVICE = "npu"

    def __init__(self, shape, dtype=np.uint8, device="npu"):
        super().__init__(shape, dtype=dtype, device=device)
        self._shape = tuple(shape)
        nbytes = int(np.prod(self._shape)) * np.dtype(dtype).itemsize
        self._transport = TwoMemoryTransport(nbytes)
        self._storage = tensor_class.Storage(
            self._transport,
            nbytes,
            self._initial_device,
            self._resolve_coherence_granule(),
        )
        self._offset_bytes = 0
        self._data = self._storage.host_bytes.view(dtype).reshape(self._shape)

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape

    def _sync_to_device(self):
        start, end = self._extent
        self._storage.sync_to_device(start, end - start)

    def _sync_from_device(self):
        start, end = self._extent
        self._storage.sync_from_device(start, end - start)

    def _subview(self, offset_bytes, shape, dtype):
        nbytes = int(np.prod(shape)) * dtype.itemsize
        view = type(self).__new__(type(self))
        NpuTensor.__init__(view, shape, dtype=dtype, device=self.device)
        view._storage = self
        view._shape = tuple(shape)
        view._storage = self._storage
        absolute = self.storage_offset + offset_bytes
        view._offset_bytes = absolute
        view._data = (
            self._storage.host_bytes[absolute : absolute + nbytes]
            .view(dtype)
            .reshape(view._shape)
        )
        return view


def fragmented(fill=0xAA):
    """A tensor whose first window is host-written and whose second is not.

    This is the state an ordinary caller reaches by writing one window through
    a view and then touching the enclosing tensor before reconciling.
    """
    tensor = TwoMemoryTensor((2 * GRANULE,))
    tensor.storage._transport.device_bytes[:] = fill
    tensor.storage.host_bytes[:] = fill
    with tensor.subview(0, (GRANULE,)).mutate() as window:
        window[:] = 0x11
    return tensor


def test_the_premise_a_written_window_leaves_the_extent_mixed():
    """Guard for the tests below: they mean nothing if the state is uniform."""
    tensor = fragmented()
    assert tensor.storage.coherence.uniform is None
    assert tensor.device == "cpu"


def test_fill_reaches_the_device_even_where_the_extent_was_device_held():
    """``fill_`` writes every byte, so every byte must arrive.

    The tensor reports ``cpu`` because part of it is host-written, and a guard
    that reads that as "nothing to send" skips the region that is still
    device-held, losing the bytes ``fill_`` just wrote there.
    """
    tensor = fragmented()
    tensor.fill_(0x22)
    tensor.to("npu")
    assert (tensor.storage._transport.device_bytes == 0x22).all()


def test_setitem_reaches_the_device_even_where_the_extent_was_device_held():
    tensor = fragmented()
    tensor[GRANULE:] = 0x33
    tensor.to("npu")
    assert (tensor.storage._transport.device_bytes[GRANULE:] == 0x33).all()


def test_numpy_sees_device_writes_to_a_region_it_does_not_hold():
    """A read must reconcile the regions the device holds, not skip them.

    The mirror of the write case: the extent reports ``cpu``, so a guard that
    reads that as "already current" returns the stale host copy of the window
    the device actually owns.
    """
    tensor = fragmented()
    tensor.storage._transport.device_bytes[GRANULE:] = 0xBB
    assert (tensor.numpy()[GRANULE:] == 0xBB).all()


def test_getitem_sees_device_writes_to_a_region_it_does_not_hold():
    tensor = fragmented()
    tensor.storage._transport.device_bytes[GRANULE:] = 0xCC
    assert tensor[GRANULE] == 0xCC


def test_array_sees_device_writes_to_a_region_it_does_not_hold():
    tensor = fragmented()
    tensor.storage._transport.device_bytes[GRANULE:] = 0xDD
    assert (np.asarray(tensor)[GRANULE:] == 0xDD).all()


def test_a_uniform_tensor_still_takes_the_cheap_path():
    """The fix must not make the ordinary case reconcile when it need not."""
    tensor = TwoMemoryTensor((2 * GRANULE,))
    tensor.to("npu")
    transfers = []
    tensor.storage.sync_from_device = lambda o, n: transfers.append((o, n))
    tensor.fill_(0x44)
    tensor.to("npu")
    assert transfers == []
    assert (tensor.storage._transport.device_bytes == 0x44).all()


def test_a_buffer_must_supply_host_bytes():
    """``host_bytes`` is part of the contract, so the base must require it.

    Every tensor reaches its bytes through it, so a backend that omits it fails
    at first use rather than at construction, which is the wrong end. The two
    transfer methods are declared; this one was not.
    """

    class TransportWithoutHostBytes(tensor_class.Transport):
        def to_device(self, offset, nbytes):
            pass

        def from_device(self, offset, nbytes):
            pass

    with pytest.raises(TypeError, match="host_bytes"):
        TransportWithoutHostBytes()


def test_repr_reaches_the_device_for_a_region_it_does_not_hold():
    """repr renders the data, so it reconciles like the other readers do."""
    tensor = fragmented()
    tensor.storage._transport.device_bytes[GRANULE:] = 0x7B  # 123
    assert "123" in repr(tensor)


def test_a_write_that_raises_records_nothing():
    """A scope that failed wrote nothing, so it must claim nothing.

    ``overwrite()`` skips the reconcile on entry because the caller promised to
    replace every byte. When the write then raises, that promise was not kept
    and the host copy was never pulled, so recording the region as host-written
    claims bytes the host never held: the next read returns them and the next
    transfer pushes them over what the device has.
    """
    tensor = TwoMemoryTensor((GRANULE,), dtype=np.int32)
    tensor.storage._transport.device_bytes[:] = 0xEE
    tensor.to("npu")
    with pytest.raises(ValueError):
        tensor.fill_(float("nan"))  # numpy rejects the cast, writing nothing
    assert tensor.device == "npu"
    assert (tensor.numpy().view(np.uint8) == 0xEE).all()


def test_a_partial_write_that_raises_keeps_the_device_copy():
    """The same at the scope level, where some bytes did land.

    Neither promise can be honoured once the body raises partway. Keeping the
    device copy loses the partial write, which is what a failed write should
    look like; claiming the host copy would lose whatever the device holds.
    """
    tensor = TwoMemoryTensor((2 * GRANULE,), dtype=np.uint8)
    tensor.storage._transport.device_bytes[:] = 0xEE
    tensor.to("npu")
    with pytest.raises(RuntimeError):
        with tensor.overwrite() as array:
            array[:GRANULE] = 0x11
            raise RuntimeError("caller gave up half way")
    assert tensor.device == "npu"
    assert (tensor.numpy() == 0xEE).all()


def test_a_mutate_that_raises_leaves_residency_alone():
    """A failed write must not move residency either, not just not record one.

    ``mutate`` reconciles on entry so a partial update sees current contents.
    Doing that with ``to("cpu")`` also *claims* the extent for the host, which
    survives a body that raises and outlives the write that never happened: the
    next transfer flushes a region nobody wrote, and a failed ``mutate`` and a
    failed ``overwrite`` disagree about what a failed write leaves behind.
    """
    tensor = TwoMemoryTensor((2 * GRANULE,), dtype=np.uint8)
    tensor.storage._transport.device_bytes[:] = 0xEE
    tensor.to("npu")
    with pytest.raises(RuntimeError):
        with tensor.mutate() as array:
            array[:GRANULE] = 0x11
            raise RuntimeError("caller gave up half way")
    assert tensor.device == "npu"
    assert (tensor.numpy() == 0xEE).all()


def test_a_mutate_that_completes_still_sees_current_contents():
    """The reconcile on entry is what a partial update depends on."""
    tensor = TwoMemoryTensor((2 * GRANULE,), dtype=np.uint8)
    tensor.storage._transport.device_bytes[:] = 0xEE
    tensor.to("npu")
    with tensor.mutate() as array:
        array[:GRANULE] = 0x11  # leaves the second window untouched
    assert tensor.device == "cpu"
    tensor.to("npu")
    assert (tensor.storage._transport.device_bytes[:GRANULE] == 0x11).all()
    assert (tensor.storage._transport.device_bytes[GRANULE:] == 0xEE).all()
