# test_trace_split_segments.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# RUN: %python -m pytest %s -v

"""Tests for split_trace_segments and write_trace fixes.

These functions handle multi-channel trace buffers where two S2MM DMA
channels write to the same buffer at different offsets, producing a
zero-word gap between the two channels' trace data.
"""

import pytest
import numpy as np
import os
import tempfile

from aie.utils.trace.utils import split_trace_segments
from aie.utils.trace.config import TraceConfig


# -- split_trace_segments tests ------------------------------------------------


class TestSplitTraceSegments:
    """Tests for splitting trace buffers into per-channel segments."""

    def test_single_segment_no_zeros(self):
        """All non-zero data: returns one segment unchanged."""
        data = ["deadbeef", "cafebabe", "12345678"]
        result = split_trace_segments(data)
        assert result == [data]

    def test_single_segment_short_zero_run(self):
        """Short zero run (< min_gap) stays within the segment."""
        data = ["deadbeef", "00000000", "00000000", "cafebabe"]
        result = split_trace_segments(data, min_gap=8)
        assert result == [data]

    def test_two_segments_separated_by_gap(self):
        """Two data blocks separated by >= min_gap zeros: two segments."""
        ch0 = ["deadbeef", "cafebabe"]
        gap = ["00000000"] * 10
        ch1 = ["12345678", "abcd1234"]
        data = ch0 + gap + ch1
        result = split_trace_segments(data, min_gap=8)
        assert result == [ch0, ch1]

    def test_exact_min_gap_boundary(self):
        """Exactly min_gap zeros: treated as a gap (boundary is inclusive)."""
        ch0 = ["aabbccdd"]
        gap = ["00000000"] * 8
        ch1 = ["11223344"]
        data = ch0 + gap + ch1
        result = split_trace_segments(data, min_gap=8)
        assert result == [ch0, ch1]

    def test_just_under_min_gap(self):
        """One fewer than min_gap zeros: stays as one segment."""
        ch0 = ["aabbccdd"]
        short_gap = ["00000000"] * 7
        ch1 = ["11223344"]
        data = ch0 + short_gap + ch1
        result = split_trace_segments(data, min_gap=8)
        assert result == [data]

    def test_trailing_zeros_dropped(self):
        """Trailing zeros after last data are not included in any segment."""
        data = ["deadbeef", "cafebabe"] + ["00000000"] * 20
        result = split_trace_segments(data, min_gap=8)
        assert result == [["deadbeef", "cafebabe"]]

    def test_leading_zeros_dropped(self):
        """Leading zeros before first data are dropped."""
        data = ["00000000"] * 20 + ["deadbeef", "cafebabe"]
        result = split_trace_segments(data, min_gap=8)
        assert result == [["deadbeef", "cafebabe"]]

    def test_all_zeros(self):
        """All-zero buffer returns empty list."""
        data = ["00000000"] * 100
        result = split_trace_segments(data)
        assert result == []

    def test_empty_input(self):
        """Empty input returns empty list."""
        result = split_trace_segments([])
        assert result == []

    def test_empty_strings_ignored(self):
        """Empty strings (from file reading) are skipped."""
        data = ["deadbeef", "", "cafebabe", ""]
        result = split_trace_segments(data)
        assert result == [["deadbeef", "cafebabe"]]

    def test_three_segments(self):
        """Three data blocks separated by gaps: three segments."""
        ch0 = ["aaaaaaaa"]
        ch1 = ["bbbbbbbb"]
        ch2 = ["cccccccc"]
        gap = ["00000000"] * 10
        data = ch0 + gap + ch1 + gap + ch2
        result = split_trace_segments(data, min_gap=8)
        assert result == [ch0, ch1, ch2]

    def test_realistic_trace_buffer(self):
        """Simulates a real split trace buffer layout.

        Channel 0: 16 words of trace data at offset 0
        Channel 1: 16 words of trace data at offset 64 (buffer_size=64 words)
        Total buffer: 128 words (2 * buffer_size)
        """
        buffer_size = 64  # words per channel
        ch0_data = [f"{i:08x}" for i in range(0x10000001, 0x10000011)]  # 16 words
        ch0_pad = ["00000000"] * (buffer_size - len(ch0_data))
        ch1_data = [f"{i:08x}" for i in range(0x20000001, 0x20000011)]  # 16 words
        ch1_pad = ["00000000"] * (buffer_size - len(ch1_data))

        full_buffer = ch0_data + ch0_pad + ch1_data + ch1_pad
        assert len(full_buffer) == 2 * buffer_size

        result = split_trace_segments(full_buffer, min_gap=8)
        assert len(result) == 2
        assert result[0] == ch0_data
        assert result[1] == ch1_data


# -- write_trace tests ---------------------------------------------------------


class TestWriteTrace:
    """Tests for write_trace stripping only trailing zeros."""

    def _write_and_read(self, trace_array):
        """Helper: write trace to temp file, read back as hex strings."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            tc = TraceConfig(trace_size=len(trace_array) * 4, trace_file=path)
            tc.write_trace(trace_array)
            with open(path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            return lines
        finally:
            os.unlink(path)

    def test_strips_trailing_zeros(self):
        """Trailing zeros should be removed."""
        trace = np.array([0xDEADBEEF, 0xCAFEBABE, 0, 0, 0], dtype=np.uint32)
        lines = self._write_and_read(trace)
        assert lines == ["deadbeef", "cafebabe"]

    def test_preserves_internal_zeros(self):
        """Internal zeros (between non-zero data) should be kept."""
        trace = np.array([0xDEADBEEF, 0, 0, 0xCAFEBABE, 0, 0], dtype=np.uint32)
        lines = self._write_and_read(trace)
        assert lines == ["deadbeef", "00000000", "00000000", "cafebabe"]

    def test_split_channel_buffer_preserved(self):
        """Simulates a 2-channel buffer: gap between channels is preserved."""
        # Channel 0 data (4 words) + gap (8 words) + Channel 1 data (4 words)
        # + trailing zeros (8 words)
        ch0 = [0x10000001, 0x10000002, 0x10000003, 0x10000004]
        gap = [0] * 8
        ch1 = [0x20000001, 0x20000002, 0x20000003, 0x20000004]
        trailing = [0] * 8
        trace = np.array(ch0 + gap + ch1 + trailing, dtype=np.uint32)

        lines = self._write_and_read(trace)

        # Trailing zeros stripped, but gap zeros preserved
        expected_count = len(ch0) + len(gap) + len(ch1)
        assert len(lines) == expected_count

        # Channel boundaries are intact
        assert lines[0] == "10000001"
        assert lines[3] == "10000004"
        assert lines[4] == "00000000"  # gap start
        assert lines[11] == "00000000"  # gap end
        assert lines[12] == "20000001"  # ch1 start
        assert lines[15] == "20000004"  # ch1 end

    def test_all_zeros(self):
        """All-zero buffer produces empty file."""
        trace = np.array([0, 0, 0, 0], dtype=np.uint32)
        lines = self._write_and_read(trace)
        assert lines == []

    def test_no_zeros(self):
        """Buffer with no zeros: all data preserved."""
        trace = np.array([0xAA, 0xBB, 0xCC], dtype=np.uint32)
        lines = self._write_and_read(trace)
        assert lines == ["000000aa", "000000bb", "000000cc"]
