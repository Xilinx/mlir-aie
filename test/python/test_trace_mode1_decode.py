# test_trace_mode1_decode.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python -m pytest %s -v

"""Tests for mode-1 (EVENT_PC) trace byte-stream decoding.

Field expectations are computed directly from the EventPC bit layout. The
distilled sequence is capture-derived, but is not an independent decoder oracle.
"""

import pytest
from aie.utils.trace.utils import decode_event_pc_stream


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        (
            [0xC4, 0x20, 0x01, 0x50],
            {"type": "EventPC", "pc": 336, "event3": 3},
        ),
        (
            [0xC6, 0x24, 0x01, 0x50],
            {
                "type": "EventPC",
                "pc": 336,
                "event0": 0,
                "event3": 3,
                "event7": 7,
            },
        ),
        (
            [0xC4, 0x23, 0xC1, 0x50],
            {"type": "EventPC", "pc": 336, "event3": 3},
        ),
    ],
)
def test_event_pc_fields(stream, expected):
    assert decode_event_pc_stream(stream) == [expected]


def test_shared_opcode_framing():
    stream = (
        [0xF1, 0, 0, 0, 0, 0x04, 0xE5, 0xF7]
        + [0xDC, 0, 0, 0]
        + [0xC4, 0x20, 0x01, 0x50]
        + [0xE5, 0xD9, 0x34, 0xFF]
    )

    assert decode_event_pc_stream(stream, zero=False) == [
        {"type": "Start", "timer_value": 321015},
        {"type": "EventPC", "pc": 336, "event3": 3},
        {"type": "Repeat0", "repeats": 5},
        {"type": "Repeat1", "repeats": 308},
        {"type": "Event_Sync"},
    ]


def test_start_rejects_other_opcode_families():
    assert decode_event_pc_stream([0xF9] + [0] * 7) == []
    assert decode_event_pc_stream([0xFD] + [0] * 7) == []


def test_distilled_hardware_capture():
    stream = (
        [0xF1, 0, 0, 0, 0, 0x04, 0xE5, 0xF7]
        + [0xC6, 0x00, 0x03, 0x30]
        + [0xDB, 0xFF]
        + [0xDA, 0xD0]
        + [0xC6, 0x20, 0x03, 0x30]
        + [0xE7]
        + [0xC4, 0x20, 0x01, 0x50]
    )

    assert decode_event_pc_stream(stream, zero=False) == [
        {"type": "Start", "timer_value": 321015},
        {"type": "EventPC", "pc": 816, "event7": 7},
        {"type": "Repeat1", "repeats": 1023},
        {"type": "Repeat1", "repeats": 720},
        {"type": "EventPC", "pc": 816, "event3": 3, "event7": 7},
        {"type": "Repeat0", "repeats": 7},
        {"type": "EventPC", "pc": 336, "event3": 3},
    ]
