# test_trace_multiple_repeat.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python -m pytest %s -v

"""A repeated trace command must produce the same timeline as its expansion.

Repeating a Multiple command advanced the timer but activated none of its
events, because the activation loop read event fields from the Repeat command
itself. A repeated Single is checked too, since both now share one loop.
"""

import pytest
from aie.utils.trace.events import get_events_for_device
from aie.utils.trace.parse import convert_commands_to_json
from aie.utils.trace.utils import convert_to_commands

# Multiple0(event0, event1, +1) and Single0(event0, +1).
MULTIPLE = [0xC0, 0x31]
SINGLE = [0x01]
# Single0(event2, +1), to end each stream on a different event.
END = [0x21]


def timeline(frames):
    # Trace frames cannot cross a 32-bit word boundary. Insert Filler bytes
    # before frames that would cross, and pad the final word.
    byte_stream = []
    for frame in frames:
        remaining = 4 - len(byte_stream) % 4
        if len(frame) > remaining:
            byte_stream.extend([0xFE] * remaining)
        byte_stream.extend(frame)
    byte_stream.extend([0xFE] * (-len(byte_stream) % 4))
    pid_events = [{"2,0": [33, 34, 35, 0, 0, 0, 0, 0, 0]}, {}, {}, {}]
    commands = convert_to_commands([{"2,0": byte_stream}, {}, {}, {}])
    events = []
    convert_commands_to_json(
        events, commands, pid_events, get_events_for_device("npu1_1col")
    )
    return events


# Repeat0 encodes counts below 16; Repeat1 encodes larger counts.
REPEATS = pytest.mark.parametrize(
    ("repeats", "repeat_bytes"),
    [(2, [0xE2]), (16, [0xD8, 0x10])],
    ids=["Repeat0", "Repeat1"],
)


@REPEATS
def test_repeated_multiple_matches_expansion(repeats, repeat_bytes):
    compressed = [MULTIPLE, repeat_bytes, END]
    expanded = [MULTIPLE] * (repeats + 1) + [END]
    assert timeline(compressed) == timeline(expanded)


@REPEATS
def test_repeated_single_matches_expansion(repeats, repeat_bytes):
    compressed = [SINGLE, repeat_bytes, END]
    expanded = [SINGLE] * (repeats + 1) + [END]
    assert timeline(compressed) == timeline(expanded)
