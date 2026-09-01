# test_trace_event_sync_carry.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python -m pytest %s -v

"""Regression test: convert_commands_to_json() must account for Event_Sync.

Event_Sync is the carry marker for Single2/Multiple2's 18-bit cycle-delta field (max 0x3FFFF =
262143, from convert_to_commands()'s byte layout): it means the true delta since the last packet
exceeded that field's range, so add one full field range (2**18) before applying the following
command's own (remainder) delta. convert_commands_to_json() had no case for it, so every
Event_Sync was silently dropped -- the running `timer` neither advanced nor emitted anything.

Found via route_b_kernels/codec_block/trace_conv_dispatch.py on real hardware: its `--events
sparse` run reported a per-tile INSTR_EVENT_0->INSTR_EVENT_1 cost of 44602 cycles, 6.9x below
`--events dense`'s 306746 (independently confirmed against an untraced wall-clock measurement of
the same dispatch). 306746 - 44602 == 262144 == 2**18 exactly, and the sparse capture carries
exactly one Event_Sync per tile bracket. The bytes in test_event_sync_carried_across_bracket are
that capture's tile 0, verbatim (loc "2,0"'s core byte stream, offset 27).
"""

from aie.utils.trace.events import get_events_for_device
from aie.utils.trace.parse import EVENT_SYNC_CYCLES, convert_commands_to_json
from aie.utils.trace.utils import convert_to_commands

EVENTS_MODULE = get_events_for_device("npu1_1col")

# pid_events[trace_type][loc] = 8 event-slot codes + the assigned pid (index 8, NUM_EVENTS).
# Slot 0 = INSTR_EVENT_0 (code 33), slot 1 = INSTR_EVENT_1 (code 34) -- matches
# trace_conv_dispatch.py's CORETILE_EVENTS_SPARSE ordering.
PID_EVENTS = [{"2,0": [33, 34, 0, 0, 0, 0, 0, 0, 0]}, {}, {}, {}]


def test_event_sync_cycles_is_the_18_bit_field_range():
    assert EVENT_SYNC_CYCLES == 1 << 18 == 262144


def test_event_sync_carried_across_bracket():
    # Real capture: Single0(event0, +9) Event_Sync Single2(event1, +44601).
    byte_stream = [0x09, 0xFF, 0xA4, 0xAE, 0x39]
    commands = convert_to_commands([{"2,0": byte_stream}, {}, {}, {}])

    trace_events = []
    convert_commands_to_json(trace_events, commands, PID_EVENTS, EVENTS_MODULE)

    begins = {e["name"]: e["ts"] for e in trace_events if e["ph"] == "B"}
    # Without the carry this is 44602 (1 + 9 skipped as the deactivate tick, then 1 + 44601) --
    # a 6.9x undercount of the true, wall-clock-validated 306746.
    assert begins["INSTR_EVENT_1"] - begins["INSTR_EVENT_0"] == 262144 + 44602 == 306746


def test_event_sync_emits_no_trace_event():
    commands = [{"2,0": [{"type": "Event_Sync"}]}, {}, {}, {}]
    trace_events = []
    convert_commands_to_json(trace_events, commands, PID_EVENTS, EVENTS_MODULE)
    assert trace_events == []
