# test_trace_mode1_decode.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python -m pytest %s -v

"""Tests for mode-1 (EVENT_PC) trace decoding and parsing.

Field expectations are computed directly from the EventPC bit layout. The
distilled sequence is capture-derived, but is not an independent decoder oracle.
"""

import pytest
from aie.dialects.aie import AIEDevice, TraceMode, device, tile
from aie.extras.context import mlir_mod_ctx
from aie.iron import Program, Runtime, Worker
from aie.iron.device import NPU1
from aie.utils.trace import configure_trace
from aie.utils.trace.events import get_events_for_device
from aie.utils.trace.parse import (
    _convert_to_commands_by_mode,
    convert_commands_to_json,
    parse_mlir_trace_events,
    parse_trace,
)
from aie.utils.trace.utils import decode_event_pc_stream

MODE1_CONFIG_MLIR = """module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @sequence() {
      %address = arith.constant 213200 : i32
      %value = arith.constant 5 : i32
      aiex.npu.write32(%address, %value) {column = 2 : i32, row = 2 : i32} : i32, i32
      %events_address = arith.constant 213216 : i32
      %events_value = arith.constant 553648128 : i32
      aiex.npu.write32(%events_address, %events_value) {column = 2 : i32, row = 2 : i32} : i32, i32
    }
  }
}"""


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


def test_configure_trace_accepts_event_pc_mode():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            configure_trace([tile(0, 2)], core_trace_mode=TraceMode.EventPC)

    assert 'aie.trace.mode "Event-PC"' in str(ctx.module)


def test_program_enable_trace_forwards_event_pc_mode():
    worker = Worker(None)
    program = Program(NPU1(), Runtime(lambda: None, []), workers=[worker])
    program.enable_trace(
        trace_size=8192,
        workers=[worker],
        core_trace_mode=TraceMode.EventPC,
    )

    assert 'aie.trace.mode "Event-PC"' in str(program.resolve_program())


def test_parse_mlir_trace_events_returns_core_mode():
    _, trace_modes, _ = parse_mlir_trace_events(MODE1_CONFIG_MLIR)

    assert trace_modes == [{"2,2": 1}, {}, {}, {}]


def test_command_dispatch_preserves_mode0_and_replaces_mode1():
    byte_streams = [
        {"2,0": [0x03], "2,1": [0xC4, 0x20, 0x01, 0x50]},
        {},
        {},
        {},
    ]
    trace_modes = [{"2,1": 1}, {}, {}, {}]

    assert _convert_to_commands_by_mode(byte_streams, trace_modes) == [
        {
            "2,0": [{"type": "Single0", "event": 0, "cycles": 3}],
            "2,1": [{"type": "EventPC", "pc": 336, "event3": 3}],
        },
        {},
        {},
        {},
    ]


def test_command_dispatch_rejects_unsupported_mode():
    with pytest.raises(NotImplementedError, match="mode 2.*2,1"):
        _convert_to_commands_by_mode(
            [{"2,1": [0x03]}, {}, {}, {}],
            [{"2,1": 2}, {}, {}, {}],
        )


def test_event_pc_json_uses_capture_order_and_expands_repeats():
    commands = [
        {
            "2,1": [
                {"type": "EventPC", "pc": 336, "event3": 3, "event7": 7},
                {"type": "Repeat0", "repeats": 2},
                {"type": "EventPC", "pc": 528, "event3": 3},
            ]
        },
        {},
        {},
        {},
    ]
    pid_events = [{"2,1": [0, 0, 0, 33, 0, 0, 0, 34, 42]}, {}, {}, {}]
    trace_events = []

    convert_commands_to_json(
        trace_events, commands, pid_events, get_events_for_device("npu1_1col")
    )

    assert [
        (event["ts"], event["tid"], event["name"], event["args"]["pc"])
        for event in trace_events
    ] == [
        (0, 3, "INSTR_EVENT_0", 336),
        (0, 7, "INSTR_EVENT_1", 336),
        (1, 3, "INSTR_EVENT_0", 336),
        (1, 7, "INSTR_EVENT_1", 336),
        (2, 3, "INSTR_EVENT_0", 336),
        (2, 7, "INSTR_EVENT_1", 336),
        (3, 3, "INSTR_EVENT_0", 528),
    ]
    assert {(event["pid"], event["ph"], event["s"]) for event in trace_events} == {
        (42, "i", "t")
    }


def test_parse_trace_aligns_mode_before_dispatch():
    trace_buffer = [
        0x00220002,
        0xF1000000,
        0,
        0xC4200150,
        0xE2FEFEFE,
        0xC4200210,
        0xFEFEFEFE,
        0xFEFEFEFE,
    ]

    trace_events = parse_trace(trace_buffer, MODE1_CONFIG_MLIR)

    assert [
        (event["ts"], event["tid"], event["args"]["pc"])
        for event in trace_events
        if event["ph"] == "i"
    ] == [(0, 3, 336), (1, 3, 336), (2, 3, 336), (3, 3, 528)]
