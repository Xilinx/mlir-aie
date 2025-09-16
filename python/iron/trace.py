#!/usr/bin/env python3
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

import json
import argparse
import sys
import re
import numpy as np
from .tensor import zeros
from .trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent

# Add missing imports from parse_trace.py
from aie.extras.util import find_ops
from aie.ir import Context, Module, Location
import aie.dialects.aie as aiedialect
import aie.dialects.aiex as aiexdialect

# Number of different trace types, currently 4
# core:    pkt type 0
# mem:     pkt type 1
# shim:   pkt type 2
# memtile: pkt type 3
NumTraceTypes = 4
NUM_EVENTS = 8  # number of events we can view per trace

# Global variables
_trace_active = False
_trace_tensor = None
DEBUG = False
_trace_size = 4096  # Default trace size (4k)
_trace_buffer = []
_dummy_tensor = None  # Reusable dummy tensor
_mlir_module = None  # Global to store MLIR from JIT


def _prepare_trace_input(trace_input):
    """
    Prepare trace input by detecting if it's a file path or data string.

    Args:
        trace_input: Either a file path string or data string

    Returns:
        List of trace data strings
    """
    if isinstance(trace_input, str):
        # Assume it's a file path unless it contains newlines (then it's data)
        if "\n" in trace_input:
            # It's trace data as a string, split by newlines
            return trace_input.split("\n")
        else:
            # Assume it's a file path
            try:
                with open(trace_input, "r") as f:
                    return f.read().split("\n")
            except Exception as e:
                raise RuntimeError(f"Could not open trace file {trace_input}: {e}")
    else:
        return trace_input


def parse_and_save_trace(
    trace_input, mlir_input, colshift=None, debug=False, output_file=None
):
    """
    Parse trace data and optionally save to file.

    Args:
        trace_input: Either a string (file path) or list of strings (trace data)
        mlir_input: Either a string (file path) or MLIR module string
        colshift: Optional column shift adjustment
        debug: Whether to enable debug mode
        output_file: Optional file path or file object for output

    Returns:
        None (output is written to file if output_file is provided)
    """
    global DEBUG
    DEBUG = debug

    # Handle trace input - can be file path, data, or numpy array
    if hasattr(trace_input, "dtype"):  # numpy array
        trace_pkts = [f"{val:08x}" for val in trace_input]
    else:
        trace_pkts = _prepare_trace_input(trace_input)

    # Handle MLIR input - can be file path or data
    if isinstance(mlir_input, str) and not mlir_input.strip().startswith("module"):
        # Assume it's a file path
        try:
            with open(mlir_input, "r") as f:
                mlir_module_str = f.read()
        except Exception as e:
            raise RuntimeError(f"Could not open MLIR file {mlir_input}: {e}")
    else:
        mlir_module_str = mlir_input

    # Parse MLIR trace events
    try:
        pid_events = parse_mlir_trace_events(mlir_module_str, colshift)
    except Exception as e:
        raise RuntimeError(f"Could not parse MLIR module: {e}")

    # Validate trace data
    if not check_for_valid_trace(
        trace_input if isinstance(trace_input, str) else "input_data",
        trace_pkts,
        output_file,
    ):
        raise RuntimeError("Invalid trace data")

    # Process trace data
    trimmed_trace_pkts = trim_trace_pkts(trace_pkts)
    if DEBUG and output_file:
        lines_removed = len(trace_pkts) - len(trimmed_trace_pkts)
        print(f"DEBUG: trimmed {lines_removed} lines", file=output_file)

    trace_pkts_sorted = trace_pkts_de_interleave(trimmed_trace_pkts)
    byte_streams = convert_to_byte_stream(trace_pkts_sorted)
    commands_0 = convert_to_commands(byte_streams, False)

    # Align column start index if no colshift provided
    if colshift is None:
        pid_events = align_column_start_index(pid_events, commands_0)

    # Generate trace events
    trace_events = []
    setup_trace_metadata(trace_events, pid_events)
    convert_commands_to_json(trace_events, commands_0, pid_events, output_file)

    # If output_file is provided, write JSON directly
    if output_file:
        # output_file is always a string filename, so open it and write
        with open(output_file, "w") as f:
            print(
                json.dumps(trace_events).replace("'", '"').replace(", {", ",\n{"),
                file=f,
            )


def lookup_event_name_by_type(trace_type, code):
    """Look up event name by trace type and code."""
    event = ""
    events_enum = None
    if trace_type == 0:  # Core traces
        events_enum = CoreEvent
    elif trace_type == 1:  # Mem traces
        events_enum = MemEvent
    elif trace_type == 2:  # Shim traces
        events_enum = ShimTileEvent
    elif trace_type == 3:  # MemTile traces
        events_enum = MemTileEvent
    if events_enum is not None and code in set(x.value for x in events_enum):
        event = events_enum(code).name
    else:
        event = "Unknown"
    return event


def lookupEventNameInStr(event, pid, pid_events):
    """
    TODO Expand to other pid for multiple cores? even/odd
    For now, we assume a single trace event and key based on that
    in the future, the pid will be used to match the right events
    """
    return lookup_event_name_by_code(pid_events[0][int(event)])


def lookup_event_name_by_code(code, pid_events):
    """
    TODO Expand to other pid for multiple cores? even/odd
    For now, we assume a single trace event and key based on that
    in the future, the pid will be used to match the right events
    """
    return lookup_event_name_by_type(0, pid_events[0][int(code)])


def make_event_lists(commands):
    """Create event lists from commands."""
    events = {}
    ts = 0
    for i, command in enumerate(commands):
        if command["type"] == "Start":
            ts = command["timer_value"]
        if command["type"] == "Event_Sync":
            ts += 0x3FFFF  # Typo in spec
        if "Single" in command["type"]:
            ts += command["cycles"]
            if command["event"] in events:
                events[command["event"]].append(ts)
            else:
                events[command["event"]] = [ts]
    return events


def flatten_repeat_command(commands):
    """Flatten repeat commands for processing."""
    prev = 0
    flat_commands = list()
    for c in commands:
        if c["type"] == "Repeat0" or c["type"] == "Repeat1":
            for i in range(int(c["repeats"])):
                flat_commands.append(prev)
        else:
            flat_commands.append(c)
            prev = c
    return flat_commands


def deactivate_events(
    multiples,
    active_events,
    timer,
    cycles,
    pid,
    trace_type,
    loc,
    pid_events,
    trace_events,
):
    """Deactivate events based on conditions."""
    for k in active_events.keys():  # an active event
        if cycles > 0 or (cycles == 0 and not k in multiples):
            if active_events[k] > 0:
                trace_event = {
                    "name": lookup_event_name_by_type(
                        trace_type, pid_events[trace_type][loc][k]
                    )
                }
                trace_event["ts"] = timer
                trace_event["ph"] = "E"
                trace_event["pid"] = pid
                trace_event["tid"] = k
                trace_event["args"] = {}
                trace_events.append(trace_event)
            active_events[k] = 0


def activate_event(event, tt, loc, timer, pid, active_events, pid_events, trace_events):
    """Activate an event."""
    try:
        if active_events[event] == 0:
            trace_event = {
                "name": lookup_event_name_by_type(tt, pid_events[tt][loc][event])
            }
            trace_event["ts"] = timer
            trace_event["ph"] = "B"
            trace_event["pid"] = pid
            trace_event["tid"] = event
            trace_event["args"] = {}
            trace_events.append(trace_event)
            active_events[event] = 1
    except KeyError:
        pass


def process_name_metadata(trace_events, pid, trace_type, loc):
    """Process name metadata for trace events."""
    trace_event = {"name": "process_name"}
    trace_event["ph"] = "M"
    trace_event["pid"] = pid
    trace_event["args"] = {}
    if trace_type == 0:
        trace_event["args"]["name"] = "core_trace for tile" + str(loc)
    elif trace_type == 1:
        trace_event["args"]["name"] = "mem_trace for tile" + str(loc)
    elif trace_type == 2:
        trace_event["args"]["name"] = "shim_trace for tile" + str(loc)
    elif trace_type == 3:
        trace_event["args"]["name"] = "memtile_trace for tile" + str(loc)
    trace_events.append(trace_event)


def thread_name_metadata(trace_events, trace_type, loc, pid, tid, pid_events):
    """Process thread name metadata."""
    trace_event = {"name": "thread_name"}
    trace_event["ph"] = "M"
    trace_event["pid"] = pid
    trace_event["tid"] = tid
    trace_event["args"] = {}
    trace_event["args"]["name"] = lookup_event_name_by_type(
        trace_type, pid_events[trace_type][loc][tid]
    )
    trace_events.append(trace_event)


def start_tracing(size=4096):
    """
    Start tracing functionality with optional size parameter.

    Args:
        size (int, optional): The maximum size of the trace buffer. Defaults to 4096 (4k).
    """
    global _trace_active, _trace_buffer, _trace_tensor, _trace_size, _dummy_tensor

    # Set trace size
    if size > 0:
        _trace_size = size
    else:
        raise RuntimeError("Trace size must be positive")

    if not _trace_active:
        _trace_active = True
        _trace_buffer = []

        # Create the trace tensor when starting trace
        _trace_tensor = zeros(_trace_size, dtype=np.uint32)
        # Create a reusable dummy tensor
        _dummy_tensor = zeros(1, dtype=np.uint32)


def stop_tracing(output_file="trace.json"):
    """Stop tracing functionality and optionally save to file."""
    if not output_file.endswith(".json"):
        raise RuntimeError("Only JSON output files are supported")

    global _trace_active, _trace_tensor, _mlir_module
    if _trace_active:
        _trace_active = False
        # Read trace tensor data and save to file
        if _trace_tensor is not None:
            # Check if MLIR module is available
            if _mlir_module is None:
                raise RuntimeError(
                    "No MLIR module available for tracing. Did you call set_mlir_module()?"
                )

            # Parse trace data using the library function
            parse_and_save_trace(
                trace_input=_trace_tensor.numpy(),
                mlir_input=_mlir_module,
                colshift=None,
                debug=False,
                output_file=output_file,
            )
        else:
            raise RuntimeError("No trace tensor found")
    else:
        raise RuntimeError("Tracing was not active")


def set_mlir_module(mlir_module):
    """Set the MLIR module for tracing to use."""
    global _mlir_module
    _mlir_module = mlir_module


# Getter functions
def get_trace_size():
    """Get the current trace buffer size limit."""
    return _trace_size


def _get_trace_active():
    """Get the current tracing status."""
    return _trace_active


def _get_trace_tensor():
    """Get the current trace tensor."""
    return _trace_tensor


def _get_dummy_tensor():
    """Get the reusable dummy tensor."""
    return _dummy_tensor


# Check for valid trace packets data
# 1) if only 1 trace packet
# 2) if first trace packet is all 0's
def check_for_valid_trace(filename, trace_pkts, of):
    if DEBUG:
        print("len(trace_pkts): ", str(len(trace_pkts)), file=of)
        print("trace_pkts[0]:", trace_pkts[0], file=of)
    if len(trace_pkts) < 2 or trace_pkts[0] == "00000000":
        print(
            "[ERROR] Empty trace file. Valid trace was not written to",
            filename,
            file=sys.stderr,
        )
        print(
            "See https://github.com/Xilinx/mlir-aie/tree/main/programming_guide/section-4/section-4b#Additional-Debug-Hints for additional trace debug tips.",
            file=sys.stderr,
        )
        return False
    return True


def trim_trace_pkts(trace_pkts):
    for i in range(len(trace_pkts)):
        if trace_pkts[i] == "fefefefe" or trace_pkts[i] == "FEFEFEFE":
            if i + 2 < len(trace_pkts):
                if trace_pkts[i + 1] == "00000000" and trace_pkts[i + 2] == "00000000":
                    return trace_pkts[0 : i + 1]
    return trace_pkts


def check_odd_word_parity(word):
    val = 0
    for i in range(32):
        val = val ^ ((word >> i) & 0x1)
    return val == 1


def parse_pkt_hdr_in_stream(word):
    hdr = dict()
    w = int(word)
    hdr["valid"] = check_odd_word_parity(w)
    # TODO can we assume non used fields must be 0 to rule out other data packets?
    # what about bit[5:10]?
    if (((w >> 5) & 0x7F) != 0) or (((w >> 19) & 0x1) != 0) or (((w >> 28) & 0x7) != 0):
        hdr["valid"] = False
    else:
        # TODO Do we need to check for valid row/col for given device?
        hdr["col"] = (w >> 21) & 0x7F
        hdr["row"] = (w >> 16) & 0x1F
        hdr["type"] = (w >> 12) & 0x3
        hdr["id"] = w & 0x1F
    return hdr


# Sorts list of trace packets into a list indexed by trace type (core, mem, shim, memtile)
# and the value is dictionary tile location (key) and trace packets (value)
#
# trace_pkts_sorted:   list (idx = types of traces, currently 4, value = stream_dict)
# stream_dict: dict (key = row,col, value = list of word streams)
def trace_pkts_de_interleave(word_stream):
    trace_pkts_sorted = list()
    for t in range(NumTraceTypes):
        trace_pkts_sorted.append(dict())

    curr_pkt_type = 0
    curr_loc = ""
    curr_vld = False  # only used in the beginning

    for i in range(len(word_stream)):
        if word_stream[i] == "":
            break  # TODO Assumes a blank line is the last line
        if (i % 8) == 0:
            pkt_hdr = parse_pkt_hdr_in_stream(int(word_stream[i], 16))
            if pkt_hdr["valid"]:
                curr_loc = str(pkt_hdr["row"]) + "," + str(pkt_hdr["col"])
                valid_type_found = False
                for tt in range(NumTraceTypes):
                    if pkt_hdr["type"] == tt:
                        curr_pkt_type = tt
                        if trace_pkts_sorted[tt].get(curr_loc) == None:
                            trace_pkts_sorted[tt][curr_loc] = list()
                        valid_type_found = True
                if not valid_type_found:
                    sys.exit("Error: Invalid packet type")
            curr_vld = True
        else:
            if curr_vld:  # ignores first 8 chunks of data is pkt hdr was invalid
                trace_pkts_sorted[curr_pkt_type][curr_loc].append(word_stream[i])
    return trace_pkts_sorted


# Convert trace packets into byte streams
def convert_to_byte_stream(toks_list):
    byte_stream_list = list()
    for l in toks_list:
        byte_stream_dict = dict()
        for loc, stream in l.items():
            byte_stream_dict[loc] = list()
            f = ["", "a5a5a5a5"]
            toks = [t for t in stream if not t in f]
            events = [int(t, 16) for t in toks]
            for event in events:
                for top in range(4):
                    byte = 3 - top
                    opcode = event >> (byte * 8) & 0xFF
                    byte_stream_dict[loc].append(opcode)
        byte_stream_list.append(byte_stream_dict)
    return byte_stream_list


# Convert byte streams to equivalent packet commands
def convert_to_commands(byte_stream_list, zero=True):
    commands = list()
    for t in range(NumTraceTypes):
        commands.append(dict())

    for t in range(NumTraceTypes):
        for key, byte_stream in byte_stream_list[t].items():
            cursor = 0
            commands[t][key] = list()
            try:
                while True:
                    if (byte_stream[cursor] & 0b11111011) == 0b11110000:
                        com = {"type": "Start", "timer_value": 0}
                        if not zero:
                            for i in range(7):
                                com["timer_value"] += (byte_stream[cursor + i + 1]) * (
                                    256 ** (6 - i)
                                )
                        commands[t][key].append(com)
                        cursor = cursor + 8
                    if (byte_stream[cursor] & 0b11111100) == 0b11011100:
                        # We don't care about these
                        cursor = cursor + 4
                    if (byte_stream[cursor] & 0b10000000) == 0b00000000:
                        com = {"type": "Single0"}
                        com["event"] = (byte_stream[cursor]) >> 4 & 0b111
                        com["cycles"] = (byte_stream[cursor]) & 0b1111
                        commands[t][key].append(com)
                        cursor = cursor + 1
                    if (byte_stream[cursor] & 0b11100000) == 0b10000000:
                        com = {"type": "Single1"}
                        com["event"] = (byte_stream[cursor]) >> 2 & 0b111
                        com["cycles"] = ((byte_stream[cursor]) & 0b11) * 256
                        com["cycles"] += byte_stream[cursor + 1]
                        commands[t][key].append(com)
                        cursor = cursor + 2
                    if (byte_stream[cursor] & 0b11100000) == 0b10100000:
                        com = {"type": "Single2"}
                        com["event"] = (byte_stream[cursor]) >> 2 & 0b111
                        com["cycles"] = ((byte_stream[cursor]) & 0b11) * 256 * 256
                        com["cycles"] += byte_stream[cursor + 1] * 256
                        com["cycles"] += byte_stream[cursor + 2]
                        commands[t][key].append(com)
                        cursor = cursor + 3
                    if (byte_stream[cursor] & 0b11110000) == 0b11000000:
                        com = {"type": "Multiple0"}
                        com["cycles"] = byte_stream[cursor + 1] & 0b1111
                        events = (byte_stream[cursor] & 0b1111) << 4
                        events = events + (byte_stream[cursor + 1] >> 4)
                        for i in range(0, 8):
                            e = (events >> i) & 0b1
                            if e:
                                com["event" + str(i)] = i
                        commands[t][key].append(com)
                        cursor = cursor + 2
                    if (byte_stream[cursor] & 0b11111100) == 0b11010000:
                        com = {"type": "Multiple1"}
                        cycles = (byte_stream[cursor + 1] & 0b11) << 8
                        com["cycles"] = cycles + (byte_stream[cursor + 2])
                        events = (byte_stream[cursor] & 0b11) << 6
                        events = events + (byte_stream[cursor + 1] >> 2)
                        for i in range(0, 8):
                            e = (events >> i) & 0b1
                            if e:
                                com["event" + str(i)] = i
                        commands[t][key].append(com)
                        cursor = cursor + 3
                    if (byte_stream[cursor] & 0b11111100) == 0b11010100:
                        com = {"type": "Multiple2"}
                        cycles = (byte_stream[cursor + 1] & 0b11) << 16
                        cycles = cycles + ((byte_stream[cursor + 2]) << 8)
                        com["cycles"] = cycles + (byte_stream[cursor + 3])
                        events = (byte_stream[cursor] & 0b11) << 6
                        events = events + (byte_stream[cursor + 1] >> 2)
                        for i in range(0, 8):
                            e = (events >> i) & 0b1
                            if e:
                                com["event" + str(i)] = i
                        commands[t][key].append(com)
                        cursor = cursor + 4
                    if (byte_stream[cursor] & 0b11110000) == 0b11100000:
                        com = {"type": "Repeat0"}
                        com["repeats"] = (byte_stream[cursor]) & 0b1111
                        commands[t][key].append(com)
                        cursor = cursor + 1
                    if (byte_stream[cursor] & 0b11111100) == 0b11011000:
                        com = {"type": "Repeat1"}
                        com["repeats"] = ((byte_stream[cursor]) & 0b11) * 256
                        com["repeats"] += byte_stream[cursor + 1]
                        commands[t][key].append(com)
                        cursor = cursor + 2
                    if (byte_stream[cursor] & 0b11111111) == 0b11111110:
                        # No one likes you filler, get out of here
                        cursor = cursor + 1
                    if (byte_stream[cursor] & 0b11111111) == 0b11111111:
                        com = {"type": "Event_Sync"}
                        commands[t][key].append(com)
                        cursor = cursor + 1
            except IndexError:
                pass

    return commands


def parse_mlir_trace_events(mlir_module_str, colshift=None):
    """
    Parse MLIR module to extract trace event configurations.

    This searches for npu.write32 and categorizes them based on address and row.
    It's probably not the best way to do it but it's the initial implementation.
    memtile and core/shim tiles have different addresses. For now, we distinguish
    between core and shim tile by row=0
    """
    pid_events = list()
    for t in range(NumTraceTypes):
        pid_events.append(dict())

    with Context(), Location.unknown():
        module = Module.parse(mlir_module_str)

        write32s = find_ops(
            module.operation,
            lambda o: isinstance(o.operation.opview, aiexdialect.NpuWrite32Op),
        )
        device = find_ops(
            module.operation,
            lambda o: isinstance(o.operation.opview, aiedialect.DeviceOp),
        )
        device = aiedialect.AIEDevice(int(device[0].device))
        target_model = aiedialect.get_target_model(device)

    for write32 in write32s:
        address = None
        row = None
        col = None
        value = None
        if write32.address:
            address = write32.address.value
        if write32.row:
            row = write32.row.value
        if write32.column:
            col = write32.column.value
        if write32.value:
            value = write32.value.value

        if row is None and col is None:
            row = (address >> target_model.get_row_shift()) & 0x1F
            col = (address >> target_model.get_column_shift()) & 0x1F
            address = address & 0xFFFFF  # 20 bits address

        if None in [row, col, address, value]:
            print(f"[ERROR] Could not decode write32 op '{write32}'")
            sys.exit(1)

        # Adjust column based on colshift
        if colshift is not None:
            col = col + colshift
        key = str(row) + "," + str(col)

        # core event 0
        if address == 0x340E0:  # 213216, match ignoring case
            if row == 0:  # shim
                if pid_events[2].get(key) == None:
                    pid_events[2][key] = [0] * 8
                pid_events[2][key][0] = value & 0xFF
                pid_events[2][key][1] = (value >> 8) & 0xFF
                pid_events[2][key][2] = (value >> 16) & 0xFF
                pid_events[2][key][3] = (value >> 24) & 0xFF
            else:  # core
                if pid_events[0].get(key) == None:
                    pid_events[0][key] = [0] * 8
                pid_events[0][key][0] = value & 0xFF
                pid_events[0][key][1] = (value >> 8) & 0xFF
                pid_events[0][key][2] = (value >> 16) & 0xFF
                pid_events[0][key][3] = (value >> 24) & 0xFF
        # core event 1
        elif address == 0x340E4:  # 213220, match ignoring case
            if row == 0:  # shim
                if pid_events[2].get(key) == None:
                    pid_events[2][key] = [0] * 8
                pid_events[2][key][4] = value & 0xFF
                pid_events[2][key][5] = (value >> 8) & 0xFF
                pid_events[2][key][6] = (value >> 16) & 0xFF
                pid_events[2][key][7] = (value >> 24) & 0xFF
            else:  # core
                if pid_events[0].get(key) == None:
                    pid_events[0][key] = [0] * 8
                pid_events[0][key][4] = value & 0xFF
                pid_events[0][key][5] = (value >> 8) & 0xFF
                pid_events[0][key][6] = (value >> 16) & 0xFF
                pid_events[0][key][7] = (value >> 24) & 0xFF
        # mem event 0
        elif address == 0x140E0:  # 82144
            if pid_events[1].get(key) == None:
                pid_events[1][key] = [0] * 8
            pid_events[1][key][0] = value & 0xFF
            pid_events[1][key][1] = (value >> 8) & 0xFF
            pid_events[1][key][2] = (value >> 16) & 0xFF
            pid_events[1][key][3] = (value >> 24) & 0xFF
        # mem event 1
        elif address == 0x140E4:  # 82148
            if pid_events[1].get(key) == None:
                pid_events[1][key] = [0] * 8
            pid_events[1][key][4] = value & 0xFF
            pid_events[1][key][5] = (value >> 8) & 0xFF
            pid_events[1][key][6] = (value >> 16) & 0xFF
            pid_events[1][key][7] = (value >> 24) & 0xFF
        # memtile event 0
        elif address == 0x940E0:  # 606432
            if pid_events[3].get(key) == None:
                pid_events[3][key] = [0] * 8
            pid_events[3][key][0] = value & 0xFF
            pid_events[3][key][1] = (value >> 8) & 0xFF
            pid_events[3][key][2] = (value >> 16) & 0xFF
            pid_events[3][key][3] = (value >> 24) & 0xFF
        # memtile event 1
        elif address == 0x940E4:  # 606436
            if pid_events[3].get(key) == None:
                pid_events[3][key] = [0] * 8
            pid_events[3][key][4] = value & 0xFF
            pid_events[3][key][5] = (value >> 8) & 0xFF
            pid_events[3][key][6] = (value >> 16) & 0xFF
            pid_events[3][key][7] = (value >> 24) & 0xFF

    return pid_events


def align_column_start_index(events, commands):
    """
    Attempt to align the starting column of trace in the design (from 'events')
    with the start first column observed in the trace ('commands'). This is needed
    because the runtime/firmware can start the design on any valid column
    """
    # find min column of commands
    min_commands_col = float("inf")
    for t in range(NumTraceTypes):
        for loc in commands[t]:
            col = int(loc.split(",")[1])
            if col < min_commands_col:
                min_commands_col = col

    # find min column of events
    min_events_col = float("inf")
    for t in range(NumTraceTypes):
        for loc in events[t]:
            col = int(loc.split(",")[1])
            if col < min_events_col:
                min_events_col = col

    # The shift is the difference between the expected and observed leftmost
    # column for which trace was enabled (in 'events')
    colshift = min_events_col - min_commands_col

    # Shift all event keys by colshift
    new_events = []
    for t in range(NumTraceTypes):
        updated = {}
        for loc, l in events[t].items():
            row, col = map(int, loc.split(","))
            new_col = col - colshift
            new_key = f"{row},{new_col}"
            updated[new_key] = l
        new_events.append(updated)
    return new_events


def setup_trace_metadata(trace_events, pid_events):
    """
    This sets up the trace metadata and also assigned the unique pid that's referred
    eleswhere for each process (combination of tile(row,col) and trace type).
    NOTE: This assume the pid_events has already be analyzed and populated.
    """
    pid = 0
    for t in range(NumTraceTypes):
        for loc in pid_events[t]:  # return loc
            process_name_metadata(trace_events, pid, t, loc)
            for e in range(8):
                thread_name_metadata(trace_events, t, loc, pid, e, pid_events)
                pid_events[t][loc].append(pid)  # assign unique pid
            pid = pid + 1


def convert_commands_to_json(trace_events, commands, pid_events, output_file):
    """
    Convert commands to JSON format for trace events.

    commands:  list (idx = trace type, value = byte_stream_dict)
    byte_stream_dict: dict (key = row,col, value = list of commands)
    """
    # byte_stream_dict for each trace type.
    for [tt, byte_stream_dict] in enumerate(commands):  # tt = trace type

        for loc, command in byte_stream_dict.items():  # row,col with list of commands
            timer = 0  # TODO Some way to set this or sync this between trace types and row,col
            # timer on each execution is the time for the last execution
            # so we by default will increment it by 1 for each event
            if DEBUG:
                print(
                    "tt: "
                    + str(tt)
                    + ", loc: "
                    + str(loc)
                    + ", NUM_EVENTS: "
                    + str(NUM_EVENTS),
                    file=output_file,
                )

            if loc in pid_events[tt]:
                pid = pid_events[tt][loc][NUM_EVENTS]
            else:
                print(
                    "[ERROR] tile in",
                    loc,
                    "not found in trace packet data file (e.g trace.txt).",
                    file=sys.stderr,
                )
                tiles = []
                for tt_tmp in range(len(commands)):
                    for keys in pid_events[tt_tmp]:
                        tiles.append(keys)
                print("Defined tiles in design are at:", tiles, file=sys.stderr)
                print(
                    "Consider changing --colshift value if you think this is an error.",
                    file=sys.stderr,
                )
                sys.exit(1)

            active_events = dict()
            for i in range(8):  # 8 max events at a time
                active_events[i] = 0

            if DEBUG:
                print("num commands:", len(command), file=output_file)
            for c in command:
                t = c["type"]
                if "Single" in t:
                    event = c["event"]
                    cycles = int(c["cycles"])
                    timer = timer + 1
                    multiple_list = list()
                    multiple_list.append(c["event"])
                    deactivate_events(
                        multiple_list,
                        active_events,
                        timer,
                        cycles,
                        pid,
                        tt,
                        loc,
                        pid_events,
                        trace_events,
                    )
                    timer = timer + cycles
                    activate_event(
                        event,
                        tt,
                        loc,
                        timer,
                        pid,
                        active_events,
                        pid_events,
                        trace_events,
                    )

                elif "Multiple" in t:
                    cycles = int(c["cycles"])
                    timer = timer + 1
                    multiple_list = list()
                    for k in c.keys():
                        if "event" in k:
                            multiple_list.append(c[k])
                    deactivate_events(
                        multiple_list,
                        active_events,
                        timer,
                        cycles,
                        pid,
                        tt,
                        loc,
                        pid_events,
                        trace_events,
                    )
                    timer = timer + cycles

                    for k in c.keys():
                        if "event" in k:
                            activate_event(
                                c[k],
                                tt,
                                loc,
                                timer,
                                pid,
                                active_events,
                                pid_events,
                                trace_events,
                            )

                elif "Repeat" in t:
                    if (
                        cycles == 0
                    ):  # last event has cycles == 0 so we just extend it by the repaet count
                        timer = timer + int(c["repeats"])
                    else:
                        for repeats_cnt in range(int(c["repeats"])):
                            timer = timer + 1
                            deactivate_events(
                                multiple_list,
                                active_events,
                                timer,
                                cycles,
                                pid,
                                tt,
                                loc,
                                pid_events,
                                trace_events,
                            )
                            timer = timer + cycles
                            if len(multiple_list) > 1:
                                for k in c.keys():
                                    if "event" in k:
                                        activate_event(
                                            c[k],
                                            tt,
                                            loc,
                                            timer,
                                            pid,
                                            active_events,
                                            pid_events,
                                            trace_events,
                                        )
                            else:
                                activate_event(
                                    event,
                                    tt,
                                    loc,
                                    timer,
                                    pid,
                                    active_events,
                                    pid_events,
                                    trace_events,
                                )


# Main function for command-line usage
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input trace file", required=True)
    parser.add_argument("--mlir", help="mlir source file", required=True)
    parser.add_argument(
        "--colshift", help="column shift adjustment to source mlir", required=False
    )
    parser.add_argument("--output", help="Output json file", required=True)
    parser.add_argument("--debug", help="debug mode", required=False)
    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    # Command-line execution
    opts = parse_args()

    DEBUG = opts.debug
    if DEBUG:
        print("Debug mode enable\n")

    # set colshift based on optional argument
    colshift = int(opts.colshift) if opts.colshift else None

    # Use parse_and_save_trace for consistent processing
    try:
        parse_and_save_trace(
            trace_input=opts.input,
            mlir_input=opts.mlir,
            colshift=colshift,
            debug=DEBUG,
            output_file=opts.output,  # Pass output file path directly
        )

        print(f"Trace data successfully parsed and saved to {opts.output}")

    except Exception as e:
        print(f"ERROR: Failed to parse trace data: {e}", file=sys.stderr)
        exit(1)
