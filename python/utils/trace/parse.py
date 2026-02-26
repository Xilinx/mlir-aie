#!/usr/bin/env python3
# (c) Copyright 2026 Advanced Micro Devices, Inc.
import json
import argparse
import sys
import re
from collections import defaultdict

from aie.extras.util import find_ops
from aie.ir import Context, Module, Location
from aie.utils.trace.utils import (
    parity,
    extract_tile,
    parse_pkt_hdr_in_stream,
    trace_pkts_de_interleave,
    convert_to_byte_stream,
    convert_to_commands,
    trim_trace_pkts,
)
from aie.utils.trace.events import (
    NUM_TRACE_TYPES,
    PacketType,
    CoreEvent,
    MemEvent,
    ShimTileEvent,
    MemTileEvent,
    get_events_for_device,
)

import aie.dialects.aie as aiedialect
import aie.dialects.aiex as aiexdialect

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

NUM_EVENTS = 8  # number of events we can view per trace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input trace file", required=True)
    parser.add_argument("--mlir", help="mlir source file", required=True)
    parser.add_argument(
        "--colshift", help="column shift adjustment to source mlir", required=False
    )
    parser.add_argument("--output", help="Output json file", required=True)
    parser.add_argument("--debug", help="debug mode", required=False)
    # TODO tracelabels removed since we can have multiple sets of labels for each pkt_type & loc combination
    # parser.add_argument('--tracelabels',
    #         nargs='+',
    #         help='Labels for traces', required=False)
    return parser.parse_args(sys.argv[1:])


# Check for valid trace packets data
# 1) if only 1 trace packet
# 2) if first trace packet is all 0's
def check_for_valid_trace(filename, trace_pkts, of=None, debug=False):
    if debug and of:
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


def make_event_lists(commands):
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


# testing a flattening of repeat commands
def flatten_repeat_command(commands):
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


# Using trace_event_0 = 0x4B222125, trace_event_1 = 0x2D2C1A4F
def lookupEventNameInStr(event, pid, pid_events):
    # TODO Expand to other pid for multiple cores? even/odd
    # For now, we assume a single trace event and key based on that
    # in the future, the pid will be used to match the right events
    # print("pid_events[0]: ",pid_events[0])
    # print("event: ",event)
    # print("pid_events[0][event]: ",pid_events[0][int(event)])
    return lookup_event_name_by_code(pid_events[0][int(event)])

    # if pid == 0 or pid == 2: # Core trace
    #     if event == "0":
    #         return "KernelExecutesVectorInstruction"
    #     elif event == "1":
    #         return "KernelStarts"
    #     elif event == "2":
    #         return "KernelDone"
    #     elif event == "3":
    #         return "PortRunning0"
    #     elif event == "4":
    #         return "PortRunning1"
    #     elif event == "5":
    #         return "LockStall"
    #     elif event == "6":
    #         return "LockAcquireInstr"
    #     elif event == "7":lookupEventNameInstr
    #         return "LockReleaseInstr"
    # elif pid == 1 or pid == 3: # Memory trace
    #     if event == "0":
    #         return "S2mm0StartTask"
    #     elif event == "1":
    #         return "S2mm1StartTask"
    #     elif event == "2":
    #         return "Mm2s0StartTask"
    #     elif event == "3":
    #         return "Mm2s1StartTask"
    #     elif event == "4":
    #         return "S2mm0FinishedTask"
    #     elif event == "5":
    #         return "S2mm1FinishedTask"
    #     elif event == "6":
    #         return "Mm2s0FinishedTask"
    #     elif event == "7":
    #         return "Mm2s1FinishedTask"


# This function assert an end event for all active events if:
# 1) the cycles from the last event is > 0
# 2) active event is not in list of new events (multiples)
#
# if cycles > 0, deactivate all events
# if cycles == 0, deactivate all events except this one
#
# multiples - list of new active events
# active_events - list of existing active events
# timer - current running time
# cycles - # of cycles from last event
# pid -
# trace_type -
# loc -
# pid_events -
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
    events_module,
):
    for k in active_events.keys():  # an active event
        if cycles > 0 or (cycles == 0 and not k in multiples):
            # if not k in multiples: # active event it not in multiples list
            if active_events[k] > 0:
                trace_event = {
                    "name": lookup_event_name_by_type(
                        trace_type, pid_events[trace_type][loc][k], events_module
                    )
                }  # TODO remove
                trace_event["ts"] = timer
                trace_event["ph"] = "E"
                trace_event["pid"] = pid
                trace_event["tid"] = k
                trace_event["args"] = {}
                trace_events.append(trace_event)
            active_events[k] = 0


# Assert a begin siganl for the current event unless the event is still active
def activate_event(
    event, tt, loc, timer, pid, active_events, pid_events, trace_events, events_module
):
    try:
        if active_events[event] == 0:
            trace_event = {
                "name": lookup_event_name_by_type(
                    tt, pid_events[tt][loc][event], events_module
                )
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


# def convert_commands_to_json(trace_events, commands, pid, pid_events):
# TODO iterate over commands which is ...
#
# commands:  list (idx = trace type, value = byte_stream_dict)
# byte_stream_dict: dict (key = row,col, value = list of commands)
def convert_commands_to_json(
    trace_events, commands, pid_events, events_module, of=None, debug=False
):
    # byte_stream_dict for each trace type.
    for [tt, byte_stream_dict] in enumerate(commands):  # tt = trace type

        for loc, command in byte_stream_dict.items():  # row,col with list of commands
            timer = 0  # TODO Some way to set this or sync this between trace types and row,col
            # timer on each execution is the time for the last execution
            # so we by default will increment it by 1 for each event
            if debug and of:
                print(
                    "tt: "
                    + str(tt)
                    + ", loc: "
                    + str(loc)
                    + ", NUM_EVENTS: "
                    + str(NUM_EVENTS),
                    file=of,
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

            if debug and of:
                print("num commands:", len(command), file=of)
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
                        events_module,
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
                        events_module,
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
                        events_module,
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
                                events_module,
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
                                events_module,
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
                                            events_module,
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
                                    events_module,
                                )


def process_name_metadata(trace_events, pid, trace_type, loc):
    trace_event = {"name": "process_name"}
    trace_event["ph"] = "M"
    trace_event["pid"] = pid
    trace_event["args"] = {}

    pt = PacketType(trace_type)
    name = pt.name.lower()
    if name == "shimtile":
        name = "shim"

    trace_event["args"]["name"] = f"{name}_trace for tile{loc}"
    trace_events.append(trace_event)


# def thread_name_metadata(trace_events, pid, tid, pid_events):
def thread_name_metadata(
    trace_events, trace_type, loc, pid, tid, pid_events, events_module
):
    # def thread_name_metadata(trace_events, trace_type, pid, tid):
    trace_event = {"name": "thread_name"}
    trace_event["ph"] = "M"
    trace_event["pid"] = pid
    trace_event["tid"] = tid
    trace_event["args"] = {}
    # trace_event['args']['name'] = lookupEventNameInStr(str(tid), pid, pid_events)
    trace_event["args"]["name"] = lookup_event_name_by_type(
        trace_type, pid_events[trace_type][loc][tid], events_module
    )
    trace_events.append(trace_event)


# pid_events: list(idx=pkt_type, value=labels_dict)
# label_dict: dict(key=row,col, value=labels list)
# labels_list: list(idx=label idx, value=label code)
#
# This searches for npu.write32 and categorizes them based on address and row.
# It's probably not the best way to do it but it's the initial implementation.
# memtile and core/shim tiles have different addresses. For now, we distinguish
# between core and shim tile by row=0
def parse_mlir_trace_events(mlir_module_str, colshift=None):

    pid_events = list()
    for t in range(NUM_TRACE_TYPES):
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
        events_module = get_events_for_device(str(device))

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

        # print(f"write32: address={hex(address)}, row={row}, col={col}, value={hex(value)}")
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
                # print("Trace event 0 configured to be ",hex(value))
                pid_events[2][key][0] = value & 0xFF
                pid_events[2][key][1] = (value >> 8) & 0xFF
                pid_events[2][key][2] = (value >> 16) & 0xFF
                pid_events[2][key][3] = (value >> 24) & 0xFF
            else:  # core
                if pid_events[0].get(key) == None:
                    pid_events[0][key] = [0] * 8
                # print("Trace event 0 configured to be ",hex(value))
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
            # print("Trace event 0 configured to be ",hex(value))
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
            # print("Trace event 0 configured to be ",hex(value))
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
        # TODO shim event 0, 1 needs to also be defined

    # print("Found labels:\n")
    # for j in pid_events:
    #     print("row:",j['row'],", col: ",j['col'])
    #     print("0: ", j[0], "1: ", j[1], "2: ", j[2], "3: ", j[3])
    #     print("4: ", j[4], "5: ", j[5], "6: ", j[6], "7: ", j[7])
    return pid_events, events_module


def lookup_event_name_by_type(trace_type, code, events_module):
    if trace_type == PacketType.CORE:
        enum_class = events_module.CoreEvent
    elif trace_type == PacketType.MEM:
        enum_class = events_module.MemEvent
    elif trace_type == PacketType.SHIMTILE:
        enum_class = events_module.ShimTileEvent
    elif trace_type == PacketType.MEMTILE:
        enum_class = events_module.MemTileEvent
    else:
        return "Unknown"

    try:
        return enum_class(code).name
    except ValueError:
        pass
    return "Unknown"


# def lookup_event_name_by_code(code, traceType):
#     event = ""
#     # Core traces
#     if traceType == 0:
#         if code == 0x1:
#             event = "True"
#         elif code == 24: # 0x18:
#             event = "StreamStall"
#         elif code == 26: # 0x1A:
#             event = "LockStall"
#         elif code == 32: # 0x20, all events 33-45
#             event = "CoreProgramFlow"
#         elif code == 33: # 0x21:
#             event = "Event0"
#         elif code == 34: # 0x22:
#             event = "Event1"
#         elif code == 37: # 0x25:
#             event = "VectorInstr"
#         elif code == 38: # 0x26:
#             event = "InstrLoad"
#         elif code == 39: # 0x27:
#             event = "InstrStore"
#         elif code == 44: # 0x2C:
#             event = "LockAcquireInstr"
#         elif code == 45: # 0x2D:
#             event = "LockReleaseInstr"
#         elif code == 75: # 0x4B:
#             event = "PortRunning0"
#         elif code == 79: # 0x4F:
#             event = "PortRunning1"
#         else:
#             event = "Unknown"
#     # Mem traces
#     elif traceType == 1:
#         # TODO Need to define these
#         event = "Unknown"
#     else
#         event = "Unknown"

# match code:
#     case 0x1:
#         event = "True"
#     case 0x18:
#         event = "StreamStall"
#     case 0x1A:
#         event = "LockStall"
#     case 0x22:
#         event = "Event 1"
#     case 0x21:
#         event = "Event 0
#     case 0x25:
#         event = "VectorInstr"
#     case 0x2C:
#         event = "LockAcquireInstr"
#     case 0x2D:
#         event = "LockReleaseInstr"
#     case 0x4F:
#         event = "PortRunning1"
#     case 0x4B:
#         event = "PortRunning0"
#     case _:
#         event = "Unknown"
# return event


# lookupEventNameInStr(event, pid):
# # TODO Expand to other pid for multiple cores? even/odd
# if pid == 0 or pid == 2: # Core trace
#     if event == "0":
#         return "KernelExecutesVectorInstruction"
#     elif event == "1":
#         return "KernelStarts"
#     elif event == "2":
#         return "KernelDone"
#     elif event == "3":
#         return "PortRunning0"
#     elif event == "4":
#         return "PortRunning1"
#     elif event == "5":
#         return "LockStall"
#     elif event == "6":
#         return "LockAcquireInstr"
#     elif event == "7":
#         return "LockReleaseInstr"


# This sets up the trace metadata and also assigned the unique pid that's referred
# eleswhere for each process (combination of tile(row,col) and trace type).
# NOTE: This assume the pid_events has already be analyzed and populated.
def setup_trace_metadata(trace_events, pid_events, events_module):
    pid = 0
    for t in range(NUM_TRACE_TYPES):
        # for j in len(pid_events[i]):
        for loc in pid_events[t]:  # return loc
            process_name_metadata(trace_events, pid, t, loc)
            for e in range(8):
                thread_name_metadata(
                    trace_events, t, loc, pid, e, pid_events, events_module
                )
                pid_events[t][loc].append(pid)  # assign unique pid
            pid = pid + 1


# Attempt to align the starting column of trace in the design (from 'events')
# with the start first column observed in the trace ('commands'). This is needed
# because the runtime/firmware can start the design on any valid column
def align_column_start_index(events, commands):
    # find min column of commands
    min_commands_col = float("inf")
    for t in range(NUM_TRACE_TYPES):
        for loc in commands[t]:
            col = int(loc.split(",")[1])
            if col < min_commands_col:
                min_commands_col = col

    # find min column of events
    min_events_col = float("inf")
    for t in range(NUM_TRACE_TYPES):
        for loc in events[t]:
            col = int(loc.split(",")[1])
            if col < min_events_col:
                min_events_col = col

    # The shift is the difference between the expected and observed leftmost
    # column for which trace was enabled (in 'events')
    colshift = min_events_col - min_commands_col

    # Shift all event keys by colshift
    new_events = []
    for t in range(NUM_TRACE_TYPES):
        updated = {}
        for loc, l in events[t].items():
            row, col = map(int, loc.split(","))
            new_col = col - colshift
            new_key = f"{row},{new_col}"
            updated[new_key] = l
        new_events.append(updated)
    return new_events


# ------------------------------------------------------------------------------
# Library API
# ------------------------------------------------------------------------------


def create_timeline_from_commands(
    commands, pid_events, output_file="trace_timeline.png", title="Trace Timeline"
):
    """
    Create a timeline visualization directly from parsed commands.

    This is more efficient than create_timeline_from_events() as it works directly
    with the command representation without converting to Trace Event Format first.

    Args:
        commands: list of command dictionaries (output from convert_to_commands)
                  Format: list(idx=trace_type, value=dict(key=row,col, value=list of commands))
        pid_events: parsed event configuration (output from parse_mlir_trace_events)
                    Format: list(idx=trace_type, value=dict(key=row,col, value=list of event codes))
        output_file: path to save the PNG file (default: "trace_timeline.png")
        title: title for the timeline chart (default: "Trace Timeline")

    Returns:
        bool: True if visualization was created successfully, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("WARNING: matplotlib not available, skipping timeline visualization")
        return False

    # Build timeline data from commands
    all_intervals = []
    process_info = {}  # pid -> (trace_type, location)
    thread_info = {}  # (pid, tid) -> event_name

    pid = 0
    for trace_type in range(len(commands)):
        for loc, command_list in commands[trace_type].items():
            # Setup process and thread metadata
            process_info[pid] = (trace_type, loc)

            # Get event names for this location
            if loc in pid_events[trace_type]:
                for event_idx in range(min(8, len(pid_events[trace_type][loc]))):
                    event_code = pid_events[trace_type][loc][event_idx]
                    event_name = lookup_event_name_by_type(trace_type, event_code)
                    thread_info[(pid, event_idx)] = event_name

            # Parse commands to build intervals
            timer = 0
            active_events = {}
            for i in range(8):
                active_events[i] = None  # None means inactive, value is start time

            cycles = 0
            multiple_list = []

            for cmd in command_list:
                cmd_type = cmd["type"]

                if "Single" in cmd_type:
                    event = cmd["event"]
                    cycles = int(cmd["cycles"])
                    timer += 1

                    # End active events
                    multiple_list = [event]
                    for k in active_events.keys():
                        if cycles > 0 or (cycles == 0 and k not in multiple_list):
                            if active_events[k] is not None:
                                # Create interval
                                all_intervals.append(
                                    {
                                        "pid": pid,
                                        "tid": k,
                                        "name": thread_info.get((pid, k), f"Event_{k}"),
                                        "start": active_events[k],
                                        "end": timer,
                                        "duration": timer - active_events[k],
                                    }
                                )
                                active_events[k] = None

                    timer += cycles

                    # Start new event
                    if active_events[event] is None:
                        active_events[event] = timer

                elif "Multiple" in cmd_type:
                    cycles = int(cmd["cycles"])
                    timer += 1

                    # Find all events in this multiple
                    multiple_list = []
                    for key in cmd.keys():
                        if "event" in key:
                            multiple_list.append(cmd[key])

                    # End events not in multiple list
                    for k in active_events.keys():
                        if cycles > 0 or (cycles == 0 and k not in multiple_list):
                            if active_events[k] is not None:
                                all_intervals.append(
                                    {
                                        "pid": pid,
                                        "tid": k,
                                        "name": thread_info.get((pid, k), f"Event_{k}"),
                                        "start": active_events[k],
                                        "end": timer,
                                        "duration": timer - active_events[k],
                                    }
                                )
                                active_events[k] = None

                    timer += cycles

                    # Start all events in multiple list
                    for event in multiple_list:
                        if active_events[event] is None:
                            active_events[event] = timer

                elif "Repeat" in cmd_type:
                    # Handle repeat commands
                    if cycles == 0:
                        timer += int(cmd["repeats"])
                    else:
                        for _ in range(int(cmd["repeats"])):
                            timer += 1
                            for k in active_events.keys():
                                if cycles > 0 or (
                                    cycles == 0 and k not in multiple_list
                                ):
                                    if active_events[k] is not None:
                                        all_intervals.append(
                                            {
                                                "pid": pid,
                                                "tid": k,
                                                "name": thread_info.get(
                                                    (pid, k), f"Event_{k}"
                                                ),
                                                "start": active_events[k],
                                                "end": timer,
                                                "duration": timer - active_events[k],
                                            }
                                        )
                                        active_events[k] = None

                            timer += cycles

                            # Restart events
                            if len(multiple_list) > 1:
                                for event in multiple_list:
                                    if active_events[event] is None:
                                        active_events[event] = timer
                            elif len(multiple_list) == 1:
                                event = multiple_list[0]
                                if active_events[event] is None:
                                    active_events[event] = timer

            pid += 1

    if not all_intervals:
        print("No trace intervals found in commands!")
        return False

    # Create visualization using same logic as create_timeline_from_events
    lanes = {}
    lane_idx = 0
    for pid_val in sorted(set(i["pid"] for i in all_intervals)):
        for tid in sorted(set(i["tid"] for i in all_intervals if i["pid"] == pid_val)):
            lanes[(pid_val, tid)] = lane_idx
            lane_idx += 1

    fig, ax = plt.subplots(figsize=(20, max(8, min(lane_idx * 0.5, 20))))

    min_time = min(i["start"] for i in all_intervals)
    max_time = max(i["end"] for i in all_intervals)
    time_range = max_time - min_time

    ax.set_xlim(min_time - time_range * 0.02, max_time + time_range * 0.02)

    event_colors = {}
    color_palette = plt.cm.tab20.colors
    color_idx = 0

    for interval in all_intervals:
        lane = lanes[(interval["pid"], interval["tid"])]
        event_name = interval["name"]

        if event_name not in event_colors:
            event_colors[event_name] = color_palette[color_idx % len(color_palette)]
            color_idx += 1

        color = event_colors[event_name]

        rect = mpatches.Rectangle(
            (interval["start"], lane - 0.4),
            interval["duration"],
            0.8,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.7,
        )
        ax.add_patch(rect)

        label_threshold = time_range / 100
        if interval["duration"] > label_threshold:
            ax.text(
                interval["start"] + interval["duration"] / 2,
                lane,
                event_name.replace("INSTR_", "").replace("DMA_", "").replace("_", " "),
                ha="center",
                va="center",
                fontsize=7,
                weight="bold",
                clip_on=True,
            )

    ax.set_ylim(-0.5, lane_idx - 0.5)
    ax.set_yticks(range(lane_idx))

    # Create lane labels with trace type and location
    lane_labels = []
    for (pid_val, tid), lane in sorted(lanes.items(), key=lambda x: x[1]):
        if pid_val in process_info:
            trace_type, loc = process_info[pid_val]
            type_names = ["Core", "Mem", "Shim", "MemTile"]
            type_name = (
                type_names[trace_type]
                if trace_type < len(type_names)
                else f"Type{trace_type}"
            )
            thread_name = thread_info.get((pid_val, tid), f"Event_{tid}")
            lane_labels.append(f"{type_name} {loc}\n{thread_name}")
        else:
            lane_labels.append(f"PID {pid_val}\nTID {tid}")

    ax.set_yticklabels(lane_labels, fontsize=8)
    ax.set_xlabel("Time (cycles)", fontsize=10)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    event_counts = defaultdict(int)
    for interval in all_intervals:
        event_counts[interval["name"]] += 1

    top_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    legend_patches = [
        mpatches.Patch(color=event_colors[name], label=f"{name} ({count})")
        for name, count in top_events
        if name in event_colors
    ]

    if legend_patches:
        ax.legend(
            handles=legend_patches,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
            framealpha=0.9,
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Trace visualization saved to: {output_file}")

    print(f"\\nTrace Statistics:")
    print(f"  Total intervals: {len(all_intervals)}")
    print(f"  Total lanes: {lane_idx}")
    print(f"  Time range: {min_time} - {max_time} cycles")
    print(f"  Duration: {time_range} cycles")

    print("\\nTop Events:")
    for name, count in top_events[:10]:
        print(f"  {name}: {count}")

    plt.close()
    return True


def create_timeline_from_events(
    events, output_file="trace_timeline.png", title="Trace Timeline"
):
    """
    Create a timeline visualization of trace events.

    Args:
        events: list of trace events in Trace Event Format
        output_file: path to save the PNG file (default: "trace_timeline.png")
        title: title for the timeline chart (default: "Trace Timeline")

    Returns:
        bool: True if visualization was created successfully, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("WARNING: matplotlib not available, skipping timeline visualization")
        return False

    # Extract metadata and organize events
    processes = {}
    threads = {}
    event_list = []

    for entry in events:
        if entry["ph"] == "M":  # Metadata
            if entry["name"] == "process_name":
                processes[entry["pid"]] = entry["args"]["name"]
            elif entry["name"] == "thread_name":
                threads[(entry["pid"], entry["tid"])] = entry["args"]["name"]
        elif entry["ph"] in ["B", "E"]:  # Begin or End events
            event_list.append(entry)

    # Group events into intervals (B -> E pairs)
    active_events = {}
    intervals = []

    for event in event_list:
        key = (event["pid"], event["tid"], event["name"])

        if event["ph"] == "B":  # Begin
            active_events[key] = event["ts"]
        elif event["ph"] == "E":  # End
            if key in active_events:
                start_ts = active_events[key]
                end_ts = event["ts"]
                intervals.append(
                    {
                        "pid": event["pid"],
                        "tid": event["tid"],
                        "name": event["name"],
                        "start": start_ts,
                        "end": end_ts,
                        "duration": end_ts - start_ts,
                    }
                )
                del active_events[key]

    if not intervals:
        print("No trace intervals found!")
        return False

    # Create unique lanes for each (pid, tid) combination
    lanes = {}
    lane_idx = 0
    for pid in sorted(set(i["pid"] for i in intervals)):
        for tid in sorted(set(i["tid"] for i in intervals if i["pid"] == pid)):
            lanes[(pid, tid)] = lane_idx
            lane_idx += 1

    # Create figure
    fig, ax = plt.subplots(figsize=(20, max(8, min(lane_idx * 0.5, 20))))

    # Calculate time range
    min_time = min(i["start"] for i in intervals)
    max_time = max(i["end"] for i in intervals)
    time_range = max_time - min_time

    ax.set_xlim(min_time - time_range * 0.02, max_time + time_range * 0.02)

    # Color map for different event types
    event_colors = {}
    color_palette = plt.cm.tab20.colors
    color_idx = 0

    # Plot intervals
    for interval in intervals:
        lane = lanes[(interval["pid"], interval["tid"])]
        event_name = interval["name"]

        # Assign color to event type
        if event_name not in event_colors:
            event_colors[event_name] = color_palette[color_idx % len(color_palette)]
            color_idx += 1

        color = event_colors[event_name]

        # Draw rectangle for the interval
        rect = mpatches.Rectangle(
            (interval["start"], lane - 0.4),
            interval["duration"],
            0.8,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.7,
        )
        ax.add_patch(rect)

        # Add text label if duration is large enough
        label_threshold = time_range / 100
        if interval["duration"] > label_threshold:
            ax.text(
                interval["start"] + interval["duration"] / 2,
                lane,
                event_name.replace("INSTR_", "").replace("DMA_", "").replace("_", " "),
                ha="center",
                va="center",
                fontsize=7,
                weight="bold",
                clip_on=True,
            )

    # Set up axes
    ax.set_ylim(-0.5, lane_idx - 0.5)
    ax.set_yticks(range(lane_idx))

    # Create lane labels
    lane_labels = []
    for (pid, tid), lane in sorted(lanes.items(), key=lambda x: x[1]):
        proc_name = processes.get(pid, f"Process {pid}")
        thread_name = threads.get((pid, tid), f"Thread {tid}")
        lane_labels.append(f"{proc_name}\n{thread_name}")

    ax.set_yticklabels(lane_labels, fontsize=8)
    ax.set_xlabel("Time (cycles)", fontsize=10)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Add legend for event types
    event_counts = defaultdict(int)
    for interval in intervals:
        event_counts[interval["name"]] += 1

    top_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    legend_patches = [
        mpatches.Patch(color=event_colors[name], label=f"{name} ({count})")
        for name, count in top_events
        if name in event_colors
    ]

    if legend_patches:
        ax.legend(
            handles=legend_patches,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
            framealpha=0.9,
        )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Trace visualization saved to: {output_file}")

    # Print statistics
    print(f"\nTrace Statistics:")
    print(f"  Total intervals: {len(intervals)}")
    print(f"  Total lanes: {lane_idx}")
    print(f"  Time range: {min_time} - {max_time} cycles")
    print(f"  Duration: {time_range} cycles")

    print("\nTop Events:")
    for name, count in top_events[:10]:
        print(f"  {name}: {count}")

    plt.close()
    return True


def parse_trace(
    trace_buffer, mlir_module_str, colshift=None, debug=False, generate_timeline=None
):
    """
    Parse AIE trace buffer and return trace events as list in Trace Event Format

    Args:
        trace_buffer: numpy array containing trace data (uint32 words)
        mlir_module_str: string containing MLIR module with trace configuration
        colshift: optional column shift adjustment (int or None for auto-align)
        debug: enable debug output (default: False)
        generate_timeline: optional path to generate timeline PNG (default: None)
                          If provided, creates visualization directly from commands

    Returns:
        list: trace events in Trace Event Format
    """

    # Convert numpy array to list of hex strings (format expected by existing functions)
    trace_pkts = []
    for word in trace_buffer:
        # Convert uint32 to 8-character hex string (lowercase, no '0x' prefix)
        hex_str = f"{int(word):08x}"
        trace_pkts.append(hex_str)

    # Parse MLIR to extract event configuration
    pid_events, events_module = parse_mlir_trace_events(mlir_module_str, colshift)

    # Check for valid trace
    if not check_for_valid_trace("<numpy_array>", trace_pkts, of=None, debug=debug):
        raise ValueError("Invalid trace data: empty or all zeros")

    # Trim trailing empty packets
    trimmed_trace_pkts = trim_trace_pkts(trace_pkts)

    # De-interleave packets by type and location
    trace_pkts_sorted = trace_pkts_de_interleave(trimmed_trace_pkts)

    # Convert to byte streams
    byte_streams = convert_to_byte_stream(trace_pkts_sorted)

    # Convert byte streams to command dictionaries
    commands = convert_to_commands(byte_streams, False)

    # Auto-align column indices if colshift not provided
    if colshift is None:
        pid_events = align_column_start_index(pid_events, commands)

    # Optional: Generate timeline directly from commands (more efficient)
    if generate_timeline:
        create_timeline_from_commands(
            commands,
            pid_events,
            output_file=generate_timeline,
            title="AIE Trace Timeline",
        )

    # Initialize trace events list
    trace_events = []

    # Setup metadata (process names, thread names, assign PIDs)
    setup_trace_metadata(trace_events, pid_events, events_module)

    # Convert commands to Chrome Trace Event Format
    convert_commands_to_json(
        trace_events, commands, pid_events, events_module, of=None, debug=debug
    )

    return trace_events


# ------------------------------------------------------------------------------
# Script execution start - Open trace file and convert to commands
# ------------------------------------------------------------------------------


def main():
    """Command-line interface entry point"""
    opts = parse_args()

    DEBUG = opts.debug
    if DEBUG:
        print("Debug mode enable\n")

    # set colshift based on optional argument
    colshift = int(opts.colshift) if opts.colshift else None

    try:
        with open(opts.input, "r") as f:
            # Create array of trace packets
            trace_pkts = f.read().split("\n")
    except Exception:
        print(
            "ERROR:",
            opts.input,
            "could not be opened. Check for valid trace source file.",
        )
        sys.exit(1)

    try:
        with open(opts.mlir, "r") as mf:
            mlir_module_str = mf.read()
        pid_events, events_module = parse_mlir_trace_events(mlir_module_str, colshift)
    except Exception as e:
        print("ERROR:", opts.mlir, "could not be opened. Check for valid MLIR file.", e)
        sys.exit(1)

    try:
        of = open(opts.output, "w")
    except Exception:
        print(
            "ERROR:",
            opts.mlir,
            "could not be opened. Check for valid output JSON file.",
        )
        sys.exit(1)

    if DEBUG:
        print("DEBUG mode enabled:", file=of)
        print("pkt type 0: core tile", file=of)
        print("pkt type 1: core mem tile", file=of)
        print("pkt type 2: shim tile", file=of)
        print("pkt type 3: mem tile", file=of)
        print("", file=of)
        print("DEBUG: trace_pkts", file=of)
        print(trace_pkts, file=of)
        print("", file=of)

        print("DEBUG: pid events\n", file=of)
        # print(pid_events, file=of)
        for idx, dict_i in enumerate(pid_events):
            print("pkt type", idx, ":", file=of)
            for key, value in dict_i.items():
                print(key, value, file=of)
        print("", file=of)

    if not check_for_valid_trace(opts.input, trace_pkts, of, DEBUG):
        sys.exit(1)

    trimmed_trace_pkts = trim_trace_pkts(trace_pkts)
    if DEBUG:
        lines_removed = len(trace_pkts) - len(trimmed_trace_pkts)
        print("DEBUG: trimmed ", lines_removed, " lines", file=of)

    trace_pkts_sorted = trace_pkts_de_interleave(trimmed_trace_pkts)

    if DEBUG:
        print("DEBUG: trace_pkts_sorted", file=of)
        for idx, dict_i in enumerate(trace_pkts_sorted):
            print("pkt type", idx, ":", file=of)
            for key, value in dict_i.items():
                print(key, value, file=of)
        print("", file=of)

    byte_streams = convert_to_byte_stream(trace_pkts_sorted)

    if DEBUG:
        print("DEBUG: byte stream", file=of)
        for idx, dict_i in enumerate(byte_streams):
            print("pkt type", idx, ":", file=of)
            for key, value in dict_i.items():
                print(key, value, file=of)
        print("", file=of)

    commands_0 = convert_to_commands(byte_streams, False)

    if DEBUG:
        print("DEBUG: commands_0", file=of)
        for idx, dict_i in enumerate(commands_0):
            print("pkt type", idx, ":", file=of)
            for key, commands in dict_i.items():
                print(key, file=of)
                for i in commands:
                    print("\t", i, file=of)
        print("", file=of)

    if colshift is None:
        pid_events = align_column_start_index(pid_events, commands_0)

    trace_events = list()

    setup_trace_metadata(trace_events, pid_events, events_module)

    convert_commands_to_json(
        trace_events, commands_0, pid_events, events_module, of, DEBUG
    )

    print(json.dumps(trace_events).replace("'", '"').replace(", {", ",\n{"), file=of)

    of.close()


if __name__ == "__main__":
    main()
