#!/usr/bin/env python3
import json
import argparse
import sys
import re

from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent

# Number of different trace types, currently 4
# core:    pkt type 0
# mem:     pkt type 1
# shim:   pkt type 2
# memtile: pkt type 3
NumTraceTypes = 4
NUM_EVENTS = 8  # number of events we can view per trace

# DEBUG = False
# DEBUG = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input trace file", required=True)
    parser.add_argument("--mlir", help="mlir source file", required=True)
    parser.add_argument(
        "--colshift", help="column shift adjustment to source mlir", required=False
    )
    parser.add_argument("--output", help="Output json file",required=True)
    parser.add_argument("--debug", help="debug mode", required=False)
    # TODO tracelabels removed since we can have multiple sets of labels for each pkt_type & loc combination
    # parser.add_argument('--tracelabels',
    #         nargs='+',
    #         help='Labels for traces', required=False)
    return parser.parse_args(sys.argv[1:])


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

    # core_streams = dict()   # pkt type 0
    # mem_stream = dict()     # pkt type 1
    # shim_stream = dict()   # pkt type 2
    # memtile_stream = dict() # pkt type 3

    # index lists based on row/col and if its not null, that means it already exists

    curr_pkt_type = 0
    curr_loc = ""
    curr_vld = False  # only used in the beginning

    # print(len(word_stream))
    for i in range(len(word_stream)):
        if word_stream[i] == "":
            break  # TODO Assumes a blank line is the last line
        if (i % 8) == 0:
            # print(str(i)+':'+word_stream[i])
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
            # Crate a list for the loc if it doesn't exist
            curr_vld = True
        else:
            if (
                curr_vld
            ):  # ignores first 8 chunks of data is pkt hdr was invalid. TODO Is this right?
                # or shoudl we require valid header in first chunk of data
                trace_pkts_sorted[curr_pkt_type][curr_loc].append(
                    word_stream[i]
                )  # TODO assume curr_pkt_type is valid
                # for tt in range(NumTraceTypes):
                #     if curr_pkt_type == tt:
                #         toks_list[tt][curr_loc].append(word_stream[i])
    return trace_pkts_sorted


# Convert trace packets into byte streams
#
# toks_list is a list of toks dictionaries where each dictionary is a type (core, mem, shim, memtile)
# each dictionary key is a tile location (row,col) whose value is a list of stream data
def convert_to_byte_stream(toks_list):
    byte_stream_list = list()
    for l in toks_list:
        byte_stream_dict = dict()
        for loc, stream in l.items():
            # byte_stream = list()
            byte_stream_dict[loc] = list()
            f = ["", "a5a5a5a5"]
            toks = [t for t in stream if not t in f]
            events = [int(t, 16) for t in toks]
            for event in events:
                for top in range(4):
                    byte = 3 - top
                    opcode = event >> (byte * 8) & 0xFF
                    byte_stream_dict[loc].append(opcode)
        # for key, value in l.items():
        #     # byte_stream = list()
        #     byte_stream_dict[key] = list()
        #     f = ['', 'a5a5a5a5']
        #     toks = [t for t in value if not t in f]
        #     events = [int(t,16) for t in toks]
        #     for (i,event) in enumerate(events):
        #         if ((i % 8) == 0): # assumed hdr every 8 words and ignores it
        #             pass
        #         else: # breaks line into list of bytes
        #             for top in range(4):
        #                 byte = 3-top
        #                 opcode = (event >> (byte * 8) & 0xff)
        #                 byte_stream_dict[key].append(opcode)
        byte_stream_list.append(byte_stream_dict)
    return byte_stream_list


# Convert byte streams to equivalent packet commands
#
# byte_stream_list: list (idx = trace type, value = word_stream_dict)
# word_stream_dict: dict (key = row,col, value = list of words)
#
# return commands:  list (idx = trace type, value = byte_stream_dict)
# byte_stream_dict: dict (key = row,col, value = list of commands)
#
# command: dict
#   keys: type (Single0/1, Multiple0/1/2, Start, Repeat0/1, EventSync)
#         event (integer value)
#         cycles (integer value)
#         event# (integer value matching event number #)
#         repeats (integer value)
def convert_to_commands(byte_stream_list, zero=True):
    # commands = dict()
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
                                com["event" + str(i)] = (
                                    i  # TODO is this how event# is stored in IR?
                                )
                        commands[t][key].append(com)
                        cursor = cursor + 2
                    if (byte_stream[cursor] & 0b11111100) == 0b11010000:
                        # TODO Don't we need to extract events here?
                        # print("Multiple1")
                        com = {"type": "Multiple1"}
                        cycles = (byte_stream[cursor + 1] & 0b11) << 8
                        com["cycles"] = cycles + (byte_stream[cursor + 2])
                        events = (byte_stream[cursor] & 0b11) << 6
                        events = events + (byte_stream[cursor + 1] >> 2)
                        for i in range(0, 8):
                            e = (events >> i) & 0b1
                            if e:
                                com["event" + str(i)] = (
                                    i  # TODO is this how event# is stored in IR?
                                )
                        commands[t][key].append(com)
                        cursor = cursor + 3
                    if (byte_stream[cursor] & 0b11111100) == 0b11010100:
                        # TODO Don't we need to extract events here?
                        # print("Multiple2")
                        com = {"type": "Multiple2"}
                        cycles = (byte_stream[cursor + 1] & 0b11) << 16
                        cycles = cycles + ((byte_stream[cursor + 2]) << 8)
                        com["cycles"] = cycles + (byte_stream[cursor + 3])
                        events = (byte_stream[cursor] & 0b11) << 6
                        events = events + (byte_stream[cursor + 1] >> 2)
                        for i in range(0, 8):
                            e = (events >> i) & 0b1
                            if e:
                                com["event" + str(i)] = (
                                    i  # TODO is this how event# is stored in IR?
                                )
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
    multiples, active_events, timer, cycles, pid, trace_type, loc, pid_events
):
    for k in active_events.keys():  # an active event
        if cycles > 0 or (cycles == 0 and not k in multiples):
            # if not k in multiples: # active event it not in multiples list
            if active_events[k] > 0:
                trace_event = {
                    "name": lookup_event_name_by_type(
                        trace_type, pid_events[trace_type][loc][k]
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
def activate_event(event, tt, loc, timer, pid, active_events, pid_events, trace_events):
    try:
        if active_events[event] == 0:
            trace_event = {
                "name": lookup_event_name_by_type(
                    tt, pid_events[tt][loc][event]
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
def convert_commands_to_json(trace_events, commands, pid_events, of):
    # for bsd in commands: # byte_stream_dict for each trace type. TODO how to get index of bsd?
    for tt in range(
        len(commands)
    ):  # byte_stream_dict for each trace type. TODO how to get index of bsd?
        byte_stream_dict = commands[tt]

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
                    file=of
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
            for i in range(8): # 8 max events at a time
                active_events[i] = 0

            if DEBUG:
                print("num commands:",len(command), file=of)
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
                    )
                    timer = timer + cycles
                    activate_event(event, tt, loc, timer, pid, active_events, pid_events, trace_events)

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
                    )
                    timer = timer + cycles

                    for k in c.keys():
                        if "event" in k:
                            activate_event(c[k], tt, loc, timer, pid, active_events, pid_events, trace_events)

                elif "Repeat" in t:
                    if cycles == 0: # last event has cycles == 0 so we just extend it by the repaet count
                        timer = timer + int(c["repeats"])
                    else:
                        deactivate_events(
                            multiple_list,
                            active_events,
                            timer,
                            cycles,
                            pid,
                            tt,
                            loc,
                            pid_events,
                        )
                        timer = timer + cycles
                        if len(multiple_list) > 1:
                            for k in c.keys():
                                if "event" in k:
                                    activate_event(c[k], tt, loc, timer, pid, active_events, pid_events, trace_events)
                        else:
                            activate_event(event, tt, loc, timer, pid, active_events, pid_events, trace_events)


def process_name_metadata(trace_events, pid, trace_type, loc):
    trace_event = {"name": "process_name"}
    trace_event["ph"] = "M"
    trace_event["pid"] = pid
    trace_event["args"] = {}
    # if (pid == 0 or pid == 2):
    if trace_type == 0:
        trace_event["args"]["name"] = "core_trace for tile" + str(loc)
    # if (pid == 1 or pid == 3):
    elif trace_type == 1:
        trace_event["args"]["name"] = "mem_trace for tile" + str(loc)
    elif trace_type == 2:
        trace_event["args"]["name"] = "shim_trace for tile" + str(loc)
    elif trace_type == 3:
        trace_event["args"]["name"] = "memtile_trace for tile" + str(loc)

    trace_events.append(trace_event)


# def thread_name_metadata(trace_events, pid, tid, pid_events):
def thread_name_metadata(trace_events, trace_type, loc, pid, tid, pid_events):
    # def thread_name_metadata(trace_events, trace_type, pid, tid):
    trace_event = {"name": "thread_name"}
    trace_event["ph"] = "M"
    trace_event["pid"] = pid
    trace_event["tid"] = tid
    trace_event["args"] = {}
    # trace_event['args']['name'] = lookupEventNameInStr(str(tid), pid, pid_events)
    trace_event["args"]["name"] = lookup_event_name_by_type(
        trace_type, pid_events[trace_type][loc][tid]
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
def parse_mlir_trace_events(lines):
    # arg can be column, row, address or value
    # 1: arg: 2: val
    # 3: arg, 4: val
    # 5: arg, 6: val
    # 6: arg, 7: val

    # 1: arg: 2: 0x, 3: val
    # 4: arg, 5: 0x, 6: val
    # 7: arg, 8: 0x, 9: val
    # 10: arg, 11: 0x, 12: val

    # TODO Need to check if this line is commented out, check for // ? (harder to check of /* */)
    # TODO Need to support value in hex with 0x or decimal
    # pattern = r"AIEX.npu.write32\s*\{\s*(\w+)\s*=\s*(\d+)\s*:\s*\w+\s*,\s*(\w+)\s*=\s*(\d+)\s*:\s*\w+\s*,\s*(\w+)\s*=\s*(\w+)\s*:\s*\w+\s*,\s*(\w+)\s*=\s*(\w+)\s*:\s*\w+\s*\}"
    # pattern = r"AIEX.npu.write32\s*\{\s*(\w+)\s*=\s*(0x)?(\w+)\s*:\s*\w+\s*,\s*(\w+)\s*=\s*(0x)?(\w+)\s*:\s*\w+\s*,\s*(\w+)\s*=\s*(0x)?(\w+)\s*:\s*\w+\s*,\s*(\w+)\s*=\s*(0x)?(\w+)\s*:\s*\w+\s*\}"
    pattern = r"aiex.npu.write32\s*\{\s*(\w+)\s*=\s*(0x)?(\w+)\s*:\s*\w+\s*,\s*(\w+)\s*=\s*(0x)?(\w+)\s*:\s*\w+\s*,\s*(\w+)\s*=\s*(0x)?(\w+)\s*:\s*\w+\s*,\s*(\w+)\s*=\s*(0x)?(\w+)\s*:\s*\w+\s*\}"

    pid_events = list()
    for t in range(NumTraceTypes):
        pid_events.append(dict())

    for i in range(len(lines)):
        result = re.search(pattern, lines[i])
        if result:  # match found
            address = 0
            row = 0
            col = 0
            value = 0
            for i2 in range(4):
                var = result.group(3 * i2 + 1)
                if var == "address":
                    if result.group(3 * i2 + 2) == "0x":
                        address = int(result.group(3 * i2 + 3), 16)
                    else:  # assume ''
                        address = int(result.group(3 * i2 + 3))
                elif var == "row":
                    # TODO assume no 0x
                    row = int(result.group(3 * i2 + 3))
                elif var == "column":
                    col = int(result.group(3 * i2 + 3)) + colshift
                    # col = 1 if col == 0 else col
                elif var == "value":
                    if result.group(3 * i2 + 2) == "0x":
                        value = int(result.group(3 * i2 + 3), 16)
                    else:  # assume ''
                        value = int(result.group(3 * i2 + 3))

                # var = result.group(2*i2+1)
                # if(var == "address"):
                #     address = int(result.group(2*i2+2))
                # elif(var == "row"):
                #     row = int(result.group(2*i2+2))
                # elif(var == "column"):
                #     col = int(result.group(2*i2+2)) + colshift
                #     col = 1 if col == 0 else col
                # elif(var == "value"):
                #     value = int(result.group(2*i2+2))

            # labels_dict = dict()
            # labels_dict['row'] = row
            # labels_dict['col'] = col
            key = str(row) + "," + str(col)

            # core event 0
            if address == 0x340E0:  # 213216, match ignoring case
                if row == 0:  # shim
                    if pid_events[2].get(key) == None:
                        pid_events[2][key] = [
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]  # TODO no better way to init this?
                    # print("Trace event 0 configured to be ",hex(value))
                    pid_events[2][key][0] = value & 0xFF
                    pid_events[2][key][1] = (value >> 8) & 0xFF
                    pid_events[2][key][2] = (value >> 16) & 0xFF
                    pid_events[2][key][3] = (value >> 24) & 0xFF
                else:  # core
                    if pid_events[0].get(key) == None:
                        pid_events[0][key] = [
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]  # TODO no better way to init this?
                    # print("Trace event 0 configured to be ",hex(value))
                    pid_events[0][key][0] = value & 0xFF
                    pid_events[0][key][1] = (value >> 8) & 0xFF
                    pid_events[0][key][2] = (value >> 16) & 0xFF
                    pid_events[0][key][3] = (value >> 24) & 0xFF
            # core event 1
            elif address == 0x340E4:  # 213220, match ignoring case
                if row == 0:  # shim
                    if pid_events[2].get(key) == None:
                        pid_events[2][key] = [
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]  # TODO no better way to init this?
                    pid_events[2][key][4] = value & 0xFF
                    pid_events[2][key][5] = (value >> 8) & 0xFF
                    pid_events[2][key][6] = (value >> 16) & 0xFF
                    pid_events[2][key][7] = (value >> 24) & 0xFF
                else:  # core
                    if pid_events[0].get(key) == None:
                        pid_events[0][key] = [
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]  # TODO no better way to init this?
                    pid_events[0][key][4] = value & 0xFF
                    pid_events[0][key][5] = (value >> 8) & 0xFF
                    pid_events[0][key][6] = (value >> 16) & 0xFF
                    pid_events[0][key][7] = (value >> 24) & 0xFF
            # mem event 0
            elif address == 0x140E0:  # 82144
                if pid_events[1].get(key) == None:
                    pid_events[1][key] = [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]  # TODO no better way to init this?
                # print("Trace event 0 configured to be ",hex(value))
                pid_events[1][key][0] = value & 0xFF
                pid_events[1][key][1] = (value >> 8) & 0xFF
                pid_events[1][key][2] = (value >> 16) & 0xFF
                pid_events[1][key][3] = (value >> 24) & 0xFF
            # mem event 1
            elif address == 0x140E4:  # 82148
                if pid_events[1].get(key) == None:
                    pid_events[1][key] = [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]  # TODO no better way to init this?
                pid_events[1][key][4] = value & 0xFF
                pid_events[1][key][5] = (value >> 8) & 0xFF
                pid_events[1][key][6] = (value >> 16) & 0xFF
                pid_events[1][key][7] = (value >> 24) & 0xFF
            # memtile event 0
            elif address == 0x940E0:  # 606432
                if pid_events[3].get(key) == None:
                    pid_events[3][key] = [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]  # TODO no better way to init this?
                # print("Trace event 0 configured to be ",hex(value))
                pid_events[3][key][0] = value & 0xFF
                pid_events[3][key][1] = (value >> 8) & 0xFF
                pid_events[3][key][2] = (value >> 16) & 0xFF
                pid_events[3][key][3] = (value >> 24) & 0xFF
            # memtile event 1
            elif address == 0x940E4:  # 606436
                if pid_events[3].get(key) == None:
                    pid_events[3][key] = [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]  # TODO no better way to init this?
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
    return pid_events


def lookup_event_name_by_type(trace_type, code):
    # def lookup_event_name_by_type(trace_type, loc, event, pid_events):
    event = ""
    # code = pid_events[trace_type][loc][event]
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
def setup_trace_metadata(trace_events, pid_events):
    pid = 0
    for t in range(NumTraceTypes):
        # for j in len(pid_events[i]):
        for loc in pid_events[t]:  # return loc
            process_name_metadata(trace_events, pid, t, loc)
            for e in range(8):
                thread_name_metadata(trace_events, t, loc, pid, e, pid_events)
                pid_events[t][loc].append(pid)  # assign unique pid
            pid = pid + 1


# ------------------------------------------------------------------------------
# Script execution start - Open trace file and convert to commands
# ------------------------------------------------------------------------------

opts = parse_args()

DEBUG = opts.debug
if DEBUG:
    print("Debug mode enable\n")

# set colshift based on optional argument
colshift = int(opts.colshift) if opts.colshift else 0

try:
    with open(opts.input, "r") as f:
        # Create array of trace packets
        trace_pkts = f.read().split("\n")
except:
    print("ERROR:",opts.input,"could not be opened. Check for valid trace source file.")
    sys.exit(1)

pid_events = list()

try:
    with open(opts.mlir, "r") as mf:
        lines = mf.read().split("\n")
        pid_events = parse_mlir_trace_events(lines)
except:
    print("ERROR:",opts.mlir,"could not be opened. Check for valid MLIR file.")
    exit(1)

try:
    of = open(opts.output, "w")
except:
    print("ERROR:",opts.mlir,"could not be opened. Check for valid output JSON file.")
    exit(1)

if DEBUG:
    print("DEBUG mode enabled:", file=of)
    print('pkt type 0: core tile', file=of)
    print('pkt type 1: core mem tile', file=of)
    print('pkt type 2: shim tile', file=of)
    print('pkt type 3: mem tile', file=of)
    print("", file=of)
    print("DEBUG: trace_pkts", file=of)
    print(trace_pkts, file=of)
    print("", file=of)

    print("DEBUG: pid events\n", file=of)
    # print(pid_events, file=of)
    for idx, dict_i in enumerate(pid_events):
        print("pkt type",idx,":", file=of)
        for key, value in dict_i.items():
            print(key, value, file=of)
    print("", file=of)

if not check_for_valid_trace(opts.input, trace_pkts, of):
    sys.exit(1)

trace_pkts_sorted = trace_pkts_de_interleave(trace_pkts)

if DEBUG:
    print("DEBUG: trace_pkts_sorted", file=of)
    for idx, dict_i in enumerate(trace_pkts_sorted):
        print("pkt type",idx,":", file=of)
        for key, value in dict_i.items():
            print(key, value, file=of)
    print("", file=of)

byte_streams = convert_to_byte_stream(trace_pkts_sorted)

if DEBUG:
    print("DEBUG: byte stream", file=of)
    for idx, dict_i in enumerate(byte_streams):
        print("pkt type",idx,":", file=of)
        for key, value in dict_i.items():
            print(key, value, file=of)
    print("", file=of)

commands_0 = convert_to_commands(byte_streams, False)

if DEBUG:
    print("DEBUG: commands_0", file=of)
    for idx, dict_i in enumerate(commands_0):
        print("pkt type",idx,":", file=of)
        for key, commands in dict_i.items():
            print(key, file=of)
            for i in commands:
                print("\t",i, file=of)
    print("", file=of)

trace_events = list()

setup_trace_metadata(trace_events, pid_events)

convert_commands_to_json(trace_events, commands_0, pid_events, of)

print(json.dumps(trace_events).replace("\'","\"").replace(", {",",\n{"), file=of)
