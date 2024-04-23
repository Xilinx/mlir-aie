#!/usr/bin/python3
import json
import argparse
import sys
import re

# Number of different trace types, currently 4
# core:    pkt type 0
# mem:     pkt type 1
# intfc:   pkt type 2
# memtile: pkt type 3
NumTraceTypes = 4
NUM_EVENTS = 8  # number of events we can view per trace

DEBUG = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Trace file", required=True)
    parser.add_argument("--mlir", help="mlir source file", required=True)
    parser.add_argument(
        "--colshift", help="column shift adjustment to source mlir", required=False
    )
    # TODO tracelabels removed since we can have multiple sets of labels for each pkt_type & loc combination
    # parser.add_argument('--tracelabels',
    #         nargs='+',
    #         help='Labels for traces', required=False)
    return parser.parse_args(sys.argv[1:])


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


# toks_list:   list (idx = types of traces, currently 4, value = stream_dict)
# stream_dict: dict (key = row,col, value = list of word streams)
def core_trace_and_mem_trace_de_interleave(word_stream):
    toks_list = list()
    for t in range(NumTraceTypes):
        toks_list.append(dict())

    # core_streams = dict()   # pkt type 0
    # mem_stream = dict()     # pkt type 1
    # intfc_stream = dict()   # pkt type 2
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
                        if toks_list[tt].get(curr_loc) == None:
                            toks_list[tt][curr_loc] = list()
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
                toks_list[curr_pkt_type][curr_loc].append(
                    word_stream[i]
                )  # TODO assuem curr_pkt_type is valid
                # for tt in range(NumTraceTypes):
                #     if curr_pkt_type == tt:
                #         toks_list[tt][curr_loc].append(word_stream[i])
    return toks_list


# toks_list is a list of toks dictionaries where each dictionary is a type (core, mem, intfc, memtile)
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


# multiples is a list of events that are being activated
def deactivate(
    multiples, active_events, timer, cycles, pid, trace_type, loc, pid_events
):
    for k in active_events.keys():  # an active event
        if cycles > 0 or (cycles == 0 and not k in multiples):
            # if not k in multiples: # active event it not in multiples list
            if active_events[k] > 0:
                # trace_event = {'name':"Event"+str(k)}
                # trace_event = {'name':events_to_name[k]}
                # trace_event = {'name':lookupEventNameInStr(str(k), pid, pid_events)}
                # trace_event = {'name':lookup_event_name_by_type(trace_type, str(k), pid_events)} # TODO remove
                trace_event = {
                    "name": lookup_event_name_by_type(
                        trace_type, pid_events[trace_type][loc][k]
                    )
                }  # TODO remove
                # trace_event['args']['name'] = lookup_event_name_by_type(trace_type, pid_events[trace_type][loc][k])
                #                trace_event['ts'] = active_events[k]
                trace_event["ts"] = timer
                trace_event["ph"] = "E"
                # trace_event['pid'] = 0
                trace_event["pid"] = pid
                trace_event["tid"] = k
                trace_event["args"] = {}
                # trace_event['keys'] = multiples
                # trace_event['active_events'] = active_events.keys()
                # trace_event['opcode'] = "Deactivate" + str(multiples)

                trace_events.append(trace_event)
            active_events[k] = 0


# def update(t):
#     for k in active_events.keys():
#         if active_events[k]:
#             active_events[k] = t


# def convert_commands_to_json(trace_events, commands, pid, pid_events):
# TODO iterate over commands which is ...
#
# commands:  list (idx = trace type, value = byte_stream_dict)
# byte_stream_dict: dict (key = row,col, value = list of commands)
def convert_commands_to_json(trace_events, commands, pid_events):
    # for bsd in commands: # byte_stream_dict for each trace type. TODO how to get index of bsd?
    for tt in range(
        len(commands)
    ):  # byte_stream_dict for each trace type. TODO how to get index of bsd?
        byte_stream_dict = commands[tt]

        for loc, command in byte_stream_dict.items():  # row,col with list of commands
            timer = 0  # TODO Some way to set this or sync this between trace types and row,col
            # timer on each execution is the time for the last execution
            # so we by default will increment it by 1 for each event
            pid = pid_events[tt][loc][NUM_EVENTS]

            active_events = dict()
            for i in range(16):  # TODO we only have 8 events at a time though right?
                active_events[i] = 0

            for c in command:
                # for c in flat_commands:
                # print(c)
                t = c["type"]
                # if 'Start' in t:
                # timer = c['timer_value'] # TODO Turn off timer for now to test sync among tiles/type
                # elif 'Single' in t:
                if "Single" in t:
                    event = c["event"]
                    cycles = int(c["cycles"])

                    # timer = timer + 1 + int(c['cycles'])   # Timer at top
                    timer = timer + 1

                    # if cycles > 0, deactivate all events
                    # if cycles == 0, deactivate all events except this one

                    # TODO NO? deactivate all active_events that is not this event
                    multiple_list = list()
                    # for k in c.keys():
                    #     if "event" in k:
                    #         multiple_list.append(c[k]) # TODO overkill since there should only be one?
                    multiple_list.append(c["event"])
                    # for k in active_events.keys():
                    #     if cycles > 0 or (cycles == 0 and k != event):
                    #         multiple_list.append(k)
                    deactivate(
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

                    # If its already started, don't start it again ...
                    try:
                        if active_events[event] == 0:
                            # trace_event = {'name':events_to_name[event]}
                            # trace_event = {'name':lookupEventNameInStr(str(event), pid, pid_events)}
                            # trace_event = {'name':lookupEventNameInStr(str(event), pid, pid_events)} # TODO
                            trace_event = {
                                "name": lookup_event_name_by_type(
                                    tt, pid_events[tt][loc][event]
                                )
                            }
                            trace_event["ts"] = timer
                            trace_event["ph"] = "B"
                            # trace_event['pid'] = 0
                            trace_event["pid"] = pid
                            trace_event["tid"] = event
                            trace_event["args"] = {}
                            # trace_event['opcode'] = "Single"
                            trace_events.append(trace_event)
                            #                active_events[event] = timer  + 1
                            active_events[event] = 1
                    except KeyError:
                        pass
                    # timer = timer + 1 + int(c['cycles'])
                elif "Multiple" in t:
                    cycles = int(c["cycles"])

                    # timer = timer + 1 + int(c['cycles'])
                    timer = timer + 1

                    # if cycles > 0, deactivate all events
                    # if cycles == 0, deactivate all events except this one

                    # TODO NO? deactivate all active_events that is not this event
                    multiple_list = list()
                    for k in c.keys():
                        if "event" in k:
                            multiple_list.append(c[k])
                    # for k in active_events.keys():
                    #     if cycles > 0 or (cycles == 0 and k != event):
                    #         multiple_list.append(k)
                    deactivate(
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
                        if not "event" in k:
                            continue
                        # If its already started, don't start it again ...
                        try:
                            event = c[k]
                            if active_events[event] == 0:
                                # trace_event = {'name':events_to_name[event]}
                                # trace_event = {'name':lookupEventNameInStr(str(event), pid, pid_events)} # TODO
                                trace_event = {
                                    "name": lookup_event_name_by_type(
                                        tt, pid_events[tt][loc][event]
                                    )
                                }
                                trace_event["ts"] = timer
                                trace_event["ph"] = "B"
                                # trace_event['pid'] = 0
                                trace_event["pid"] = pid
                                trace_event["tid"] = event
                                trace_event["args"] = {}
                                # trace_event['opcode'] = "Multiple" + str(list(c.keys()))
                                trace_events.append(trace_event)
                                #                    active_events[event] = timer  + 1
                                active_events[event] = 1
                        except KeyError:
                            pass
                    # timer = timer + 1 + int(c['cycles'])

                elif "Repeat" in t:
                    timer = timer + int(c["repeats"])
            #        update(timer)


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
        trace_event["args"]["name"] = "intfc_trace for tile" + str(loc)
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
                    col = 1 if col == 0 else col
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

            # TODO intfc and memtile event 0, 1 needs to also be defined

    # print("Found labels:\n")
    # for j in pid_events:
    #     print("row:",j['row'],", col: ",j['col'])
    #     print("0: ", j[0], "1: ", j[1], "2: ", j[2], "3: ", j[3])
    #     print("4: ", j[4], "5: ", j[5], "6: ", j[6], "7: ", j[7])
    return pid_events


def lookup_event_name_by_type(trace_type, code):
    # def lookup_event_name_by_type(trace_type, loc, event, pid_events):
    event = ""
    # Core traces
    # code = pid_events[trace_type][loc][event]
    if trace_type == 0:
        if code == 0x1:
            event = "True"
        elif code == 24:  # 0x18:
            event = "StreamStall"
        elif code == 26:  # 0x1A:
            event = "LockStall"
        elif code == 32:  # 0x20, all events 33-45
            event = "CoreProgramFlow"
        elif code == 33:  # 0x21:
            event = "Event0"
        elif code == 34:  # 0x22:
            event = "Event1"
        elif code == 37:  # 0x25:
            event = "VectorInstr"
        elif code == 38:  # 0x26:
            event = "InstrLoad"
        elif code == 39:  # 0x27:
            event = "InstrStore"
        elif code == 44:  # 0x2C:
            event = "LockAcquireInstr"
        elif code == 45:  # 0x2D:
            event = "LockReleaseInstr"
        elif code == 75:  # 0x4B:
            event = "PortRunning0"
        elif code == 79:  # 0x4F:
            event = "PortRunning1"
        else:
            event = "Unknown"
    # Mem traces
    elif trace_type == 1:
        # TODO Need to define these
        if code == 21:  # x15
            event = "DMA s2mm 0 start bd"
        elif code == 22:  # x16
            event = "DMA s2mm 1 start bd"
        elif code == 23:  # x17
            event = "DMA mm2s 0 start bd"
        elif code == 24:  # x18
            event = "DMA mm2s 1 start bd"
        elif code == 25:  # x19
            event = "DMA s2mm 0 finish bd"
        elif code == 26:  # x1a
            event = "DMA s2mm 1 finish bd"
        elif code == 27:  # x1b
            event = "DMA mm2s 0 finish bd"
        elif code == 28:  # x1c
            event = "DMA mm2s 1 finish bd"
        elif code == 29:  # x1d
            event = "DMA s2mm 0 idle"
        elif code == 30:  # x1e
            event = "DMA s2mm 1 idle"
        elif code == 31:  # x1f
            event = "DMA mm2s 0 idle"
        elif code == 32:  # x20
            event = "DMA mm2s 1 idle"
        elif code == 33:  # x21
            event = "DMA s2mm 0 stalled lock acquire"
        elif code == 34:  # x22
            event = "DMA s2mm 1 stalled lock acquire"
        else:
            event = "Unknown"
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

# set colshift based on optional argument
colshift = int(opts.colshift) if opts.colshift else 0

with open(opts.filename, "r") as f:
    toks = f.read().split("\n")

    if DEBUG:
        print("\nDEBUG: toks")
        print(toks)
        print("\n\n")

    # De-interleave core and memory trace
    # [core_toks, mem_toks] = core_trace_and_mem_trace_de_interleave(toks)

    # TODO Change this to list of toks instead of a fixed set?
    toks_list = core_trace_and_mem_trace_de_interleave(toks)

    if DEBUG:
        print("\nDEBUG: stream")
        print(toks_list)
        print("\n\n")

    bs_0 = convert_to_byte_stream(toks_list)
    # core_bs_0    = convert_to_byte_stream(core_toks)
    # mem_bs_0     = convert_to_byte_stream(mem_toks)
    # intfc_bs_0   = convert_to_byte_stream(intfc_toks)
    # memtile_bs_0 = convert_to_byte_stream(memtile_toks)

    if DEBUG:
        print("\nDEBUG: byte stream")
        print(bs_0)
        print("\n\n")

    # core_bs_1 = convert_to_byte_stream(core_toks[1])
    # mem_bs_0 = convert_to_byte_stream(mem_toks[0])
    # mem_bs_1 = convert_to_byte_stream(mem_toks[1])

    commands_0 = convert_to_commands(bs_0, False)
    # core_commands_0    = convert_to_commands(core_bs_0, False)
    # mem_commands_0     = convert_to_commands(mem_bs_0, False)
    # intfc_commands_0   = convert_to_commands(intfc_bs_0, False)
    # memtile_commands_0 = convert_to_commands(memtile_bs_0, False)

    # core_commands_1 = convert_to_commands(core_bs_1, False)
    # mem_commands_0 = convert_to_commands(mem_bs_0, False)
    # mem_commands_1 = convert_to_commands(mem_bs_1, False)

if DEBUG:
    print("\nDEBUG: commands_0")
    print(commands_0)
    print("\n\n")

# pid_events track the labels associated with pkt_type/ trace and row,col
#
# pid_events: list(idx=pkt_type, value=labels_dict)
# label_dict: dict(key=row,col, value=labels list)
# labels_list: list(idx=label idx, value=label code), idx=len is pid
#
pid_events = list()

if opts.mlir:
    with open(opts.mlir, "r") as mf:
        lines = mf.read().split("\n")
        pid_events = parse_mlir_trace_events(lines)
# TODO need an else if mlir is not defined, some default or make it required?
# else:


# flat_commands = flatten_repeat_command(commands)

# events_to_name= list()
# core_events = dict()
# events_to_name.append(core_events)

# if opts.tracelabels:
#     for i in range(8):
#         events_to_name[0][i] = opts.tracelabels[i]
# else:
#     events_to_name[0][0] = "Instr_Vector"
#     events_to_name[0][1] = "Tile Start"
#     events_to_name[0][2] = "Tile End"
#     events_to_name[0][3] = "Read A"
#     events_to_name[0][4] = "Read B"
#     events_to_name[0][5] = "Write C"
#     events_to_name[0][6] = "Lock Stall"
#     events_to_name[0][7] = "Lock Acquire"

# print (json.dumps(commands))

trace_events = list()
# Let's name the threads
# active_events = dict()

# for i in range(8):
# for i in range(16):
#     active_events[i]  = 0
# name_dict = dict()
# # {"name": "thread_name", "ph":"M","pid":0, "tid":0,"args":{"name":"Phil"}}
# name_dict["name"] = "thread_name"
# name_dict["ph"]   = "M"           # Metadate
# name_dict["pid"]  = 0             # We only have one process, but we may want one per core
# name_dict["tid"]  = i             # I have baked in 8 here, we need this to be informed by how many events are supported
# name_dict["args"] = {"name":events_to_name[i]}
# trace_events.append(name_dict)

# timer = 0

# {'type': 'Start', 'timer_value': 0}
# {'type': 'Single2', 'event': 0, 'cycles': 1161}
# {'type': 'Single0', 'event': 0, 'cycles': 0}
# {'type': 'Repeat1', 'repeats': 34}
# {'type': 'Single0', 'event': 0, 'cycles': 2}
# {'type': 'Single0', 'event': 0, 'cycles': 0}
# {'type': 'Repeat1', 'repeats': 24}
# {'type': 'Multiple0', 'cycles': 0, 'event0': 0, 'event2': 2}
# {'type': 'Single0', 'event': 0, 'cycles': 0}
# {'type': 'Repeat0', 'repeats': 4}
# {'type': 'Single1', 'event': 3, 'cycles': 117}
# {'type': 'Single1', 'event': 0, 'cycles': 21}
# {'type': 'Single0', 'event': 0, 'cycles': 0}
# {'type': 'Repeat1', 'repeats': 21}
# {'type': 'Multiple0', 'cycles': 0, 'event0': 0, 'event1': 1}
# {'type': 'Repeat0', 'repeats': 8}
# {'type': 'Single0', 'event': 1, 'cycles': 0}

# ------------------------------------------------------------------------------
# Convert commands to json
# ------------------------------------------------------------------------------

# for pid in range(4):
# for pid in range(1):
#     process_name_metadata(trace_events, pid)
#     for tid in range(8):
#         thread_name_metadata(trace_events, pid, tid, pid_events)

# pid will be a product of number of trace types and trace tiles

setup_trace_metadata(trace_events, pid_events)

if DEBUG:
    print("\nDEBUG: pid events\n")
    print(pid_events)
    print("\n\n")

# TODO Loop over cores/mem to call convert_commands_to_json
convert_commands_to_json(trace_events, commands_0, pid_events)

# convert_commands_to_json(trace_events, mem_commands_0, 1, pid_events)
# convert_commands_to_json(trace_events, core_commands_1, 2, pid_events)
# convert_commands_to_json(trace_events, mem_commands_1, 3, pid_events)

# for t in trace_events:
#     print(t)
print(json.dumps(trace_events))
