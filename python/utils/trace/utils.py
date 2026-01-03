# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# from CppHeaderParser import CppHeader
import numpy as np
import json
import re
import os
import sys
from .port_events import NUM_TRACE_TYPES


# checks # of bits. Odd number returns a 1. Even returns 0.
def parity(x):
    return x.bit_count() & 1


def extract_tile(data):
    col = (data >> 21) & 0x7F
    row = (data >> 16) & 0x1F
    pkt_type = (data >> 12) & 0x3
    pkt_id = data & 0x1F
    return (col, row, pkt_type, pkt_id)


def pack4bytes(b3, b2, b1, b0):
    w = (b3 & 0xFF) << 24
    w |= (b2 & 0xFF) << 16
    w |= (b1 & 0xFF) << 8
    w |= (b0 & 0xFF) << 0
    return w


def create_ctrl_pkt(
    operation,
    beats,
    addr,
    ctrl_pkt_read_id=28,  # global id used for all ctrl packet reads
    # WARNING: this needs to match the packet id used in packetflow/.py
):
    header = (ctrl_pkt_read_id << 24) | (operation << 22) | (beats << 20) | addr
    header |= (0x1 ^ parity(header)) << 31
    return header


def get_kernel_code(test: dict, solutions_path: str = None) -> str:
    """Fetch the kernel code from the provided solution path, if none provided default
    to canonical solution."""
    if not solutions_path:
        return test["prompt"] + test["canonical_solution"]

    with open(
        os.path.join(solutions_path, f"{test['kernel_name']}.json"), "r"
    ) as sol_file:
        solution = json.load(sol_file)
        if not solution.get("code"):
            print(f"No code available in {solutions_path} for {test['kernel_name']}")
            return None

        srccode = solution["code"]

        # if gpt decides to be too helpful and adds a main()... remove it
        srccode = re.sub(
            r"int\s+main\s*\([^)]*\)\s*{[^{}]*({[^{}]*}[^{}]*)*}",
            "",
            srccode,
            flags=re.DOTALL,
        )

        # cppheaderparser will complain if we don't remove trailing comments
        srccode = srccode.split('// extern "C"')[0]

        return srccode


def extract_buffers(test):
    """Specific helper for the AIEval dataset - parses the test dictionary and returns
    input buffers, output buffers and RTPs as separate lists.
    """
    input_buffers = []
    for x in test["test_vectors"]["inputs"]:
        array, dtype = list(x.values())
        input_buffers.append(np.array(array, dtype=dtype))

    output_buffers = []
    for x in test["test_vectors"]["outputs"]:
        array, dtype = list(x.values())
        output_buffers.append(np.array(array, dtype=dtype))

    rtps = []
    if test["test_vectors"].get("rtps") != None:
        for rtp in test["test_vectors"]["rtps"]:
            array, dtype = rtp.values()
            rtps.append(np.array(array, dtype=dtype))
            # rtp_names.append(list(rtp.keys())[0])

    return input_buffers, output_buffers, rtps


def get_cycles(trace_path):
    """This helper function should only be used to extract cycle counts
    from NPUEval trace files where the expectation is to have exactly 1 of
    each event0 and event1.
    """
    with open(trace_path, "r") as f:
        data = json.load(f)

    event0 = []
    event1 = []
    try:
        for x in data:
            if (x["name"] == "INSTR_EVENT_0") and (x["ph"] == "B"):
                event0.append(x["ts"])
                tmp = x["ts"]
                # print("event0 found at "+str(event0[0]))

            if x["name"] == "INSTR_EVENT_1" and x["ph"] == "B":
                event1.append(x["ts"])
                # print("event1 found at "+str(event1[0]))

        return event1[0] - event0[0]
    except:
        return np.inf


def get_cycles_summary(trace_path):
    """This helper function is  used to extract cycle counts from a trace json
    file and returns an array of cycles between pairs of event0 and event1.
    This always assumes each event0 is followed by an event1 and ignores
    extra event0 and event1's.
    """
    with open(trace_path, "r") as f:
        data = json.load(f)

    try:
        deltas = []
        in_kernel = []
        event0 = []
        for x in data:
            if x["name"] == "process_name":
                deltas.append([x["args"]["name"]])
                in_kernel.append(False)
                event0.append(0)

        for x in data:
            idx = int(x["pid"])
            if (x["name"] == "INSTR_EVENT_0") and (x["ph"] == "B"):
                if in_kernel[idx] == False:
                    event0[idx] = x["ts"]
                    # print("event0 found at "+str(event0))
                    in_kernel[idx] = True

            if x["name"] == "INSTR_EVENT_1" and x["ph"] == "B":
                if in_kernel[idx] == True:
                    # print("event1 found at "+str(x['ts']))
                    deltas[idx].append(x["ts"] - event0[idx])
                    in_kernel[idx] = False

        return deltas
    except Exception as e:
        print("Exception found", e)
        return np.inf


def get_vector_time(trace):
    """This function extracts the total time spent on the vectorized unit
    from an NPUEval AIE trace (this must have exactly 1 event0 and 1 event1
    sandwiching the kernel call).
    """
    with open(trace, "r") as f:
        data = json.load(f)

    start, end = None, None

    # find start and end
    for x in data:
        if (x["name"] == "INSTR_EVENT_0") and (x["ph"] == "B"):
            start = x["ts"]
        if x["name"] == "INSTR_EVENT_1" and x["ph"] == "B":
            end = x["ts"]

    if not start or not end:
        return 0

    total_duration = 0
    stack = []

    for event in data:
        if event["name"] == "INSTR_VECTOR":
            if event["ts"] < start:
                continue

            if event["ts"] > end:
                continue

            if event["ph"] == "B":
                stack.append(event)
            elif event["ph"] == "E" and stack:
                # Get matching begin event
                begin_event = stack.pop()
                # Calculate duration for this pair
                duration = event["ts"] - begin_event["ts"]
                total_duration += duration

    return total_duration / (end - start)


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
        col, row, pkt_type, pkt_id = extract_tile(w)
        hdr["col"] = col
        hdr["row"] = row
        hdr["type"] = pkt_type
        hdr["id"] = pkt_id
    return hdr


def trace_pkts_de_interleave(word_stream):
    trace_pkts_sorted = list()
    for t in range(NUM_TRACE_TYPES):
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
                for tt in range(NUM_TRACE_TYPES):
                    if pkt_hdr["type"] == tt:
                        curr_pkt_type = tt
                        if trace_pkts_sorted[tt].get(curr_loc) == None:
                            trace_pkts_sorted[tt][curr_loc] = list()
                        valid_type_found = True
                if not valid_type_found:
                    sys.exit("Error: Invalid packet type")
            curr_vld = True
        else:
            if curr_vld:
                trace_pkts_sorted[curr_pkt_type][curr_loc].append(word_stream[i])
    return trace_pkts_sorted


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


def convert_to_commands(byte_stream_list, zero=True):
    commands = list()
    for t in range(NUM_TRACE_TYPES):
        commands.append(dict())

    for t in range(NUM_TRACE_TYPES):
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
                        cursor = cursor + 1
                    if (byte_stream[cursor] & 0b11111111) == 0b11111111:
                        com = {"type": "Event_Sync"}
                        commands[t][key].append(com)
                        cursor = cursor + 1
            except IndexError:
                pass

    return commands


def trim_trace_pkts(trace_pkts):
    for i in range(len(trace_pkts)):
        if trace_pkts[i] == "fefefefe" or trace_pkts[i] == "FEFEFEFE":
            if i + 2 < len(trace_pkts):
                if trace_pkts[i + 1] == "00000000" and trace_pkts[i + 2] == "00000000":
                    return trace_pkts[0 : i + 1]
    return trace_pkts
