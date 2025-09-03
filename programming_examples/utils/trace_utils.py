# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# from CppHeaderParser import CppHeader
import numpy as np
import subprocess
import json
import re
import os


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


def trace_to_json(trace_file: str, mlir_file: str, output_name: str = "trace.json"):
    """Subprocesses wrapper over parse_trace.py utility.
    Parameters
    ----------
    trace_file : str
        The .txt trace file of 32-byte codes.
    mlir_file : str
        Path to the corresponding MLIR file for the design being traced.
    output_name : str, optional
        Path to output json file. You can analyze it using tools like https://ui.perfetto.dev
    """
    command = [
        os.environ["MLIR_AIE_INSTALL_DIR"]
        + "/../../programming_examples/utils/parse_trace.py",
        "--input",
        trace_file,
        "--mlir",
        mlir_file,
    ]

    try:
        result = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
        with open(output_name, "w") as f:
            f.write(result)
        print(f"Trace written to {output_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Trace failed\n{e.output}")
        return e.output


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
