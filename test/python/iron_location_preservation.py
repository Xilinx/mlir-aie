# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

import os
import re

import numpy as np
from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker, kernels
from aie.iron.device import NPU1Col1

LINE_SIZE = 256
line_type = np.ndarray[(LINE_SIZE,), np.dtype[np.uint8]]
vector_type = np.ndarray[(1024,), np.dtype[np.uint8]]

THIS_FILE = os.path.abspath(__file__)
SOURCE = open(THIS_FILE).read().splitlines()


def build():
    of_in = ObjectFifo(line_type, name="in")
    of_out = ObjectFifo(line_type, name="out")
    rtp = Buffer(np.ndarray[(4,), np.dtype[np.int32]], name="rtp")
    passthrough = kernels.passthrough(tile_size=LINE_SIZE, dtype=np.uint8)

    def core_fn(a, b, kernel, scratch):
        elem_out = b.acquire(1)
        elem_in = a.acquire(1)
        kernel(elem_in, elem_out, LINE_SIZE)
        a.release(1)
        b.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod(), passthrough, rtp])

    def sequence(a_in, b_out, in_handle, out_handle):
        in_handle.fill(a_in)
        out_handle.drain(b_out, wait=True)

    rt = Runtime(sequence, [vector_type, vector_type, of_in.prod(), of_out.cons()])
    return Program(NPU1Col1(), rt, workers=[worker]).resolve_program()


# (op name, sym_name or None, text that must appear on the cited line)
EXPECTED = [
    ("aie.objectfifo", "in", 'ObjectFifo(line_type, name="in")'),
    ("aie.objectfifo", "out", 'ObjectFifo(line_type, name="out")'),
    ("aie.buffer", "rtp", 'Buffer(np.ndarray[(4,), np.dtype[np.int32]], name="rtp")'),
    ("func.func", None, "passthrough = kernels.passthrough("),
    ("aie.core", None, "worker = Worker(core_fn"),
    ("func.call", None, "kernel(elem_in, elem_out, LINE_SIZE)"),
    ("aie.objectfifo.acquire", None, "acquire(1)"),
    ("aie.objectfifo.release", None, "release(1)"),
    ("aie.runtime_sequence", None, "rt = Runtime(sequence"),
    ("aiex.dma_start_task", None, "in_handle.fill(a_in)"),
    ("aiex.dma_await_task", None, "out_handle.drain(b_out, wait=True)"),
    ("aie.device", None, "Program(NPU1Col1(), rt, workers=[worker])"),
]

FILE_LOC = re.compile(r'"([^"]+)":(\d+):(\d+)')


def walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from walk(child.operation)


def main():
    ops = list(walk(build().operation))

    unknown = [op.name for op in ops if "unknown" in str(op.location)]
    assert not unknown, f"ops with unknown location: {sorted(set(unknown))}"

    for op in ops:
        match = FILE_LOC.search(str(op.location))
        assert match, f"{op.name}: no file location in {op.location}"
        cited = os.path.abspath(match.group(1))
        assert (
            cited == THIS_FILE
        ), f"{op.name} attributed outside the design: {cited}:{match.group(2)}"

    for op_name, sym, expected_source in EXPECTED:
        matches = [
            op
            for op in ops
            if op.name == op_name
            and (sym is None or op.attributes["sym_name"].value == sym)
        ]
        label = op_name + (f" @{sym}" if sym else "")
        assert matches, f"no op matching {label} was emitted"
        match = FILE_LOC.search(str(matches[0].location))
        assert match is not None
        line_no = int(match.group(2))
        cited_text = SOURCE[line_no - 1]
        assert expected_source in cited_text, (
            f"{label} cites line {line_no}\n"
            f"  that line is:            {cited_text.strip()}\n"
            f"  expected it to contain:  {expected_source}"
        )

    fifo_lines = {
        int(FILE_LOC.search(str(op.location)).group(2))
        for op in ops
        if op.name == "aie.objectfifo"
    }
    assert len(fifo_lines) == 2, f"ObjectFifos collapsed onto lines {fifo_lines}"

    acquire_lines = {
        int(FILE_LOC.search(str(op.location)).group(2))
        for op in ops
        if op.name == "aie.objectfifo.acquire"
    }
    assert (
        len(acquire_lines) == 2
    ), f"acquires collapsed onto {acquire_lines}; expected one line each"
    for line_no in acquire_lines:
        assert (
            "acquire(1)" in SOURCE[line_no - 1]
        ), f"acquire attributed to {line_no}: {SOURCE[line_no - 1].strip()}"

    print(f"PASS: {len(ops)} ops attributed, {len(EXPECTED)} constructs verified")


main()
