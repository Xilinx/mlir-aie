# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

"""End-to-end check that a realistic IRON design is attributed to user code.

The unit-level fixture in `location_preservation.py` drives the dialect
builders directly, so the user frame it recovers is its own -- the one case
that cannot fail. IRON declares objects in one place and emits their ops much
later inside `Program.resolve_program`, and that is where attribution breaks.

Rejecting `loc(unknown)` only proves *a* location was attached. An op wrongly
blamed on an IRON internal, or on a neighbouring declaration, passes that check
happily. So each op is also matched back to the source line that should have
produced it: this file is read back and the cited line must contain the
construct named below. Walking the operation tree rather than the printed
assembly keeps that honest -- a region op prints its location on its closing
brace, which is easy to mis-associate when scraping text.
"""

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


# (op name, sym_name or None, substring that must appear on the cited line).
# Naming the construct rather than a line number keeps this from silently
# passing if the design above is edited, and turns a misattribution into a
# failure rather than a shrug.
EXPECTED = [
    ("aie.objectfifo", "in", 'ObjectFifo(line_type, name="in")'),
    ("aie.objectfifo", "out", 'ObjectFifo(line_type, name="out")'),
    ("aie.buffer", "rtp", 'Buffer(np.ndarray[(4,), np.dtype[np.int32]], name="rtp")'),
    ("func.func", None, "passthrough = kernels.passthrough("),
    ("aie.core", None, "worker = Worker(core_fn"),
    ("func.call", None, "kernel(elem_in, elem_out, LINE_SIZE)"),
    ("aie.objectfifo.acquire", None, "def core_fn("),
    ("aie.objectfifo.release", None, "def core_fn("),
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


def sym_name_of(op):
    try:
        return op.attributes["sym_name"].value
    except (KeyError, AttributeError, IndexError):
        return None


def main():
    ops = list(walk(build().operation))

    # 1. Nothing may fall back to an unknown location.
    unknown = [op.name for op in ops if "unknown" in str(op.location)]
    assert not unknown, f"ops with unknown location: {sorted(set(unknown))}"

    # 2. Every op must be attributed to THIS file. An op pointing into
    #    aie/iron/*.py is the misattribution that rejecting `unknown` misses.
    for op in ops:
        match = FILE_LOC.search(str(op.location))
        assert match, f"{op.name}: no file location in {op.location}"
        cited = os.path.abspath(match.group(1))
        assert (
            cited == THIS_FILE
        ), f"{op.name} attributed outside the design: {cited}:{match.group(2)}"

    # 3. Each op must cite the line that actually declared it.
    for op_name, sym, expected_source in EXPECTED:
        matches = [
            op
            for op in ops
            if op.name == op_name and (sym is None or sym_name_of(op) == sym)
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

    # 4. Distinct declarations must not collapse onto a single line. Two
    #    ObjectFifos declared one line apart are the easy case to get wrong.
    fifo_lines = {
        int(FILE_LOC.search(str(op.location)).group(2))
        for op in ops
        if op.name == "aie.objectfifo"
    }
    assert len(fifo_lines) == 2, f"ObjectFifos collapsed onto lines {fifo_lines}"

    print(f"PASS: {len(ops)} ops attributed, {len(EXPECTED)} constructs verified")


main()
