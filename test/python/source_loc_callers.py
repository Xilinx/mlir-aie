# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s
# RUN: env AIE_SOURCE_LOC_DEPTH=1 %PYTHON %s --depth 1

# An IRON object records the calls that led to its declaring statement, so two
# objects one helper builds are told apart by the call that built each. With a
# depth of 1, only the statement is recorded, as a plain file location.

import os
import sys

import numpy as np
from aie.ir import CallSiteLoc, FileLineColLoc, NameLoc
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1

line_type = np.ndarray[(16,), np.dtype[np.int32]]
THIS_FILE = os.path.abspath(__file__)
SOURCE = open(THIS_FILE).read().splitlines()
DEPTH = int(sys.argv[sys.argv.index("--depth") + 1]) if "--depth" in sys.argv else 4


def stage(name):
    return ObjectFifo(line_type, name=name)


def build():
    first = stage("first")
    second = stage("second")

    def core_fn(a, b):
        a.acquire(1)
        a.release(1)
        b.acquire(1)
        b.release(1)

    worker = Worker(core_fn, [first.cons(), second.cons()])

    def sequence(x, y, a, b):
        a.fill(x)
        b.fill(y)

    rt = Runtime(sequence, [line_type, line_type, first.prod(), second.prod()])
    return Program(NPU1Col1(), rt, workers=[worker]).resolve_program()


def frames(loc):
    """Return the file locations of `loc`, innermost first."""
    if isinstance(loc, NameLoc):
        return frames(loc.child_loc)
    if isinstance(loc, CallSiteLoc):
        return frames(loc.callee) + frames(loc.caller)
    assert isinstance(loc, FileLineColLoc), loc
    return [(loc.filename, loc.start_line)]


def cited(frame):
    filename, line = frame
    assert filename == THIS_FILE, frame
    return SOURCE[line - 1]


module = build()
fifos = {
    op.attributes["sym_name"].value: op.location
    for op in module.body.operations[0].regions[0].blocks[0].operations
    if op.operation.name == "aie.objectfifo"
}
for name, call in (
    ("first", 'first = stage("first")'),
    ("second", 'second = stage("second")'),
):
    stack = frames(fifos[name])
    assert "ObjectFifo(line_type, name=name)" in cited(stack[0]), stack
    if DEPTH == 1:
        assert len(stack) == 1, stack
        assert isinstance(fifos[name].child_loc, FileLineColLoc), fifos[name]
    else:
        assert 1 < len(stack) <= DEPTH, stack
        assert call in cited(stack[1]), stack

assert (frames(fifos["first"]) != frames(fifos["second"])) == (DEPTH > 1)
print(f"PASS: depth {DEPTH}")
