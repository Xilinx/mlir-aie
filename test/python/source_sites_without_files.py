# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

# A design run from a string or an IPython cell is located in that code, not at
# the statement that ran it.

from aie.ir import CallSiteLoc

DESIGN = """\
import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1

line_type = np.ndarray[(16,), np.dtype[np.int32]]


def passthrough(a, b):
    a.acquire(1)
    b.acquire(1)
    a.release(1)
    b.release(1)


act_in = ObjectFifo(line_type)
act_out = ObjectFifo(line_type)
worker = Worker(passthrough, [act_in.cons(), act_out.prod()])


def sequence(x, y, a, b):
    a.fill(x)
    b.drain(y, wait=True)


rt = Runtime(sequence, [line_type] * 2 + [act_in.prod(), act_out.cons()])
module = Program(NPU1Col1(), rt, workers=[worker]).resolve_program()
"""
ACT_IN_LINE = DESIGN.splitlines().index("act_in = ObjectFifo(line_type)") + 1


def fifo_sites(module):
    sites = {}
    for op in module.body.operations[0].regions[0].blocks[0].operations:
        if op.operation.name != "aie.objectfifo":
            continue
        loc = op.location.child_loc
        loc = loc.callee if isinstance(loc, CallSiteLoc) else loc
        sites[op.attributes["sym_name"].value] = (loc.filename, loc.start_line)
    return sites


scope = {}
exec(compile(DESIGN, "<string>", "exec"), scope)
sites = fifo_sites(scope["module"])
# The string's source can't be read back, so the fifos keep their kind's name.
assert sorted(sites) == ["of0", "of1"], sites
assert sites["of0"] == ("<string>", ACT_IN_LINE), sites
print("PASS: exec of a string")
