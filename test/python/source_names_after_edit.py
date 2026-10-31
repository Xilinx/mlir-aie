# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

# A design edited and rerun in the same process is named after its new source,
# as under IPython's %run or a reload.

import runpy
import tempfile
from pathlib import Path

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

{padding}
{first} = ObjectFifo(line_type)
{second} = ObjectFifo(line_type)
worker = Worker(passthrough, [{first}.cons(), {second}.prod()])


def sequence(x, y, a, b):
    a.fill(x)
    b.drain(y, wait=True)


rt = Runtime(sequence, [line_type] * 2 + [{first}.prod(), {second}.cons()])
module = Program(NPU1Col1(), rt, workers=[worker]).resolve_program()
"""


def fifo_names(path):
    module = runpy.run_path(str(path))["module"]
    ops = module.body.operations[0].regions[0].blocks[0].operations
    return sorted(
        op.attributes["sym_name"].value
        for op in ops
        if op.operation.name == "aie.objectfifo"
    )


with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "design.py"
    path.write_text(DESIGN.format(padding="", first="act_in", second="act_out"))
    assert fifo_names(path) == ["act_in", "act_out"]
    # The edit moves every statement down a line and renames both fifos.
    path.write_text(DESIGN.format(padding="\n", first="pixels", second="edges"))
    names = fifo_names(path)
    assert names == ["edges", "pixels"], names
print("PASS: names follow an edited design")
