# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

# An unnamed ObjectFifo or Worker is named after the variable its statement
# assigns it to, so the symbol, the iron: location name and the buffers and
# locks lowered from it all read like the source. Repeats are numbered, and an
# object that is not assigned to a name falls back to of<N> / worker<N>.

import numpy as np
from aie.helpers.sourceloc import SourceSite
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1

line_type = np.ndarray[(16,), np.dtype[np.int32]]


def walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from walk(child.operation)


def stage():
    out_fifo = ObjectFifo(line_type)
    return out_fifo


def passthrough(a, b):
    a.acquire(1)
    b.acquire(1)
    a.release(1)
    b.release(1)


act_in = ObjectFifo(line_type)
first = stage()
second = stage()
explicit = ObjectFifo(line_type, name="named_by_hand")
fifos = [ObjectFifo(line_type) for _ in range(1)]
forwarded = first.cons().forward()

producer = Worker(passthrough, [act_in.cons(), first.prod()])
workers = [Worker(passthrough, [second.cons(), explicit.prod()])]


def sequence(x, y, z, w, a, b, c, d):
    a.fill(x)
    b.fill(y)
    c.drain(z, wait=True)
    d.drain(w, wait=True)


rt = Runtime(
    sequence,
    [line_type] * 4 + [act_in.prod(), second.prod(), explicit.cons(), forwarded.cons()],
)
module = Program(NPU1Col1(), rt, workers=[producer, *workers]).resolve_program()
ops = list(walk(module.operation))

symbols = {
    op.attributes["sym_name"].value: op.location.name_str
    for op in ops
    if op.name == "aie.objectfifo"
}
for symbol, loc_name in symbols.items():
    assert loc_name == "iron:" + symbol, (symbol, loc_name)
expected = {"act_in", "out_fifo_0", "out_fifo_1", "named_by_hand", "forwarded"}
assert expected <= set(symbols), sorted(symbols)
assert not any(s.startswith("of") for s in symbols if s not in expected), sorted(
    symbols
)

cores = [op.location.name_str for op in ops if op.name == "aie.core"]
# An element of a list is named after the list.
assert cores == ["iron:producer", "iron:workers"], cores
print("PASS: names from source")

# Before Python 3.11 a site has no column. A name read without one is the name
# read with it: of the calls on its line, only those to the function the site
# called count, so a call in its arguments (`act_in.cons()`) does not.
left, right = ObjectFifo(line_type), ObjectFifo(line_type)
without_columns = {}
for name, obj in [
    ("act_in", act_in),
    ("first", first),
    ("forwarded", forwarded),
    ("fifos", fifos[0]),
    ("producer", producer),
    ("workers", workers[0]),
    ("left", left),
    ("right", right),
]:
    site = obj._site
    lined = SourceSite(
        site.filename,
        site.line,
        0,
        returned_to=tuple((f, line, 0) for f, line, _ in site.returned_to),
        callees=site.callees,
    )
    without_columns[name] = lined.assigned_name()
    assert without_columns[name] in (None, site.assigned_name()), (name, lined)
assert without_columns == {
    "act_in": "act_in",
    "first": "out_fifo",
    "forwarded": "forwarded",
    "fifos": "fifos",
    "producer": "producer",
    "workers": "workers",
    # Two ObjectFifo calls on one line, with no column to tell them apart.
    "left": None,
    "right": None,
}, without_columns
print("PASS: names without columns")
