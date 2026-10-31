# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s
# Without column positions, a site's call is picked by the function it called.
# RUN: env PYTHONNODEBUGRANGES=1 %PYTHON %s

# An unnamed ObjectFifo or Worker is named after the variable its statement
# assigns it to, so the symbol, the iron: location name and the buffers and
# locks lowered from it all read like the source. Repeats are numbered, and an
# object that is not assigned to a name falls back to of<N> / worker<N>.

import numpy as np
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
