# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

# The rules for naming an unnamed IRON object after its source: only the
# statement that constructs it counts, and each assignment form gives one
# deterministic name.

import numpy as np
from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1

line_type = np.ndarray[(16,), np.dtype[np.int32]]


class Holder:
    def __init__(self):
        self.held = ObjectFifo(line_type)


def walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from walk(child.operation)


# An alias made later does not rename it.
my_fifo = ObjectFifo(line_type)
other_fifo = my_fifo
# Chained assignment: the first target.
chained = also_chained = ObjectFifo(line_type)
# Tuple unpacking: element by element.
left, right = ObjectFifo(line_type), ObjectFifo(line_type)
# Attribute target: the attribute.
holder = Holder()
# Subscript target: the container, numbered on repeats.
slots = [None, None]
slots[0] = ObjectFifo(line_type)
slots[1] = ObjectFifo(line_type)
# Elements of a list display: named after the list, numbered.
listed = [ObjectFifo(line_type), *([ObjectFifo(line_type)] if True else [])]
# Comprehension elements and either branch of a conditional likewise.
made = [ObjectFifo(line_type) for _ in range(2)]
chosen = ObjectFifo(line_type) if True else None
# Built inside an argument to a call: not named after the outer statement.
passed = list((ObjectFifo(line_type),))
# Reassignment later does not rename it either.
reused = ObjectFifo(line_type)
reused_first = reused
reused = None
# A Buffer is named the same way.
scale = Buffer(np.ndarray[(4,), np.dtype[np.int32]], use_write_rtp=True)

fifos = [
    other_fifo,
    chained,
    left,
    right,
    holder.held,
    slots[0],
    slots[1],
    listed[0],
    listed[1],
    made[0],
    made[1],
    chosen,
    passed[0],
    reused_first,
]


def core_fn(*args):
    for handle in args[:-1]:
        handle.acquire(1)
        handle.release(1)


worker = Worker(core_fn, [f.cons() for f in fifos] + [scale])


def sequence(*args):
    for x, handle in zip(args[: len(fifos)], args[len(fifos) :]):
        handle.fill(x)


rt = Runtime(sequence, [line_type] * len(fifos) + [f.prod() for f in fifos])
module = Program(NPU1Col1(), rt, workers=[worker]).resolve_program()
ops = list(walk(module.operation))

symbols = [op.attributes["sym_name"].value for op in ops if op.name == "aie.objectfifo"]
expected = {
    "my_fifo",
    "chained",
    "left",
    "right",
    "held",
    "slots",
    "slots_1",
    "reused",
    "listed",
    "listed_1",
    "made",
    "made_1",
    "chosen",
}
assert expected <= set(symbols), sorted(symbols)
(fallback,) = set(symbols) - expected
assert fallback.startswith("of"), fallback
buffers = [op.attributes["sym_name"].value for op in ops if op.name == "aie.buffer"]
assert "scale" in buffers, buffers
print("PASS: source name rules")
