# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

# The rules for naming an unnamed IRON object after its source: only the
# statement that constructs it counts, and each assignment form gives one
# deterministic name.

import sys

import numpy as np
from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1

line_type = np.ndarray[(16,), np.dtype[np.int32]]


def _probe():
    pass


# Python 3.10, or 3.11+ under -X no_debug_ranges, records no columns, so two
# calls to the same constructor on one line cannot be told apart.
COLUMNS = sys.version_info >= (3, 11) and all(
    col is not None for _, _, col, _ in _probe.__code__.co_positions()
)


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
# Subscript target: the container, numbered from 0 on repeats.
slots = [None, None]
slots[0] = ObjectFifo(line_type)
slots[1] = ObjectFifo(line_type)
# Elements of a list display: named after the list, numbered from 0.
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
    "held",
    "slots_0",
    "slots_1",
    "reused",
    "listed_0",
    "listed_1",
    "made_0",
    "made_1",
    "chosen",
}
if COLUMNS:
    expected |= {"left", "right"}
assert expected <= set(symbols), sorted(symbols)
fallbacks = set(symbols) - expected
assert len(fallbacks) == (1 if COLUMNS else 3), sorted(symbols)
assert all(f.startswith("of") for f in fallbacks), fallbacks
buffers = [op.attributes["sym_name"].value for op in ops if op.name == "aie.buffer"]
assert "scale" in buffers, buffers
print("PASS: source name rules")

# A chain of calls names only the object its outermost call makes: the fifo
# between two forwards is no variable's.
src = ObjectFifo(line_type)
mid = src.cons().forward().cons().forward()


def move(a, b, fill, drain):
    fill.fill(a)
    drain.drain(b, wait=True)


chain_rt = Runtime(move, [line_type, line_type, src.prod(), mid.cons()])
chain = Program(NPU1Col1(), chain_rt).resolve_program()
chain_symbols = sorted(
    op.attributes["sym_name"].value
    for op in walk(chain.operation)
    if op.name == "aie.objectfifo"
)
if COLUMNS:
    (middle,) = set(chain_symbols) - {"src", "mid"}
    assert "src" in chain_symbols and "mid" in chain_symbols, chain_symbols
    assert middle.startswith("of"), chain_symbols
else:
    # Both forwards are calls to `forward` on the line.
    assert chain_symbols == ["mid_0", "mid_1", "src"], chain_symbols
print("PASS: a chain names only its outermost object")
