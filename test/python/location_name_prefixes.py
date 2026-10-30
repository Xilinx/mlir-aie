# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

# A NameLoc's name says what it names: "iron:" an IRON object (a fifo, a
# Worker's core), "py:" the Python function a traced statement ran in. Workers
# without a name are named worker0, worker1, ... in their configuration.

import numpy as np
from aie.ir import CallSiteLoc, FusedLoc, NameLoc
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1

line_type = np.ndarray[(16,), np.dtype[np.int32]]


def walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from walk(child.operation)


def names(loc):
    if isinstance(loc, NameLoc):
        return [loc.name_str] + names(loc.child_loc)
    if isinstance(loc, CallSiteLoc):
        return names(loc.callee) + names(loc.caller)
    if isinstance(loc, FusedLoc):
        return [n for child in loc.locations for n in names(child)]
    return []


def core_fn(fifo):
    fifo.acquire(1)
    fifo.release(1)


of_in = ObjectFifo(line_type, name="of_in")
named = Worker(core_fn, [of_in.cons()], name="consumer")
unnamed = Worker(None)


def sequence(x, fifo):
    fifo.fill(x)


rt = Runtime(sequence, [line_type, of_in.prod()])
module = Program(NPU1Col1(), rt, workers=[named, unnamed]).resolve_program()
ops = list(walk(module.operation))

for op in ops:
    for name in names(op.location):
        assert name.startswith(("iron:", "py:")), (op.name, op.location)

fifo = next(op for op in ops if op.name == "aie.objectfifo")
assert names(fifo.location) == ["iron:of_in"], fifo.location

cores = [op for op in ops if op.name == "aie.core"]
assert [names(core.location)[0] for core in cores] == [
    "iron:consumer",
    "iron:worker0",
], [core.location for core in cores]

acquire = next(op for op in ops if op.name == "aie.objectfifo.acquire")
assert names(acquire.location) == ["py:core_fn"], acquire.location
fill = next(op for op in ops if op.name == "aiex.dma_start_task")
assert "py:sequence" in names(fill.location), fill.location
print("PASS: location name prefixes")
