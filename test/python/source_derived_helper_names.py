# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

# An object a helper function builds and returns is named where the helper's
# caller assigns it, through a few helpers and through returned tuples.

import os
import tempfile

import numpy as np
from aie.iron import Buffer, Program, Runtime, Worker
from aie.iron.device import NPU1Col1

i8 = np.dtype[np.int8]


def wts_buffer(n):
    return Buffer(np.ndarray[(n,), i8], initial_value=np.zeros(n, np.int8))


def via(n):
    return wts_buffer(n)


def pack(n):
    return Buffer(np.ndarray[(n,), i8], initial_value=np.zeros(n, np.int8)), [0]


def built_in_argument(n):
    return [wts_buffer(n)]


# Returned once and twice.
conv_wts = wts_buffer(8)
deep_wts = via(8)
# One element of a returned tuple, unpacked by the caller.
packed_wts, offsets = pack(8)
# Assigned as a whole tuple: no single element is the target.
whole = pack(8)
# Not the returned value itself: left unnamed.
(listed,) = built_in_argument(8)


def walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from walk(child.operation)


bufs = [conv_wts, deep_wts, packed_wts, whole[0], listed]


def core_fn(*args):
    pass


worker = Worker(core_fn, bufs)


def sequence(x):
    pass


rt = Runtime(sequence, [np.ndarray[(8,), i8]])
module = Program(NPU1Col1(), rt, workers=[worker]).resolve_program()
buffers = [op for op in walk(module.operation) if op.name == "aie.buffer"]
names = [
    op.attributes["sym_name"].value for op in buffers if "sym_name" in op.attributes
]
assert {"conv_wts", "deep_wts", "packed_wts"} <= set(names), names
assert "whole" not in names and "listed" not in names, names
# The rest are left for the compiler to name.
assert len(buffers) == len(bufs) and len(names) == 3, names
print("PASS: helper-returned names")

# A runner that compiles a design under a relative name gives it a relative
# co_filename; an object a helper returns to that design is still named after
# its variable when the directory changes before the design resolves.
cwd = os.getcwd()
with tempfile.TemporaryDirectory() as tmp:
    with open(os.path.join(tmp, "rel_design.py"), "w") as f:
        f.write("rel_wts = wts_buffer(8)\n")
    os.chdir(tmp)
    try:
        with open("rel_design.py") as f:
            design = {"wts_buffer": wts_buffer}
            exec(compile(f.read(), "rel_design.py", "exec"), design)
    finally:
        os.chdir(cwd)
    module = Program(
        NPU1Col1(),
        Runtime(sequence, [np.ndarray[(8,), i8]]),
        workers=[Worker(core_fn, [design["rel_wts"]])],
    ).resolve_program()
names = [
    op.attributes["sym_name"].value
    for op in walk(module.operation)
    if op.name == "aie.buffer" and "sym_name" in op.attributes
]
assert names == ["rel_wts"], names
print("PASS: helper-returned name in a design compiled under a relative name")
