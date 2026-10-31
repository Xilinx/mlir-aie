# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

# An object a helper function builds and returns is named where the helper's
# caller assigns it, through a few helpers and through returned tuples.

import importlib
import os
import sys
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

# A helper and its caller imported through a relative sys.path entry, so
# their co_filename is relative, still name the object after the working
# directory changes before the design resolves.
HELPER = """\
import numpy as np
from aie.iron import Buffer


def make():
    return Buffer(np.ndarray[(8,), np.dtype[np.int8]], initial_value=np.zeros(8, np.int8))
"""
CALLER = """\
from rel_helper import make


def build():
    rel_wts = make()
    return rel_wts
"""
cwd = os.getcwd()
with tempfile.TemporaryDirectory() as tmp:
    os.mkdir(os.path.join(tmp, "rel"))
    with open(os.path.join(tmp, "rel", "rel_helper.py"), "w") as f:
        f.write(HELPER)
    with open(os.path.join(tmp, "rel", "rel_design.py"), "w") as f:
        f.write(CALLER)
    os.chdir(tmp)
    sys.path.insert(0, "rel")
    try:
        rel_wts = importlib.import_module("rel_design").build()
    finally:
        sys.path.remove("rel")
        os.chdir(cwd)
    module = Program(
        NPU1Col1(),
        Runtime(sequence, [np.ndarray[(8,), i8]]),
        workers=[Worker(core_fn, [rel_wts])],
    ).resolve_program()
names = [
    op.attributes["sym_name"].value
    for op in walk(module.operation)
    if op.name == "aie.buffer" and "sym_name" in op.attributes
]
assert names == ["rel_wts"], names
print("PASS: helper-returned name through a relative import path")
