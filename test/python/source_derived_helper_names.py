# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

# An object a helper function builds and returns is named where the helper's
# caller assigns it, through a few helpers and through returned tuples.

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
