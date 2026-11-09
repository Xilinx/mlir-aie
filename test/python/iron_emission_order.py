# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test that a design's names do not change the order its device is emitted in.

The SA placer searches the logical tiles in the order they are declared, so
renaming an ObjectFifo must not reorder them.
"""

import re

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2

ty = np.ndarray[(64,), np.dtype[np.int32]]


def build(names: dict[str, str | None]) -> list[str]:
    of_in = ObjectFifo(ty, name=names["in"])
    of_mid = ObjectFifo(ty, name=names["mid"])
    of_out = ObjectFifo(ty, name=names["out"])

    def copy(src, dst):
        a = src.acquire(1)
        b = dst.acquire(1)
        for i in range_(64):
            b[i] = a[i]
        src.release(1)
        dst.release(1)

    first = Worker(copy, [of_in.cons(), of_mid.prod()])
    second = Worker(copy, [of_mid.cons(), of_out.prod()])

    def sequence(a, c, in_h, out_h):
        in_h.fill(a)
        out_h.drain(c, wait=True)

    rt = Runtime(sequence, [ty, ty, of_in.prod(), of_out.cons()])
    module = str(Program(NPU2(), rt, workers=[first, second]).resolve_program())
    role = {name: r for r, name in names.items() if name is not None}
    emitted = re.findall(r"aie\.objectfifo @(\w+)\(", module)
    return [role.get(name, "?") for name in emitted]


print("\nTEST: fifo_order_ignores_names")
plain = build({"in": None, "mid": None, "out": None})
ascending = build({"in": "a_in", "mid": "b_mid", "out": "c_out"})
descending = build({"in": "z_in", "mid": "y_mid", "out": "x_out"})
assert ascending == descending, (ascending, descending)
assert len(plain) == len(ascending)
print(" ".join(ascending))

# CHECK-LABEL: fifo_order_ignores_names
# CHECK: in out mid
