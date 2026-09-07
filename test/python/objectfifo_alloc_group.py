# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# The IRON ObjectFifo constructor plumbs alloc_group onto aie.objectfifo, so
# two fifos naming different groups may be overlaid by the allocator.

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2

line_type = np.ndarray[(256,), np.dtype[np.uint8]]

of_a = ObjectFifo(line_type, name="mode_a", alloc_group="a")
of_b = ObjectFifo(line_type, name="mode_b", alloc_group="b")
of_out = ObjectFifo(line_type, name="out")


def core_fn(of_a, of_b, of_out):
    for _ in range_(2):
        elem_out = of_out.acquire(1)
        elem_a = of_a.acquire(1)
        for i in range_(256):
            elem_out[i] = elem_a[i]
        of_a.release(1)
        of_out.release(1)

        elem_out = of_out.acquire(1)
        elem_b = of_b.acquire(1)
        for i in range_(256):
            elem_out[i] = elem_b[i]
        of_b.release(1)
        of_out.release(1)


worker = Worker(core_fn, [of_a.cons(), of_b.cons(), of_out.prod()])


def sequence(a_in, b_in, c_out, a_h, b_h, out_h):
    a_h.fill(a_in)
    b_h.fill(b_in)
    out_h.drain(c_out, wait=True)


rt = Runtime(
    sequence,
    [line_type, line_type, line_type, of_a.prod(), of_b.prod(), of_out.cons()],
)

# CHECK-DAG: aie.objectfifo @mode_a{{.*}}alloc_group = "a"
# CHECK-DAG: aie.objectfifo @mode_b{{.*}}alloc_group = "b"
print(Program(NPU2(), rt, workers=[worker]).resolve_program())
