# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# A runtime sequence's transfers are named after the buffer and the channel
# they move it through, iron:<parameter>@<channel>: the task, its BDs, and the
# start, await and free that act on it.

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1

line_type = np.ndarray[(16,), np.dtype[np.int32]]
of_in = ObjectFifo(line_type)
of_out = ObjectFifo(line_type)


def core_fn(a, b):
    a.acquire(1)
    b.acquire(1)
    a.release(1)
    b.release(1)


def sequence(A, C, a, c):
    a.fill(A)
    c.drain(C, wait=True)


worker = Worker(core_fn, [of_in.cons(), of_out.prod()])
rt = Runtime(sequence, [line_type, line_type, of_in.prod(), of_out.cons()])
module = Program(NPU1Col1(), rt, workers=[worker]).resolve_program()
print(module.operation.get_asm(enable_debug_info=True))

# CHECK: %[[IN:.*]] = aiex.dma_configure_task_for @of_in {
# CHECK-NEXT: aie.dma_bd({{.*}}) loc(#[[A:loc[0-9]*]])
# CHECK: aiex.dma_start_task(%[[IN]]) loc(#[[A]])
# CHECK: %[[OUT:.*]] = aiex.dma_configure_task_for @of_out {
# CHECK-NEXT: aie.dma_bd({{.*}}) loc(#[[C:loc[0-9]*]])
# CHECK: aiex.dma_start_task(%[[OUT]]) loc(#[[C]])
# CHECK: aiex.dma_await_task(%[[OUT]]) loc(#[[C]])
# CHECK-DAG: #[[A]] = loc("iron:A@of_in"(
# CHECK-DAG: #[[C]] = loc("iron:C@of_out"(
