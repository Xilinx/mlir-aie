# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Tiles a configuration declares for fifos no worker or runtime names directly.

The middle of a chain of forwards is reached only through the links on either
side, and each of two fifos keeps its pool on a delegate tile of its own.
"""

import aie.iron as iron
import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile

iron.set_current_device(NPU2())
row_ty = np.ndarray[(16,), np.dtype[np.int32]]
rows_ty = np.ndarray[(64,), np.dtype[np.int32]]


def copy_fn(a, b):
    x = a.acquire(1)
    y = b.acquire(1)
    for i in range(16):
        y[i] = x[i]
    a.release(1)
    b.release(1)


def touch_fn(a):
    a.acquire(1)
    a.release(1)


def seq(x, y, in_h, out_h):
    in_h.fill(x)
    out_h.drain(y, wait=True)


rows = ObjectFifo(row_ty, name="rows")
held = (
    rows.cons()
    .forward(name="hold0")
    .cons()
    .forward(name="hold1")
    .cons()
    .forward(name="hold2")
)
out = ObjectFifo(row_ty, name="out")
workers = [Worker(copy_fn, [held.cons(), out.prod()])]
for col in (4, 5):
    shared = ObjectFifo(row_ty, name=f"shared{col}", delegate_tile=Tile(col, 3))
    workers.append(Worker(touch_fn, [shared.prod()], tile=Tile(col, 2)))
    workers.append(Worker(touch_fn, [shared.cons()], tile=Tile(col, 4)))
rt = Runtime(seq, [rows_ty, rows_ty, rows.prod(), out.cons()])
print(Program(iron.get_current_device(), rt, workers=workers).resolve_program())
# CHECK-DAG: %[[D4:.*]] = aie.logical_tile<CoreTile>(4, 3)
# CHECK-DAG: %[[D5:.*]] = aie.logical_tile<CoreTile>(5, 3)
# CHECK-DAG: aie.objectfifo @hold0(%[[M0:.*]], {%[[M1:.*]]}
# CHECK-DAG: aie.objectfifo @hold1(%[[M1]], {%[[M2:.*]]}
# CHECK-DAG: aie.objectfifo @hold2(%[[M2]],
# CHECK-DAG: %[[M1]] = aie.logical_tile<MemTile>(?, ?)
# CHECK-DAG: aie.objectfifo.link [@hold0] -> [@hold1]
# CHECK-DAG: aie.objectfifo.link [@hold1] -> [@hold2]
# CHECK-DAG: aie.objectfifo.allocate @shared4(%[[D4]])
# CHECK-DAG: aie.objectfifo.allocate @shared5(%[[D5]])
