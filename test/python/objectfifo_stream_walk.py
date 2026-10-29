# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""A forwarded fifo's to_stream may walk the whole stream of input objects.

Three-row windows over the rows of an image: the pattern's tensor is the
image, its outer dimension enumerates windows a row apart, and padding that
dimension adds a zero row above and below.
"""

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2Col1

iron.set_current_device(NPU2Col1())
ROWS, COLS = 4, 8
row_ty = np.ndarray[(COLS,), np.dtype[np.int32]]
window_ty = np.ndarray[(3, COLS), np.dtype[np.int32]]
image_ty = np.ndarray[(ROWS * COLS,), np.dtype[np.int32]]
out_ty = np.ndarray[(ROWS * 3 * COLS,), np.dtype[np.int32]]


def copy_fn(a, b):
    x = a.acquire(1)
    y = b.acquire(1)
    for i in range(3):
        for j in range(COLS):
            y[i, j] = x[i, j]
    a.release(1)
    b.release(1)


def seq(x, y, in_h, out_h):
    in_h.fill(x)
    out_h.drain(y, wait=True)


def resolved(walk):
    of_rows = ObjectFifo(row_ty, name="rows")
    of_windows = of_rows.cons().forward(
        obj_type=window_ty, name="windows", to_stream=walk
    )
    of_out = ObjectFifo(window_ty, name="out")
    worker = Worker(copy_fn, [of_windows.cons(), of_out.prod()])
    rt = Runtime(seq, [image_ty, out_ty, of_rows.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def windows(rows):
    tap = TensorAccessPattern((rows, COLS), 0, [ROWS, 3, COLS], [COLS, COLS, 1])
    return tap.pad([(1, 1), (0, 0), (0, 0)])


for label, walk in [("image", windows(ROWS)), ("taller tensor", windows(ROWS + 1))]:
    try:
        module = resolved(walk)
        print(f"{label}: accepted")
        print(module)
    except ValueError as e:
        print(f"{label}: {e}")
# CHECK: image: accepted
# CHECK: aie.objectfifo @windows({{.*}}dimensionsToStream [<size = 4, stride = 8>, <size = 3, stride = 8>, <size = 8, stride = 1>]
# CHECK-SAME: padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 1, const_pad_after = 1>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>]>
# CHECK-SAME: !aie.objectfifo<memref<3x8xi32>>
# CHECK: aie.objectfifo.link [@rows] -> [@windows]
# CHECK: taller tensor: windows to_stream {{.*}} walks a stream of 4 objects of 8 elements, but its tensor has shape (5, 8)
