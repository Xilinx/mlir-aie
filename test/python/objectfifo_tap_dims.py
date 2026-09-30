# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""An ObjectFifo takes a TensorAccessPattern where it takes a dims list.

`to_stream` / `from_stream` accept a `TensorAccessPattern` (a padded one
also supplies `pad_dimensions`) on the constructor, `cons()`, `forward()`,
`split()` and `join()` alike.
"""

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1

iron.set_current_device(NPU2Col1())
ROWS, COLS, PAD = 8, 16, 4
tile_ty = np.ndarray[(ROWS, COLS), np.dtype[np.int32]]
padded_ty = np.ndarray[((ROWS + 2 * PAD) * COLS,), np.dtype[np.int32]]

transpose = TensorAccessPattern.full((COLS, ROWS)).T
padded = TensorAccessPattern.full((ROWS, COLS)).pad([(PAD, PAD), (0, 0)])

of_in = ObjectFifo(tile_ty, name="in")
of_in_l1 = of_in.cons().forward(name="in_l1", to_stream=transpose)
of_mid = ObjectFifo(tile_ty, name="mid")
of_out = of_mid.cons(from_stream=transpose).forward(
    obj_type=padded_ty, name="out", to_stream=padded, pad_value=0
)


def core_fn(a, b):
    x = a.acquire(1)
    y = b.acquire(1)
    for i in range_(ROWS):
        for j in range_(COLS):
            y[i, j] = x[i, j]
    a.release(1)
    b.release(1)


worker = Worker(core_fn, [of_in_l1.cons(), of_mid.prod()])


def seq(a, c, in_h, out_h):
    in_h.fill(a)
    out_h.drain(c, wait=True)


rt = Runtime(seq, [tile_ty, padded_ty, of_in.prod(), of_out.cons()])
print(Program(iron.get_current_device(), rt, workers=[worker]).resolve_program())

# The same [(size, stride), ...] lists the patterns stand for.
# CHECK: aie.objectfifo @in_l1({{.*}}dimensionsToStream [<size = 8, stride = 1>, <size = 16, stride = 8>]
# CHECK: aie.objectfifo @mid({{.*}}dimensionsFromStream [<size = 8, stride = 1>, <size = 16, stride = 8>]
# A padded pattern carries the padding too.
# CHECK: aie.objectfifo @out({{.*}}dimensionsToStream [<size = 8, stride = 16>, <size = 16, stride = 1>]
# CHECK-SAME: padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 4, const_pad_after = 4>, <const_pad_before = 0, const_pad_after = 0>]>

# ObjectFifo dims cannot encode an offset, so a pattern that starts past element
# 0 is rejected instead of silently walking from the start.
for label, dims in [
    ("slice", TensorAccessPattern.full((ROWS, COLS))[2:6]),
    ("padded slice", TensorAccessPattern.full((ROWS, COLS))[2:6].pad([(1, 1), (0, 0)])),
    ("tap", TensorAccessPattern((ROWS, COLS), COLS, [4, COLS], [COLS, 1])),
]:
    try:
        ObjectFifo(tile_ty, name=f"off_{label}", to_stream=dims)
        print(f"{label}: accepted")
    except ValueError as e:
        print(f"{label}:", str(e)[:48])
# CHECK: slice: ObjectFifo stream dimensions cannot encode an
# CHECK: padded slice: ObjectFifo stream dimensions cannot encode an
# CHECK: tap: ObjectFifo stream dimensions cannot encode an

# Only the producer's to_stream can pad.
try:
    of_mid.cons(from_stream=padded)
    print("padded from_stream: accepted")
except ValueError as e:
    print("padded from_stream:", str(e)[:30])
# CHECK: padded from_stream: only a producer's to_stream
