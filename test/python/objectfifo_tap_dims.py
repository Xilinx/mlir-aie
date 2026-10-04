# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""An ObjectFifo's stream walks are TensorAccessPatterns over one object.

`to_stream` / `from_stream` take a `TensorAccessPattern` (a padded one also
pads the stream) on the constructor, `cons()`, `forward()`, `split()` and
`join()` alike.
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

# The (size, stride) pairs the patterns stand for.
# CHECK: aie.objectfifo @in_l1({{.*}}dimensionsToStream [<size = 8, stride = 1>, <size = 16, stride = 8>]
# CHECK: aie.objectfifo @mid({{.*}}dimensionsFromStream [<size = 8, stride = 1>, <size = 16, stride = 8>]
# A padded pattern carries the padding too.
# CHECK: aie.objectfifo @out({{.*}}dimensionsToStream [<size = 8, stride = 16>, <size = 16, stride = 1>]
# CHECK-SAME: padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 4, const_pad_after = 4>, <const_pad_before = 0, const_pad_after = 0>]>

# A walk must start at the object's first element and be a pattern.
for label, dims in [
    ("slice", TensorAccessPattern.full((ROWS, COLS))[2:6]),
    ("tap", TensorAccessPattern((ROWS, COLS), COLS, [4, COLS], [COLS, 1])),
    ("list", [(ROWS, COLS), (COLS, 1)]),
]:
    try:
        ObjectFifo(tile_ty, name=f"bad_{label}", to_stream=dims)
        print(f"{label}: accepted")
    except (TypeError, ValueError) as e:
        print(f"{label}: {type(e).__name__}: {e}")
# CHECK: slice: ValueError: to_stream {{.*}} has offset 32, but an objectfifo walks each transfer from its first element
# CHECK: tap: ValueError: to_stream {{.*}} has offset 16, but
# CHECK: list: TypeError: to_stream takes a TensorAccessPattern, got list

# Only the producer's to_stream can pad.
try:
    of_mid.cons(from_stream=padded)
    print("padded from_stream: accepted")
except ValueError as e:
    print("padded from_stream:", e)
# CHECK: padded from_stream: only a producer's walk can pad, but from_stream is


def resolved(dst_walk, joined=False):
    """Resolve `src` -> worker -> `dst`, or two workers -> halves -> join -> `dst`."""
    of_dst = ObjectFifo(tile_ty, name="dst", to_stream=dst_walk)
    if joined:
        half_ty = np.ndarray[(ROWS // 2, COLS), np.dtype[np.int32]]
        halves = of_dst.prod().join([0, ROWS // 2 * COLS], obj_types=[half_ty] * 2)
        workers = [Worker(fill_fn, [h.prod()]) for h in halves]
        rt = Runtime(drain, [tile_ty, of_dst.cons()])
    else:
        of_src = ObjectFifo(tile_ty, name="src")
        workers = [Worker(core_fn, [of_src.cons(), of_dst.prod()])]
        rt = Runtime(seq, [tile_ty, tile_ty, of_src.prod(), of_dst.cons()])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def fill_fn(a):
    x = a.acquire(1)
    for i in range_(ROWS // 2):
        for j in range_(COLS):
            x[i, j] = 0
    a.release(1)


def drain(c, out_h):
    out_h.drain(c, wait=True)


# A walk covers what each transfer moves once the program resolves: one
# object, or a join's segment of it; padding must fit the object it fills.
half_transpose = TensorAccessPattern.full((COLS, ROWS // 2)).T
for label, dims, joined in [
    ("other tensor", TensorAccessPattern.full((ROWS, 2 * COLS)), False),
    ("overflowing pad", padded, False),
    ("join, whole object", transpose, True),
    ("join, each segment", half_transpose, True),
]:
    try:
        module = resolved(dims, joined)
        print(f"{label}: accepted")
        print(module)
    except ValueError as e:
        print(f"{label}: {e}")
# CHECK: other tensor: dst to_stream {{.*}} walks a tensor of shape (8, 32), but each transfer moves 128 elements
# CHECK: overflowing pad: dst to_stream {{.*}} emits 256 elements, but each object it fills has 128
# CHECK: join, whole object: dst to_stream {{.*}} walks a tensor of shape (16, 8), but each transfer moves 64 elements
# CHECK: join, each segment: accepted
# CHECK: aie.objectfifo @dst({{.*}}dimensionsToStream [<size = 4, stride = 1>, <size = 16, stride = 4>]
