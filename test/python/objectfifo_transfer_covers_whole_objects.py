# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A fill()/drain() on an ObjectFifo must cover whole objects: the far end's
# buffer descriptor is sized by the object type, so a transfer ending mid-object
# never completes and the design hangs. The aiex.dma_configure_task_for verifier
# rejects it when the program resolves, at the fill()/drain() call's location.

# RUN: %python %s | FileCheck %s

import inspect

import aie.iron as iron
import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import AnyShimTile, from_name
from aie.ir import MLIRError

iron.set_current_device(from_name("npu2", n_cols=1))

N, LINE = 1024, 256


def build(fifo_dtype, name, line=LINE, in_ty=None, out_ty=None, in_tap=None):
    """Forward N elements shim -> memtile -> shim in objects of `line` elements."""
    in_ty = in_ty or np.ndarray[(N,), np.dtype[fifo_dtype]]
    out_ty = out_ty or np.ndarray[(N,), np.dtype[fifo_dtype]]
    line_ty = np.ndarray[(line,), np.dtype[fifo_dtype]]

    of_in = ObjectFifo(line_ty, name=f"in_{name}")
    of_out = of_in.cons().forward(name=f"out_{name}")

    def sequence(a, c, in_h, out_h):
        in_h.fill(a, tap=in_tap)
        out_h.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [in_ty, out_ty, of_in.prod(tile=AnyShimTile), of_out.cons(tile=AnyShimTile)],
    )
    return str(Program(iron.get_current_device(), rt).resolve_program())


def build_concat(first, second, name):
    """Fill one fifo from two host buffers of `first` and `second` f32."""
    line_ty = np.ndarray[(LINE,), np.dtype[np.float32]]
    of_in = ObjectFifo(line_ty, name=f"in_{name}")
    of_out = of_in.cons().forward(name=f"out_{name}")

    def sequence(x, y, c, in_h, out_h):
        in_h.fill(x)
        in_h.fill(y)
        out_h.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [
            np.ndarray[(first,), np.dtype[np.float32]],
            np.ndarray[(second,), np.dtype[np.float32]],
            np.ndarray[(first + second,), np.dtype[np.float32]],
            of_in.prod(tile=AnyShimTile),
            of_out.cons(tile=AnyShimTile),
        ],
    )
    return str(Program(iron.get_current_device(), rt).resolve_program())


def rejected(builder=build, **kwargs):
    try:
        builder(**kwargs)
    except MLIRError as e:
        return str(e)
    raise AssertionError("expected a partial-object transfer to be rejected")


# CHECK-LABEL: TEST: aligned_transfer_is_unchanged
# CHECK: aie.objectfifo @in_ok{{.*}}!aie.objectfifo<memref<256xf32>>
# CHECK: aie.dma_bd(%{{.*}} : memref<1024xf32> offset = 0 len = 1024
print("TEST: aligned_transfer_is_unchanged")
print(build(np.float32, "ok"))

# 1024 f32 over a 300-f32 object is 3.41 objects. The error points at fill().
# CHECK-LABEL: TEST: partial_object_on_the_fill_side_is_rejected
# CHECK: fill_line=[[LINE:[0-9]+]]
# CHECK: objectfifo_transfer_covers_whole_objects.py":[[LINE]]:{{.*}}moves 4096 bytes through objectFIFO @in_partial, which is not a whole number of its 1200-byte objects
print("TEST: partial_object_on_the_fill_side_is_rejected")
lines, first = inspect.getsourcelines(build)
print(f"fill_line={first + next(i for i, l in enumerate(lines) if 'in_h.fill(' in l)}")
print(rejected(fifo_dtype=np.float32, name="partial", line=300))

# The drain side is checked on its own: only the destination is short.
# CHECK-LABEL: TEST: the_drain_side_is_checked_too
# CHECK: moves 3968 bytes through objectFIFO @out_short, which is not a whole number of its 1024-byte objects
print("TEST: the_drain_side_is_checked_too")
print(
    rejected(
        fifo_dtype=np.float32,
        name="short",
        out_ty=np.ndarray[(N - 32,), np.dtype[np.float32]],
    )
)

# Only the extent is compared, not the element type: ml/mobilenet views i8 and
# ui16 streams through one i32 host allocation. 256 i32 is 4 objects of 256 i8.
# CHECK-LABEL: TEST: a_wider_host_view_of_the_same_bytes_is_allowed
# CHECK: aie.objectfifo @in_view{{.*}}!aie.objectfifo<memref<256xi8>>
# CHECK: aie.dma_bd(%{{.*}} : memref<256xi32> offset = 0 len = 256
print("TEST: a_wider_host_view_of_the_same_bytes_is_allowed")
print(
    build(
        np.int8,
        "view",
        in_ty=np.ndarray[(N // 4,), np.dtype[np.int32]],
        out_ty=np.ndarray[(N // 4,), np.dtype[np.int32]],
    )
)

# CHECK-LABEL: TEST: a_mismatched_view_that_does_not_divide_is_rejected
# CHECK: moves 1024 bytes through objectFIFO @in_mix, which is not a whole number of its 300-byte objects
print("TEST: a_mismatched_view_that_does_not_divide_is_rejected")
print(
    rejected(
        fifo_dtype=np.int8,
        name="mix",
        line=300,
        in_ty=np.ndarray[(256,), np.dtype[np.int32]],
        out_ty=np.ndarray[(256,), np.dtype[np.int32]],
    )
)

# CHECK-LABEL: TEST: block_float_still_resolves
# CHECK: aie.objectfifo{{.*}}!aiex.bfp<"v8bfp16ebs8">
print("TEST: block_float_still_resolves")
print(build(v8bfp16ebs8, "bfp"))

# The outermost tap dimension becomes the shim BD's repeat count, so the stream
# sees prod(sizes): 4 x 256 is 4 objects, 4 x 255 is not.
# CHECK-LABEL: TEST: the_repeat_dimension_is_counted
# CHECK: aie.dma_bd(%{{.*}} : memref<1024xf32> offset = 0 len = 256
# CHECK: moves 4080 bytes through objectFIFO @in_repbad
print("TEST: the_repeat_dimension_is_counted")
print(
    build(
        np.float32,
        "rep",
        in_tap=TensorAccessPattern((N,), 0, [4, 1, 1, 256], [256, 0, 0, 1]),
    )
)
print(
    rejected(
        fifo_dtype=np.float32,
        name="repbad",
        in_tap=TensorAccessPattern((N,), 0, [4, 1, 1, 255], [256, 0, 0, 1]),
    )
)

# Transfers may share an object: 128 + 896 f32 straddle an object boundary
# but are 4 whole objects together.
# CHECK-LABEL: TEST: fills_that_share_an_object_are_allowed
# CHECK: aie.dma_bd(%{{.*}} : memref<128xf32> offset = 0 len = 128
# CHECK: aie.dma_bd(%{{.*}} : memref<896xf32> offset = 0 len = 896
print("TEST: fills_that_share_an_object_are_allowed")
print(build_concat(first=128, second=896, name="concat"))

# 128 + 1024 f32 leave the last object half full; the first fill reports it.
# CHECK-LABEL: TEST: fills_that_leave_an_object_partial_are_rejected
# CHECK: fill_line=[[LINE:[0-9]+]]
# CHECK: objectfifo_transfer_covers_whole_objects.py":[[LINE]]:{{.*}}moves 512 bytes through objectFIFO @in_concatbad, which, with this runtime sequence's other transfers through it (4608 bytes in all), is not a whole number of its 1024-byte objects
print("TEST: fills_that_leave_an_object_partial_are_rejected")
lines, first = inspect.getsourcelines(build_concat)
print(
    f"fill_line={first + next(i for i, l in enumerate(lines) if 'in_h.fill(x)' in l)}"
)
print(rejected(build_concat, first=128, second=1024, name="concatbad"))
