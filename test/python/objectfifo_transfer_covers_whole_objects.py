# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A fill()/drain() emits one shim BD; the far end of the stream receives whole
# objectFIFO objects.  Nothing downstream compares the two extents, so a transfer
# that ends mid-object used to build clean and then hang: the receiving buffer
# descriptor never completes and the consumer's acquire never unblocks.  Check
# that it is rejected at construction, that an aligned transfer is untouched, that
# a host buffer viewing the stream's bytes through a wider element type is still
# allowed, and that a block-float element type, which reports no width at all,
# still resolves.

# RUN: %python %s | FileCheck %s

import aie.iron as iron
import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import AnyShimTile, from_name

iron.set_current_device(from_name("npu2", n_cols=1))

N, LINE = 1024, 256


def build(fifo_dtype, name, line=LINE, in_ty=None, out_ty=None, decoupled=False):
    """Forward N elements shim -> memtile -> shim, per-object ``line`` elements.

    The host buffers are typed independently of the fifo so either the fill side or
    the drain side can be given a mismatched extent on its own.
    """
    in_ty = in_ty or np.ndarray[(N,), np.dtype[fifo_dtype]]
    out_ty = out_ty or np.ndarray[(N,), np.dtype[fifo_dtype]]
    line_ty = np.ndarray[(line,), np.dtype[fifo_dtype]]

    of_in = ObjectFifo(line_ty, name=f"in_{name}", stream_len_decoupled=decoupled)
    of_out = of_in.cons().forward(
        name=f"out_{name}", stream_len_decoupled=decoupled
    )

    def sequence(a, c, in_h, out_h):
        in_h.fill(a)
        out_h.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [in_ty, out_ty, of_in.prod(tile=AnyShimTile), of_out.cons(tile=AnyShimTile)],
    )
    return str(Program(iron.get_current_device(), rt).resolve_program())


# An aligned transfer is unaffected: the BD carries the host buffer's element type,
# and its len is the element count of that type, not a byte count.
# CHECK-LABEL: TEST: aligned_transfer_is_unchanged
# CHECK: aie.objectfifo @in_ok{{.*}}!aie.objectfifo<memref<256xf32>>
# CHECK: aie.dma_bd(%{{.*}} : memref<1024xf32> offset = 0 len = 1024
print("TEST: aligned_transfer_is_unchanged")
print(build(np.float32, "ok"))

# 1024 elements over a 300-element object is 3.41 objects; the last one never fills.
# CHECK-LABEL: TEST: partial_object_on_the_fill_side_is_rejected
# CHECK: fill() on ObjectFifo 'in_partial': the transfer covers 1024 f32,
# CHECK-SAME: which is 3.413 of the fifo's object of 300 f32
print("TEST: partial_object_on_the_fill_side_is_rejected")
try:
    build(np.float32, "partial", line=300)
    raise AssertionError("expected a partial-object transfer to be rejected")
except ValueError as e:
    print(e)

# The drain side is checked on its own: here the fill is aligned and only the
# destination buffer is short of a whole object.
# CHECK-LABEL: TEST: the_drain_side_is_checked_too
# CHECK: drain() on ObjectFifo 'out_short'
print("TEST: the_drain_side_is_checked_too")
try:
    build(np.float32, "short", out_ty=np.ndarray[(N - 32,), np.dtype[np.float32]])
    raise AssertionError("expected a partial-object transfer to be rejected")
except ValueError as e:
    print(e)

# The element types need not agree. A DMA does not interpret its payload, and a
# host buffer that views the stream's bytes through a wider type is an ordinary
# idiom - ml/mobilenet reuses one i32 allocation for i8 activations and ui16 FC
# data. 256 i32 is 1024 i8, an exact 4 objects, so it must still build.
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

# When the element types differ the extents are only comparable in bits, and the
# message says so: 256 i32 is 1024 bytes against a 300-byte object.
# CHECK-LABEL: TEST: a_mismatched_view_that_does_not_divide_is_rejected
# CHECK: which is 3.413 of the fifo's object of 2400 bits (300 i8)
print("TEST: a_mismatched_view_that_does_not_divide_is_rejected")
try:
    build(
        np.int8,
        "mix",
        line=300,
        in_ty=np.ndarray[(256,), np.dtype[np.int32]],
        out_ty=np.ndarray[(256,), np.dtype[np.int32]],
    )
    raise AssertionError("expected a mismatched partial transfer to be rejected")
except ValueError as e:
    print(e)

# A block-float element type has no bit width to ask for, so an extent in a common
# unit cannot be computed for a differing pair; identical types still compare
# directly in elements, which keeps this resolvable.
# CHECK-LABEL: TEST: block_float_still_resolves
# CHECK: aie.objectfifo{{.*}}!aiex.bfp<"v8bfp16ebs8">
print("TEST: block_float_still_resolves")
print(build(v8bfp16ebs8, "bfp"))


# The access pattern may also be given as explicit sizes/strides, with the
# outermost dimension becoming the shim BD's repeat count, so the stream sees
# prod(sizes) elements, not prod(sizes[-3:]).  test/npu-xrt uses this form, and
# `matmul_whole_array_dynamic` also passes transfer_len alongside it, in which
# case the BD's own len covers the inner three dimensions and sizes[0] repeats
# it.  Both are counted here; neither is reachable from tap=.


def build_explicit(name, fifo_dtype, buf_elems, **kwargs):
    line_ty = np.ndarray[(LINE,), np.dtype[fifo_dtype]]
    buf_ty = np.ndarray[(buf_elems,), np.dtype[fifo_dtype]]
    of_in = ObjectFifo(line_ty, name=f"in_{name}")
    of_out = of_in.cons().forward(name=f"out_{name}")

    def sequence(a, c, in_h, out_h):
        in_h.fill(a, **kwargs)
        out_h.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [buf_ty, buf_ty, of_in.prod(tile=AnyShimTile), of_out.cons(tile=AnyShimTile)],
    )
    return str(Program(iron.get_current_device(), rt).resolve_program())


# 4 repeats of 256 elements is 4 whole objects, and the BD carries the inner len.
# CHECK-LABEL: TEST: explicit_sizes_count_the_repeat_dimension
# CHECK: aie.dma_bd(%{{.*}} : memref<1024xf32> offset = 0 len = 256
print("TEST: explicit_sizes_count_the_repeat_dimension")
print(
    build_explicit("rep", np.float32, N, sizes=[4, 1, 1, 256], strides=[256, 0, 0, 1])
)

# Same pattern, one element short per issue: 4 x 255 is 3.984 objects.
# CHECK-LABEL: TEST: explicit_sizes_are_checked_across_the_repeat
# CHECK: which is 3.984 of the fifo's object of 256 f32
print("TEST: explicit_sizes_are_checked_across_the_repeat")
try:
    build_explicit(
        "repbad", np.float32, N, sizes=[4, 1, 1, 255], strides=[256, 0, 0, 1]
    )
    raise AssertionError("expected a partial-object repeat to be rejected")
except ValueError as e:
    print(e)

# An explicit transfer_len overrides the inner dimensions; sizes[0] still repeats
# it. 2 x 128 elements is one object, so this builds.
# CHECK-LABEL: TEST: explicit_transfer_len_overrides_the_inner_dimensions
# CHECK: aie.dma_bd(%{{.*}} : memref<1024xf32> offset = 0 len = 128
print("TEST: explicit_transfer_len_overrides_the_inner_dimensions")
print(
    build_explicit(
        "len",
        np.float32,
        N,
        sizes=[2, 1, 1, 256],
        strides=[128, 0, 0, 1],
        transfer_len=128,
    )
)


# A (de)compressing channel moves a byte count the object size cannot predict, so
# the same partial extent rejected above is well formed here.
# CHECK-LABEL: TEST: a_decoupled_stream_is_exempt
# CHECK: aie.objectfifo @in_decoupled{{.*}}stream_len_decoupled
print("TEST: a_decoupled_stream_is_exempt")
print(build(np.float32, "decoupled", line=300, decoupled=True))
