# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# Coverage for shim_dma_single_bd_task's handling of the tap/sizes rank.
#
# The shim DMA BD has 3 access dimensions plus a hardware repeat/iteration
# dimension. shim_dma_single_bd_task hoists sizes[0] into that iteration
# dimension (repeat_count = sizes[0] - 1), but the transferred extent is
# prod(sizes[-3:]). That only lines up when there are 4 dimensions. A tap with
# fewer than 4 dims and sizes[0] > 1 used to be counted both as an access dim
# (in transfer_len and the BD dimensions) and as repeat_count, so the shim
# re-issued the whole transfer sizes[0] times and dma_await_task deadlocked.
#
# Each case builds one task and FileChecks the emitted BD. The fix must:
#  - normalize rank < 4 (leading size > 1) to the canonical 4-dim BD, no repeat
#  - leave the common rank-4 leading-unit case unchanged
#  - keep repeat_count for a genuine rank-4 iteration dim (no over-correction)
#  - work the same for the explicit sizes=/strides= path as for tap=
#  - coalesce a tap of rank > 4 to fit
#  - pass constant sizes of rank > 4 that do not fit, from a tap or explicit,
#    through for the compiler to split, with the repeat count covering every
#    iteration dimension, and leave runtime ones for the compiler to reject

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import MLIRError
from aie.passmanager import PassManager


def case(name, dims, sizes, strides, use_tap=True):
    print(f"// CASE {name}")
    total = 1
    for s in sizes:
        total *= s
    try:
        with mlir_mod_ctx() as ctx:

            @device(AIEDevice.npu1_1col)
            def device_body():
                shim = tile(0, 0)
                core = tile(0, 2)
                of = object_fifo("of", core, shim, 2, T.memref(sizes[-1], T.bf16()))

                @runtime_sequence(T.memref(total, T.bf16()))
                def seq(out):
                    if use_tap:
                        tap = TensorAccessPattern(dims, 0, list(sizes), list(strides))
                        shim_dma_single_bd_task(of, out, tap=tap, issue_token=True)
                    elif strides is None:
                        shim_dma_single_bd_task(
                            of, out, sizes=list(sizes), issue_token=True
                        )
                    else:
                        shim_dma_single_bd_task(
                            of,
                            out,
                            sizes=list(sizes),
                            strides=list(strides),
                            issue_token=True,
                        )

            print(ctx.module)
    except ValueError as e:
        print(f"RAISED ValueError: {e}")


# rank 2, leading > 1: sub-4-dim, must normalize (no repeat).
# CHECK-LABEL: CASE rank2
# CHECK: aie.dma_bd(%{{.*}} : memref<8192xbf16> offset = {{.*}} len = {{.*}} sizes = [1, 1, 8, 1024] strides = [0, 0, 1024, 1])
# CHECK-NOT: repeat_count
case("rank2", (8, 1024), [8, 1024], [1024, 1])

# rank 3, leading > 1 (the deinterleave drain that first exposed this).
# CHECK-LABEL: CASE rank3
# CHECK: aie.dma_bd(%{{.*}} : memref<262144xbf16> offset = {{.*}} len = {{.*}} sizes = [1, 64, 4, 1024] strides = [0, 1024, 65536, 1])
# CHECK-NOT: repeat_count
case("rank3", (4, 64, 1024), [64, 4, 1024], [1024, 65536, 1])

# rank 4, leading unit (the common simple_tiler fill): unchanged, no repeat.
# CHECK-LABEL: CASE rank4_lead1
# CHECK: aie.dma_bd(%{{.*}} : memref<262144xbf16> offset = {{.*}} len = {{.*}} sizes = [1, 1, 64, 4096] strides = [0, 0, 4096, 1])
# CHECK-NOT: repeat_count
case("rank4_lead1", (1, 1, 64, 4096), [1, 1, 64, 4096], [0, 0, 4096, 1])

# rank 4, leading > 1: a genuine iteration dim, repeat_count MUST be kept.
# CHECK-LABEL: CASE rank4_lead4
# CHECK: aie.dma_bd(%{{.*}} : memref<32768xbf16> offset = {{.*}} len = {{.*}} sizes = [4, 1, 8, 1024] strides = [8192, 0, 1024, 1])
# CHECK: repeat_count = 3
case("rank4_lead4", (4, 1, 8, 1024), [4, 1, 8, 1024], [8192, 0, 1024, 1])

# a pure repeat stays outermost as the repeat_count; the padding goes after it.
# CHECK-LABEL: CASE repeat_rank3
# CHECK: aie.dma_bd(%{{.*}} : memref<48xbf16> offset = {{.*}} len = 16 sizes = [3, 1, 1, 16] strides = [0, 0, 0, 1])
# CHECK: repeat_count = 2
case("repeat_rank3", (1, 16), [3, 1, 16], [0, 0, 1])

# a repeated tile: [R, th, tw] becomes [R, 1, th, tw].
# CHECK-LABEL: CASE repeat_tile
# CHECK: aie.dma_bd(%{{.*}} : memref<48xbf16> offset = {{.*}} len = 16 sizes = [3, 1, 2, 8] strides = [0, 0, 8, 1])
# CHECK: repeat_count = 2
case("repeat_tile", (8, 8), [3, 2, 8], [0, 8, 1])

# rank 3 via explicit sizes=/strides= (no tap): same normalization as tap=.
# CHECK-LABEL: CASE rank3_explicit
# CHECK: aie.dma_bd(%{{.*}} : memref<262144xbf16> offset = {{.*}} len = {{.*}} sizes = [1, 64, 4, 1024] strides = [0, 1024, 65536, 1])
# CHECK-NOT: repeat_count
case("rank3_explicit", None, [64, 4, 1024], [1024, 65536, 1], use_tap=False)

# the documented explicit contiguous form (sizes only, no strides) must be
# unchanged: 4-dim, leading units, single linear transfer, no repeat.
# CHECK-LABEL: CASE explicit_contiguous
# CHECK: aie.dma_bd(%{{.*}} : memref<4096xbf16> offset = {{.*}} len = {{.*}} sizes = [1, 1, 1, 4096] strides = [0, 0, 0, 1])
# CHECK-NOT: repeat_count
case("explicit_contiguous", None, [1, 1, 1, 4096], None, use_tap=False)

# rank 5 that coalesces: a contiguous walk is one linear transfer.
# CHECK-LABEL: CASE rank5
# CHECK: aie.dma_bd(%{{.*}} : memref<32xbf16> offset = {{.*}} len = {{.*}} sizes = [1, 1, 1, 32] strides = [0, 0, 0, 1])
# CHECK-NOT: repeat_count
case("rank5", (2, 2, 2, 2, 2), [2, 2, 2, 2, 2], [16, 8, 4, 2, 1])

# rank 5 that does not coalesce: passes through for the compiler to split.
# CHECK-LABEL: CASE rank5_transposed
# CHECK: aie.dma_bd(%{{.*}} : memref<32xbf16> offset = {{.*}} len = 8 sizes = [2, 2, 2, 2, 2] strides = [1, 2, 4, 8, 16])
# CHECK: repeat_count = 3
case("rank5_transposed", (2, 2, 2, 2, 2), [2, 2, 2, 2, 2], [1, 2, 4, 8, 16])

# CHECK-LABEL: CASE rank5_explicit
# CHECK: aie.dma_bd(%{{.*}} : memref<32xbf16> offset = {{.*}} len = 8 sizes = [2, 2, 2, 2, 2] strides = [16, 8, 4, 2, 1])
# CHECK: repeat_count = 3
case("rank5_explicit", None, [2, 2, 2, 2, 2], [16, 8, 4, 2, 1], use_tap=False)

# rank 5 without strides: the default strides take the rank of sizes.
# CHECK-LABEL: CASE rank5_no_strides
# CHECK: aie.dma_bd(%{{.*}} : memref<64xbf16> offset = {{.*}} len = 16 sizes = [2, 2, 1, 1, 16] strides = [0, 0, 0, 0, 1])
# CHECK: repeat_count = 3
case("rank5_no_strides", None, [2, 2, 1, 1, 16], None, use_tap=False)


# rank 5 with a runtime size: the compiler cannot split it, so the shim tile's
# limit rejects it once the fifo lowers.
# CHECK-LABEL: CASE rank5_runtime
# CHECK: RAISED MLIRError
# CHECK-NEXT: 'aie.dma_bd' op Cannot give more than 4 dimensions for step sizes and wraps on this tile (got 5 dimensions)
print("// CASE rank5_runtime")
try:
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            shim = tile(0, 0)
            core = tile(0, 2)
            of = object_fifo("of", core, shim, 2, T.memref(16, T.bf16()))

            @runtime_sequence(T.memref(64, T.bf16()), T.i64())
            def seq(out, n):
                shim_dma_single_bd_task(
                    of,
                    out,
                    sizes=[n, 2, 1, 1, 16],
                    strides=[32, 16, 0, 0, 1],
                    issue_token=True,
                )

        PassManager.parse(
            "builtin.module(aie.device(aie-objectFifo-stateful-transform))"
        ).run(ctx.module.operation)
except MLIRError as e:
    print(f"RAISED MLIRError: {e}")
