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
#  - pass a constant rank > 4 through for the compiler to split, with the
#    repeat count covering every iteration dimension, and reject a runtime one

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorAccessPattern


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

# rank 5, constant: every dimension before the last three is iterated, so the
# task repeats 2 * 2 times over [2, 2, 2]; the compiler splits the BD into BDs
# of 4 dimensions.
# CHECK-LABEL: CASE rank5
# CHECK: aie.dma_bd(%{{.*}} : memref<32xbf16> offset = {{.*}} len = 8 sizes = [2, 2, 2, 2, 2] strides = [16, 8, 4, 2, 1])
# CHECK: repeat_count = 3
case("rank5", (2, 2, 2, 2, 2), [2, 2, 2, 2, 2], [16, 8, 4, 2, 1])

# CHECK-LABEL: CASE rank5_explicit
# CHECK: aie.dma_bd(%{{.*}} : memref<32xbf16> offset = {{.*}} len = 8 sizes = [2, 2, 2, 2, 2] strides = [16, 8, 4, 2, 1])
# CHECK: repeat_count = 3
case("rank5_explicit", None, [2, 2, 2, 2, 2], [16, 8, 4, 2, 1], use_tap=False)

# rank 5 without strides: the default strides take the rank of sizes.
# CHECK-LABEL: CASE rank5_no_strides
# CHECK: aie.dma_bd(%{{.*}} : memref<64xbf16> offset = {{.*}} len = 16 sizes = [2, 2, 1, 1, 16] strides = [0, 0, 0, 0, 1])
# CHECK: repeat_count = 3
case("rank5_no_strides", None, [2, 2, 1, 1, 16], None, use_tap=False)


# rank 5 with a runtime size: the compiler cannot split it, so it must raise
# rather than emit a BD it cannot lower.
# CHECK-LABEL: CASE rank5_runtime
# CHECK: RAISED ValueError: a DMA BD with more than 4 dimensions (got 5) needs constant sizes and strides
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

        print(ctx.module)
except ValueError as e:
    print(f"RAISED ValueError: {e}")


# Unlike shim_dma_bd, a tile BD has no default strides: sizes alone would reach
# aie.dma_bd without strides and fail its verifier.
# CHECK-LABEL: CASE tile_sizes_only
# CHECK: RAISED ValueError: tile_dma_single_bd_task needs sizes and strides together, got sizes=[4, 16] and strides=None
print("// CASE tile_sizes_only")
try:
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            mem = tile(0, 1)
            resident = buffer(mem, T.memref(64, T.bf16(), memory_space=1), name="r")

            @runtime_sequence(T.memref(64, T.bf16()))
            def seq(out):
                tile_dma_single_bd_task(
                    mem, DMAChannelDir.MM2S, 0, resident, sizes=[4, 16]
                )

        print(ctx.module)
except ValueError as e:
    print(f"RAISED ValueError: {e}")


# tile_dma_single_bd_task takes the same rank-5 form.
# CHECK-LABEL: CASE tile_rank5
# CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
# CHECK: aie.dma_bd(%{{.*}} : memref<64xbf16, 1> offset = 0 sizes = [2, 2, 1, 1, 16] strides = [32, 16, 0, 0, 1])
# CHECK: repeat_count = 3
print("// CASE tile_rank5")
with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu1_1col)
    def device_body():
        mem = tile(0, 1)
        resident = buffer(mem, T.memref(64, T.bf16(), memory_space=1), name="resident")

        @runtime_sequence(T.memref(64, T.bf16()))
        def seq(out):
            task = tile_dma_single_bd_task(
                mem,
                DMAChannelDir.MM2S,
                0,
                resident,
                sizes=[2, 2, 1, 1, 16],
                strides=[32, 16, 0, 0, 1],
                issue_token=True,
            )
            dma_start_task(task)
            dma_await_task(task)

    print(ctx.module)
