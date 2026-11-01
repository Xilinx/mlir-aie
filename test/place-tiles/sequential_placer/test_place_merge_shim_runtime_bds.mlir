//===- test_place_merge_shim_runtime_bds.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two shim LTOs whose DMA channels fit one shim tile, but whose runtime tasks
// do not fit its 16 buffer descriptors. Each task's descriptor strides
// 4194304 bf16 (2097152 words) on its outer dimension, past the shim's 20-bit
// iteration stride, so aie-decompose-large-dma-bd splits it into 4 BDs:
// @in_a holds 12 at once and @in_b 8. The placer keeps them on separate shims.
// With a single column there is no other shim: the merge goes ahead and
// aie-assign-runtime-sequence-bd-ids reports the overflow, as before.

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s
// RUN: not aie-opt --split-input-file --pass-pipeline='builtin.module(aie.device(aie-place-tiles,aie-objectFifo-stateful-transform,aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-decompose-large-dma-bd,aie-assign-runtime-sequence-bd-ids))' %s | FileCheck %s --check-prefix=LOWER
// RUN: not aie-opt --split-input-file --pass-pipeline='builtin.module(aie.device(aie-place-tiles,aie-objectFifo-stateful-transform,aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-decompose-large-dma-bd,aie-assign-runtime-sequence-bd-ids))' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=BDERR

// CHECK-LABEL: @split_shims
// CHECK-DAG:   %[[S0:.*]] = aie.tile(0, 0)
// CHECK-DAG:   %[[S1:.*]] = aie.tile(1, 0)
// CHECK:       aie.objectfifo @in_a(%[[S0]],
// CHECK:       aie.objectfifo @in_b(%[[S1]],
// LOWER-LABEL: @split_shims
// LOWER:       aiex.dma_configure_task(%shim_noc_tile_0_0,
// LOWER:       {bd_id = 11 : i32}
// LOWER-NOT:   {bd_id = 12 : i32}

// CHECK-LABEL: @one_column
// CHECK:       %[[S:.*]] = aie.tile(0, 0)
// CHECK:       aie.objectfifo @in_a(%[[S]],
// CHECK:       aie.objectfifo @in_b(%[[S]],
// BDERR-NOT:   error:
// BDERR:       error: 'aiex.dma_configure_task' op Too many simultaneously active buffer descriptors on tile (0,0), which supports up to 16
// BDERR-NOT:   error:

module @split_shims {
  aie.device(npu2) {
    %core = aie.logical_tile<CoreTile>(0, 2)
    %shim_a = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_b = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in_a(%shim_a, {%core}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
    aie.objectfifo @in_b(%shim_b, {%core}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
    aie.core(%core) {
      %0 = aie.objectfifo.acquire @in_a(Consume, 1) : memref<4096xbf16>
      aie.objectfifo.release @in_a(Consume, 1)
      %1 = aie.objectfifo.acquire @in_b(Consume, 1) : memref<4096xbf16>
      aie.objectfifo.release @in_b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence(%a: memref<16777216xbf16>, %b: memref<16777216xbf16>) {
      %ta0 = aiex.dma_configure_task_for @in_a {
        aie.dma_bd(%a : memref<16777216xbf16> offset = 0 len = 524288 sizes = [4, 128, 64, 64] strides = [4194304, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%ta0)
      %ta1 = aiex.dma_configure_task_for @in_a {
        aie.dma_bd(%a : memref<16777216xbf16> offset = 64 len = 524288 sizes = [4, 128, 64, 64] strides = [4194304, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%ta1)
      %ta2 = aiex.dma_configure_task_for @in_a {
        aie.dma_bd(%a : memref<16777216xbf16> offset = 128 len = 524288 sizes = [4, 128, 64, 64] strides = [4194304, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%ta2)
      %tb0 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<16777216xbf16> offset = 0 len = 524288 sizes = [4, 128, 64, 64] strides = [4194304, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%tb0)
      %tb1 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<16777216xbf16> offset = 64 len = 524288 sizes = [4, 128, 64, 64] strides = [4194304, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%tb1)
      aiex.dma_await_task(%ta0)
      aiex.dma_await_task(%ta1)
      aiex.dma_await_task(%ta2)
      aiex.dma_await_task(%tb0)
      aiex.dma_await_task(%tb1)
    }
  }
}

// -----

module @one_column {
  aie.device(npu2_1col) {
    %core = aie.logical_tile<CoreTile>(0, 2)
    %shim_a = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_b = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in_a(%shim_a, {%core}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
    aie.objectfifo @in_b(%shim_b, {%core}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
    aie.core(%core) {
      %0 = aie.objectfifo.acquire @in_a(Consume, 1) : memref<4096xbf16>
      aie.objectfifo.release @in_a(Consume, 1)
      %1 = aie.objectfifo.acquire @in_b(Consume, 1) : memref<4096xbf16>
      aie.objectfifo.release @in_b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence(%a: memref<16777216xbf16>, %b: memref<16777216xbf16>) {
      %ta0 = aiex.dma_configure_task_for @in_a {
        aie.dma_bd(%a : memref<16777216xbf16> offset = 0 len = 524288 sizes = [4, 128, 64, 64] strides = [4194304, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%ta0)
      %ta1 = aiex.dma_configure_task_for @in_a {
        aie.dma_bd(%a : memref<16777216xbf16> offset = 64 len = 524288 sizes = [4, 128, 64, 64] strides = [4194304, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%ta1)
      %ta2 = aiex.dma_configure_task_for @in_a {
        aie.dma_bd(%a : memref<16777216xbf16> offset = 128 len = 524288 sizes = [4, 128, 64, 64] strides = [4194304, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%ta2)
      %tb0 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<16777216xbf16> offset = 0 len = 524288 sizes = [4, 128, 64, 64] strides = [4194304, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%tb0)
      %tb1 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<16777216xbf16> offset = 64 len = 524288 sizes = [4, 128, 64, 64] strides = [4194304, 64, 8192, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%tb1)
      aiex.dma_await_task(%ta0)
      aiex.dma_await_task(%ta1)
      aiex.dma_await_task(%ta2)
      aiex.dma_await_task(%tb0)
      aiex.dma_await_task(%tb1)
    }
  }
}
