//===- test_place_shim_runtime_bd_copies.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// @in_a's four dma_memcpy_nd ops pin ids 0-3 and @in_b's 13 tasks hold 13
// ids at once: 17 on one shim, which has 16. Both placers count a copy's id
// as the BD-id pass does and keep the two shims apart, so the runtime
// sequence's ids fit.

// RUN: aie-opt --pass-pipeline='builtin.module(aie-place-tiles,aie.device(aie-objectFifo-stateful-transform,aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-decompose-large-dma-bd,aie-assign-runtime-sequence-bd-ids))' %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='builtin.module(aie-place-tiles{placer=sa_placer sa-seed=1},aie.device(aie-objectFifo-stateful-transform,aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-decompose-large-dma-bd,aie-assign-runtime-sequence-bd-ids))' %s | FileCheck %s

// CHECK-LABEL: @copies
// CHECK:       aie.shim_dma_allocation @in_a__shim_alloc(%[[A:shim_noc_tile_[0-9]+_0]], MM2S, 0)
// CHECK-NOT:   @in_b__shim_alloc(%[[A]],
// CHECK:       aie.shim_dma_allocation @in_b__shim_alloc(%shim_noc_tile_{{[0-9]+}}_0, MM2S, 0)
module @copies {
  aie.device(npu2) {
    %core = aie.logical_tile<CoreTile>(0, 2)
    %shim_a = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_b = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in_a(%shim_a, {%core}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @in_b(%shim_b, {%core}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.core(%core) {
      %0 = aie.objectfifo.acquire @in_a(Consume, 1) : memref<64xi32>
      aie.objectfifo.release @in_a(Consume, 1)
      %1 = aie.objectfifo.acquire @in_b(Consume, 1) : memref<64xi32>
      aie.objectfifo.release @in_b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence(%a: memref<4096xi32>, %b: memref<4096xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @in_a, issue_token = true} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 64][1, 1, 1, 64][0, 0, 0, 1]) {id = 1 : i64, metadata = @in_a, issue_token = true} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 128][1, 1, 1, 64][0, 0, 0, 1]) {id = 2 : i64, metadata = @in_a, issue_token = true} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 192][1, 1, 1, 64][0, 0, 0, 1]) {id = 3 : i64, metadata = @in_a, issue_token = true} : memref<4096xi32>
      %tb0 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<4096xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      %tb1 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<4096xi32> offset = 64 len = 64)
        aie.end
      } {issue_token = true}
      %tb2 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<4096xi32> offset = 128 len = 64)
        aie.end
      } {issue_token = true}
      %tb3 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<4096xi32> offset = 192 len = 64)
        aie.end
      } {issue_token = true}
      %tb4 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<4096xi32> offset = 256 len = 64)
        aie.end
      } {issue_token = true}
      %tb5 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<4096xi32> offset = 320 len = 64)
        aie.end
      } {issue_token = true}
      %tb6 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<4096xi32> offset = 384 len = 64)
        aie.end
      } {issue_token = true}
      %tb7 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<4096xi32> offset = 448 len = 64)
        aie.end
      } {issue_token = true}
      %tb8 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<4096xi32> offset = 512 len = 64)
        aie.end
      } {issue_token = true}
      %tb9 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<4096xi32> offset = 576 len = 64)
        aie.end
      } {issue_token = true}
      %tb10 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<4096xi32> offset = 640 len = 64)
        aie.end
      } {issue_token = true}
      %tb11 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<4096xi32> offset = 704 len = 64)
        aie.end
      } {issue_token = true}
      %tb12 = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<4096xi32> offset = 768 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tb0)
      aiex.dma_await_task(%tb0)
      aiex.dma_free_task(%tb1)
      aiex.dma_free_task(%tb2)
      aiex.dma_free_task(%tb3)
      aiex.dma_free_task(%tb4)
      aiex.dma_free_task(%tb5)
      aiex.dma_free_task(%tb6)
      aiex.dma_free_task(%tb7)
      aiex.dma_free_task(%tb8)
      aiex.dma_free_task(%tb9)
      aiex.dma_free_task(%tb10)
      aiex.dma_free_task(%tb11)
      aiex.dma_free_task(%tb12)
      aiex.npu.dma_wait {symbol = @in_a}
      aiex.npu.dma_wait {symbol = @in_a}
      aiex.npu.dma_wait {symbol = @in_a}
      aiex.npu.dma_wait {symbol = @in_a}
    }
  }
}
