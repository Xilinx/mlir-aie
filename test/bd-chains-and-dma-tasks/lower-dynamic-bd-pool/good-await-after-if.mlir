//===- good-await-after-if.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s

// A task configured in an scf.if branch and awaited AFTER the if. No configure
// dominates the await (each is branch-local), so the pass cannot redirect the
// await to a dominating configure. Instead it records the physical channel on
// the await as sync_* attributes and drops the task operand: the sync only ever
// needed the channel, and BD reuse is serialized by queue backpressure. Both
// branches must agree on the channel (checked); the id is freed once after.

// CHECK-LABEL: @await_after_if
// CHECK: %[[IF:.*]]:2 = scf.if %{{.*}} -> (index, i32) {
// CHECK: aiex.dma_await_task() {sync_channel = 0 : i32, sync_col = 0 : i32, sync_direction = 1 : i32, sync_row = 0 : i32}
// CHECK: aiex.dma_bd_pool_push(0, 0) bd_id %[[IF]]#1 : i32

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @await_after_if(%arg0: memref<1024xi32>, %cond: i1) {
    %r = scf.if %cond -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      scf.yield %t : index
    } else {
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      scf.yield %t2 : index
    }
    aiex.dma_await_task(%r)
    aiex.dma_free_task(%r)
  }
}
