//===- good-if-carry-result.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s

// A task configured (and awaited) inside each branch of a runtime scf.if, then
// freed AFTER the if. The popped id must cross the if boundary, so the pass
// rebuilds the scf.if with an appended i32 result: each branch yields its own
// task and its own popped id, and the post-if free pushes the carried id.

// CHECK-LABEL: @if_carry_result
// CHECK: %[[R:.*]]:2 = scf.if %{{.*}} -> (index, i32) {
// CHECK:   %[[TID:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK:   scf.yield %{{.*}}, %[[TID]] : index, i32
// CHECK: } else {
// CHECK:   %[[EID:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK:   scf.yield %{{.*}}, %[[EID]] : index, i32
// CHECK: }
// CHECK: aiex.dma_bd_pool_push(0, 0) bd_id %[[R]]#1 : i32

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @if_carry_result(%arg0: memref<1024xi32>, %cond: i1) {
    %r = scf.if %cond -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      scf.yield %t : index
    } else {
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      scf.yield %t2 : index
    }
    aiex.dma_free_task(%r)
  }
}
