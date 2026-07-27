//===- good-await-after-if.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s

// A task is configured and started in each branch of a runtime scf.if, then
// awaited and freed AFTER the if. Both branches start a task on the same
// physical channel, so the transfer is always in flight on the taken path --
// awaiting after the if is deadlock-free. The scf.if result %[[R]]#0 is the phi
// of the task in flight (whichever branch ran): the await keeps that SSA operand
// (no redirect, no attributes), preserving the sync's data dependence on the
// task. The id carried out at %[[R]]#1 is freed once after the if. The npu_sync
// lowering later walks the phi to a configure for the physical channel (both
// branches were verified to agree).

// CHECK-LABEL: @await_after_if
// CHECK: %[[R:.*]]:2 = scf.if %{{.*}} -> (index, i32) {
// CHECK:   %[[TID:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK:   scf.yield %{{.*}}, %[[TID]] : index, i32
// CHECK: } else {
// CHECK:   %[[EID:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK:   scf.yield %{{.*}}, %[[EID]] : index, i32
// CHECK: }
// CHECK: aiex.dma_await_task(%[[R]]#0)
// CHECK: aiex.dma_bd_pool_push(0, 0) bd_id %[[R]]#1 : i32

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
