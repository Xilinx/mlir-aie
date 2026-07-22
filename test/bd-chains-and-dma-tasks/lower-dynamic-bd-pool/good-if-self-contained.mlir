//===- good-if-self-contained.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s

// A runtime scf.if whose task is configured, started, awaited and freed all
// inside the same branch. The pool pop/push both live in the branch and the
// task never crosses the if boundary, so the if gains no results.

// CHECK-LABEL: @if_self_contained
// CHECK: scf.if %{{.*}} {
// CHECK:   %[[ID:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK:   aiex.dma_configure_task(%{{.*}}, MM2S, 0) bd_id %[[ID]] : i32
// CHECK:   aiex.dma_bd_pool_push(0, 0) bd_id %[[ID]] : i32
// CHECK-NOT: scf.yield %{{.*}} : index, i32

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @if_self_contained(%arg0: memref<1024xi32>, %cond: i1) {
    scf.if %cond {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}
