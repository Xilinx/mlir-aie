//===- good-taskless-await-untouched.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s

// An await may already carry its physical channel in sync_* attributes with no
// task operand (e.g. lowered here on a prior run, or written that way). Such an
// await owns no pool id to return and needs no configure resolution, so the
// pass must leave it untouched rather than dereferencing the absent operand.
// The surrounding runtime control flow still lowers normally.

// CHECK-LABEL: @taskless
// CHECK: aiex.dma_bd_pool_pop(0, 0)
// CHECK: aiex.dma_await_task() {sync_channel = 0 : i32, sync_col = 0 : i32, sync_direction = 1 : i32, sync_row = 0 : i32}

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @taskless(%arg0: memref<1024xi32>, %cond: i1) {
    scf.if %cond {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
    }
    aiex.dma_await_task() {sync_channel = 0 : i32, sync_col = 0 : i32, sync_direction = 1 : i32, sync_row = 0 : i32}
  }
}
