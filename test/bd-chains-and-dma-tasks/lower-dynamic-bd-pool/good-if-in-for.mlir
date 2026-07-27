//===- good-if-in-for.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s

// A self-contained runtime scf.if nested inside a runtime scf.for. The combined
// carry walk is innermost-first, so the inner if is processed before the outer
// for. The if is self-contained (pop+push inside the branch), so it needs no
// carried result, and the for carries nothing either.

// CHECK-LABEL: @if_in_for
// CHECK: scf.for
// CHECK:   scf.if %{{.*}} {
// CHECK:     %[[ID:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK:     aiex.dma_configure_task(%{{.*}}, MM2S, 0) bd_id %[[ID]] : i32
// CHECK:     aiex.dma_bd_pool_push(0, 0) bd_id %[[ID]] : i32

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @if_in_for(%arg0: memref<1024xi32>, %n: index,
                                  %cond: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
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
}
