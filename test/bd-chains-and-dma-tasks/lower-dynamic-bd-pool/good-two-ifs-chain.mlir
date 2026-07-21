//===- good-two-ifs-chain.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s

// A task configured in the first scf.if (both branches), then carried through a
// SECOND scf.if on the same condition, then freed. Both ifs grow a parallel i32
// result; the second if just re-yields the first's carried id on each branch.
// This is the reviewer's fragility question: a later if/else that re-yields the
// task on the same condition. It threads correctly -- one pop at runtime (one
// branch of the first if), one push after the second.

// CHECK-LABEL: @two_ifs_chain
// CHECK: %[[IF1:.*]]:2 = scf.if %{{.*}} -> (index, i32) {
// CHECK:   %[[ID1:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK:   scf.yield %{{.*}}, %[[ID1]] : index, i32
// CHECK: } else {
// CHECK:   %[[ID2:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK:   scf.yield %{{.*}}, %[[ID2]] : index, i32
// CHECK: }
// CHECK: %[[IF2:.*]]:2 = scf.if %{{.*}} -> (index, i32) {
// CHECK:   scf.yield %[[IF1]]#0, %[[IF1]]#1 : index, i32
// CHECK: } else {
// CHECK:   scf.yield %[[IF1]]#0, %[[IF1]]#1 : index, i32
// CHECK: }
// CHECK: aiex.dma_bd_pool_push(0, 0) bd_id %[[IF2]]#1 : i32

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @two_ifs_chain(%arg0: memref<1024xi32>, %cond: i1) {
    %r1 = scf.if %cond -> (index) {
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
    %r2 = scf.if %cond -> (index) {
      scf.yield %r1 : index
    } else {
      scf.yield %r1 : index
    }
    aiex.dma_free_task(%r2)
  }
}
