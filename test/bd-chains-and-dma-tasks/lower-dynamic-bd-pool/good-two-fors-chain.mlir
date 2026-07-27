//===- good-two-fors-chain.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s

// One task threaded through TWO sibling loops: the first loop's result feeds the
// second loop's init. The id must carry across the loop boundary, so the second
// loop's id iter_arg is initialized from the first loop's id result. A single
// pop up front and a single push after the second loop -- the id survives both
// loops. This is the sibling producer->consumer case the old reverse-of-preorder
// walk mis-ordered (consumer grown before producer).

// CHECK-LABEL: @two_fors_chain
// CHECK: %[[ID:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK: %[[L1:.*]]:2 = scf.for {{.*}} iter_args({{.*}}, %{{.*}} = %[[ID]]) -> (index, i32)
// CHECK: %[[L2:.*]]:2 = scf.for {{.*}} iter_args(%{{.*}} = %[[L1]]#0, %{{.*}} = %[[L1]]#1) -> (index, i32)
// CHECK: aiex.dma_bd_pool_push(0, 0) bd_id %[[L2]]#1 : i32

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @two_fors_chain(%arg0: memref<1024xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    %r1 = scf.for %i = %c0 to %n step %c1 iter_args(%tk = %init) -> (index) {
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
      scf.yield %tk : index
    }
    %r2 = scf.for %i = %c0 to %n step %c1 iter_args(%tk = %r1) -> (index) {
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
      scf.yield %tk : index
    }
    aiex.dma_free_task(%r2)
  }
}
