//===- good-for-identity-reuse.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s

// One task reused across every iteration: the loop yields its own task iter_arg
// (identity carry) rather than a fresh configure. The id must carry in lockstep,
// so the loop still gains an i32 iter_arg, and the yield returns that same id
// back (an identity yield). Exactly one pop before the loop and one push after;
// the per-iteration await is a pure TCT sync and adds no push, so the single id
// is returned exactly once.

// CHECK-LABEL: @for_identity_reuse
// CHECK: %[[ID:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK: %[[T:.*]] = aiex.dma_configure_task(%{{.*}}, MM2S, 0) bd_id %[[ID]] : i32
// CHECK: %[[LOOP:.*]]:2 = scf.for {{.*}} iter_args(%[[TK:.*]] = %[[T]], %[[PID:.*]] = %[[ID]]) -> (index, i32)
// CHECK:   aiex.dma_start_task(%[[TK]])
// CHECK:   aiex.dma_await_task(%[[TK]])
// CHECK-NOT: aiex.dma_bd_pool_push
// CHECK:   scf.yield %[[TK]], %[[PID]] : index, i32
// CHECK: }
// CHECK: aiex.dma_bd_pool_push(0, 0) bd_id %[[LOOP]]#1 : i32

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @for_identity_reuse(%arg0: memref<1024xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    %last = scf.for %i = %c0 to %n step %c1 iter_args(%tk = %init) -> (index) {
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
      scf.yield %tk : index
    }
    aiex.dma_free_task(%last)
  }
}
