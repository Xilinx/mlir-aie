//===- bad-if-divergent-yield.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --verify-diagnostics %s

// A task is configured in the then branch and yielded out, but the else branch
// yields a non-pool value. There is no popped id to push on the else path, so
// the task cannot be freed after the if -- both branches must yield the task.

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @if_divergent(%arg0: memref<1024xi32>, %cond: i1,
                                     %other: index) {
    // expected-error@+1 {{yields a pool-allocated task on only one branch of an scf.if}}
    %r = scf.if %cond -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      scf.yield %t : index
    } else {
      scf.yield %other : index
    }
    aiex.dma_free_task(%r)
  }
}
