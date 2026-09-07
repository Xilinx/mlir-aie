//===- bad-for-cross-channel.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --verify-diagnostics %s

// The loop is initialized with a task on one tile, but every iteration
// reconfigures onto a DIFFERENT tile and yields that instead of the original
// iter_arg. A single carried id (and the push after the loop, which only
// tracks the loop-invariant init's tile) belongs to one physical channel, so
// the init and the per-iteration reconfiguration must agree. They do not
// here, so this is diagnosed rather than mislowered to a push/sync on the
// wrong pool.

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_1_0 = aie.tile(1, 0)
  aie.runtime_sequence @for_cross_channel(%arg0: memref<1024xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    // expected-error@+1 {{carries a task on different physical channels}}
    %last = scf.for %i = %c0 to %n step %c1 iter_args(%tk = %init) -> (index) {
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
      %t2 = aiex.dma_configure_task(%tile_1_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      scf.yield %t2 : index
    }
    aiex.dma_free_task(%last)
  }
}
