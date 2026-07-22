//===- bad-if-cross-channel.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --verify-diagnostics %s

// Each branch of the scf.if configures a task on a DIFFERENT tile, then yields
// it out as one carried result. A single carried id (and, if awaited after the
// if, a single completion sync) belongs to one physical channel, so the two
// branches must agree on tile/direction/channel. They do not here, so this is
// diagnosed rather than mislowered to a push/sync on the wrong pool.

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_1_0 = aie.tile(1, 0)
  aie.runtime_sequence @if_cross_channel(%arg0: memref<1024xi32>, %cond: i1) {
    // expected-error@+1 {{yields tasks on different physical channels}}
    %r = scf.if %cond -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      scf.yield %t : index
    } else {
      %t2 = aiex.dma_configure_task(%tile_1_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      scf.yield %t2 : index
    }
    aiex.dma_free_task(%r)
  }
}
