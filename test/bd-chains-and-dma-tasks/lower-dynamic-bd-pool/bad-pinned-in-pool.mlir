//===- bad-pinned-in-pool.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --verify-diagnostics %s

// Within a dynamic (pool) sequence, allocation is all-or-nothing: the pool
// hands out ids from the full 0..N-1 range at runtime, so a hand-pinned bd_id
// could collide with a runtime-allocated one. A pinned id here is an error --
// allocate every BD from the pool, or make the sequence straight-line.

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @dyn_pin(%arg0: memref<1024xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      // expected-error@+1 {{pins a buffer descriptor ID inside a runtime-bound sequence that draws IDs from the dynamic pool}}
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) {bd_id = 3 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_await_task(%last)
    aiex.dma_free_task(%last)
  }
}
