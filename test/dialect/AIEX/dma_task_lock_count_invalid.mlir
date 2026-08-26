//===- dma_task_lock_count_invalid.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-tasks-to-npu --split-input-file --verify-diagnostics %s

// In-order task: a single lock op is still rejected (0-or-2 contract).
module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %lock = aie.lock(%tile_0_1, 0) {init = 0 : i32}
    aie.runtime_sequence(%arg0: memref<4xi32>) {
      %t = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
        %c1 = arith.constant 1 : i32
        // expected-error@+2 {{BD blocks must have either 0 or 2 lock operations}}
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 4) {bd_id = 0 : i32}
        aie.use_lock(%lock, Release, %c1)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// An out-of-order S2MM task with a runtime bd_id forces the dynamic BD
// lowering path (rewriteSingleBDDynamic), which cannot yet thread
// out_of_order through to the completion lock; fail loud instead of
// silently dropping the release-only lock.
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4xi32>) {
      %bd = aiex.dma_bd_pool_pop(0, 0) : i32
      %t = aiex.dma_configure_task(%tile_0_0, S2MM, 0, <pkt_type = 0, pkt_id = 0>) bd_id %bd : i32 {
        // expected-error@+1 {{out-of-order is not yet supported for buffer descriptors with runtime-valued size, stride, offset, or bd_id}}
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%t)
    }
  }
}
