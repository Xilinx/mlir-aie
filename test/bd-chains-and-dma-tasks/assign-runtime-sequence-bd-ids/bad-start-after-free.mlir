//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-assign-runtime-sequence-bd-ids --split-input-file %s

// A free returns a task's BD ids to the pool, so a later start would push ids
// that the next configure may already have rewritten.

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @start_after_free(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t)
    aiex.dma_free_task(%t)
    %u = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
      aie.dma_bd(%arg1 : memref<8xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%u)
    // expected-error@+1 {{starts a task whose buffer descriptors an earlier aiex.dma_free_task released}}
    aiex.dma_start_task(%t)
  }
}

// -----

// The same restart is rejected when nothing is allocated in between: the ids
// could be handed out again while the restart is still in flight.

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @restart_after_free(%arg0: memref<8xi32>) {
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
    aiex.dma_free_task(%t)
    // expected-error@+1 {{starts a task whose buffer descriptors an earlier aiex.dma_free_task released}}
    aiex.dma_start_task(%t)
  }
}

// -----

// Restarting before the free is fine: the ids stay owned until the last start.

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @restart_then_free_ok(%arg0: memref<8xi32>) {
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi32> offset = 0 len = 8)
      aie.end
    }
    aiex.dma_start_task(%t)
    aiex.dma_start_task(%t)
    aiex.dma_free_task(%t)
  }
}
