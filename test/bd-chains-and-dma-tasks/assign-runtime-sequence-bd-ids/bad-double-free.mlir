//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-assign-runtime-sequence-bd-ids --split-input-file %s

// Freeing a task twice is a user error: the second free targets a BD id that is
// no longer in use.

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @double_free(%arg0: memref<8xi16>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = [])
      aie.end
    }
    aiex.dma_start_task(%t)
    aiex.dma_free_task(%t)
    // expected-error@+1 {{is not currently in use}}
    aiex.dma_free_task(%t)
  }
}



// -----

// Awaiting a task and THEN freeing it is the common "wait, then release" idiom,
// not a double free: the await returns the ids, and the redundant free is
// tolerated. This must NOT error (no expected-error -> --verify-diagnostics
// passes only if the pass stays silent here).

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @await_then_free_ok(%arg0: memref<8xi16>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = [])
      aie.end
    }
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
    aiex.dma_free_task(%t)
  }
}
