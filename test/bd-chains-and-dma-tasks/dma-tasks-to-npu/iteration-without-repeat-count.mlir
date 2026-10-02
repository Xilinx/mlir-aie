//===- iteration-without-repeat-count.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-dma-tasks-to-npu %s

// Each push onto the task queue runs one iteration of a BD's outermost
// dimension, so an iteration dimension without a repeat_count to push the rest
// silently moves only the first.

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4096xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-warning@+1 {{iteration dimension of size 4 is pushed with repeat_count 0, so only its first iteration runs; set repeat_count = 3}}
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 64 sizes = [4, 1, 4, 16] strides = [64, 0, 16, 1]) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A zero stride repeats the same data, but still only through repeat_count.

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4096xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-warning@+1 {{iteration dimension of size 2 is pushed with repeat_count 0}}
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 64 sizes = [2, 1, 4, 16] strides = [0, 0, 16, 1]) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// With a repeat_count, constant or runtime, there is nothing to warn about.

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4096xi32>, %r: i32) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 64 sizes = [4, 1, 4, 16] strides = [64, 0, 16, 1]) {bd_id = 0 : i32}
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%t)
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 1) repeat %r : i32 {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 64 sizes = [4, 1, 4, 16] strides = [64, 0, 16, 1]) {bd_id = 1 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
    }
  }
}

// -----

// A start's own repeat_count replaces the task's, in either direction.

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4096xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 64 sizes = [4, 1, 4, 16] strides = [64, 0, 16, 1]) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t) {repeat_count = 3 : i32}
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        // expected-warning@+1 {{iteration dimension of size 4 is pushed with repeat_count 0}}
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 64 sizes = [4, 1, 4, 16] strides = [64, 0, 16, 1]) {bd_id = 1 : i32}
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%t2) {repeat_count = 0 : i32}
    }
  }
}
