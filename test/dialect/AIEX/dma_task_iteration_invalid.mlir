//===- dma_task_iteration_invalid.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// The iteration attribute takes the iteration register, so a mem tile BD that
// already fills its four dimensions has no room for it.

module {
  aie.device(npu1_1col) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) : memref<256xi32>
    aie.runtime_sequence(%arg0: memref<256xi32>) {
      %t = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        // expected-error @+1 {{Cannot give more than 3 dimensions for step sizes and wraps alongside the iteration attribute on this tile (got 4 dimensions).}}
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 64 sizes = [2, 2, 2, 8] strides = [1, 16, 64, 128]) {iteration = #aie.bd_iteration<size = 2, stride = 8, current = 0>}
        aie.end
      }
    }
  }
}

// -----

// A task BD's iteration is held to its tile's iteration field.

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<256xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error @+1 {{BD iteration size must be in [1, 64]}}
        aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 65, stride = 1, current = 0>}
        aie.end
      }
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<256xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error @+1 {{BD iteration current must be in [0, size)}}
        aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 4, stride = 64, current = 4>}
        aie.end
      }
    }
  }
}
