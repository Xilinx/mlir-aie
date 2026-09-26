//===- bad_dma_start_task.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A per-start repeat_count may exceed the queue push's field (the BD-ID pass
// splits it), but it cannot be negative.
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<256xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      // expected-error@+1 {{repeat_count must be non-negative, got -1}}
      aiex.dma_start_task(%t) {repeat_count = -1 : i32}
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<256xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%t) {repeat_count = 4096 : i32, no_token}
    }
  }
}
