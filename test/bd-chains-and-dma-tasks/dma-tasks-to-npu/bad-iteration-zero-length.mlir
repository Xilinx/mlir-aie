//===- bad-iteration-zero-length.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s

// An iterated BD with no length and no length parameter moves nothing.

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<8x512xbf16>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error @+1 {{Transfer size of 0 bytes falls below minimum hardware transfer unit of 4 bytes}}
        aie.dma_bd(%arg0 : memref<8x512xbf16> offset = 0 len = 0) {bd_id = 3 : i32, iteration = #aie.bd_iteration<size = 8, stride = 512, current = 0>}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%t)
    }
  }
}
