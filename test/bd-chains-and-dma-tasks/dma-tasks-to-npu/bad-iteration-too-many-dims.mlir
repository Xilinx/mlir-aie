//===- bad-iteration-too-many-dims.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #aie.bd_iteration claims the same hardware slot a hoisted 4th ND dimension
// would otherwise use (see good-iteration-attribute.mlir for the accepted
// 3-dimension + iteration combination), so a full 4-dimensional access
// pattern leaves no room for it.

// RUN: aie-opt --verify-diagnostics %s

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<8192xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error @+1 {{Cannot give more than 3 dimensions for step sizes and wraps on this tile when the iteration attribute is also set (got 4 dimensions)}}
        aie.dma_bd(%arg0 : memref<8192xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1]) {bd_id = 5 : i32, iteration = #aie.bd_iteration<size = 4, stride = 16, current = 2>}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
    }
  }
}
