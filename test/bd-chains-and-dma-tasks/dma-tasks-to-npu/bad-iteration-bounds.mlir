//===- bad-iteration-bounds.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The aie.dma_bd verifier is skipped for runtime-sequence BDs, so
// verifyTaskBDDimensions enforces the #aie.bd_iteration bounds: size in [1, 64]
// and current in [0, size). With both checked, current > 63 is unreachable.

// RUN: aie-opt --verify-diagnostics --split-input-file %s

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error @+1 {{BD iteration size must be in [1, 64]}}
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64) {bd_id = 5 : i32, iteration = #aie.bd_iteration<size = 65, stride = 16, current = 0>}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
    }
  }
}

// -----

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error @+1 {{BD iteration current must be in [0, size)}}
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64) {bd_id = 5 : i32, iteration = #aie.bd_iteration<size = 4, stride = 16, current = 4>}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
    }
  }
}
