//===- canonicalize_no_linearize_iter_bd.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression for LinearizeContiguousBDTransfer; see PR #3036.

// RUN: aie-opt --canonicalize --split-input-file %s | FileCheck %s --check-prefix=CANON
// RUN: aie-opt --pass-pipeline='any(aie.device(canonicalize,aie-dma-tasks-to-npu))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=LOWER

// -----

// len < product: outer dim is an iteration dim, must NOT linearize.
// CANON-LABEL: @iter_bd_no_linearize
// CANON:         aie.dma_bd(%arg0 : memref<131072xbf16> offset = 0 len = 4096
// CANON-SAME:        sizes = [32, 1, 64, 64] strides = [4096, 64, 64, 1]
// LOWER-LABEL: @iter_bd_no_linearize
// LOWER:         aiex.npu.writebd
// LOWER-SAME:      buffer_length = 2048
// LOWER-SAME:      iteration_size = 31
// LOWER-SAME:      iteration_stride = 2047
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_iter (%tile_0_0, MM2S, 0)
    aie.runtime_sequence @iter_bd_no_linearize(%arg0 : memref<131072xbf16>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<131072xbf16> offset = 0 len = 4096 sizes = [32, 1, 64, 64] strides = [4096, 64, 64, 1]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// 4D, len == product: still linearizes.
// CANON-LABEL: @iter_bd_4d_len_matches_linearizes
// CANON:         aie.dma_bd(%arg0 : memref<131072xbf16> offset = 0 len = 131072 sizes = [] strides = [])
// CANON-NOT:         [<
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_4d_match (%tile_0_0, MM2S, 0)
    aie.runtime_sequence @iter_bd_4d_len_matches_linearizes(%arg0 : memref<131072xbf16>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<131072xbf16> offset = 0 len = 131072 sizes = [32, 64, 64, 1] strides = [4096, 64, 1, 1]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// 2D, len == product: still linearizes.
// CANON-LABEL: @iter_bd_2d_len_matches_linearizes
// CANON:         aie.dma_bd(%arg0 : memref<4096xbf16> offset = 0 len = 4096 sizes = [] strides = [])
// CANON-NOT:         [<
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_2d_match (%tile_0_0, MM2S, 0)
    aie.runtime_sequence @iter_bd_2d_len_matches_linearizes(%arg0 : memref<4096xbf16>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4096xbf16> offset = 0 len = 4096 sizes = [64, 64] strides = [64, 1]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
