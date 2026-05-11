//===- canonicalize_no_linearize_iter_bd.mlir -------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Regression for LinearizeContiguousBDTransfer; see PR #3036.

// RUN: aie-opt --canonicalize --split-input-file %s | FileCheck %s --check-prefix=CANON
// RUN: aie-opt --pass-pipeline='any(aie.device(canonicalize,aie-dma-tasks-to-npu))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=LOWER

// -----

// len < product: outer dim is an iteration dim, must NOT linearize.
// CANON-LABEL: @iter_bd_no_linearize
// CANON:         aie.dma_bd(%arg0 : memref<131072xbf16>, 0, 4096,
// CANON-SAME:        [<size = 32, stride = 4096>, <size = 1, stride = 64>, <size = 64, stride = 64>, <size = 64, stride = 1>]
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
        aie.dma_bd(%arg0 : memref<131072xbf16>, 0, 4096,
          [<size = 32, stride = 4096>, <size = 1, stride = 64>,
           <size = 64, stride = 64>, <size = 64, stride = 1>]) {bd_id = 0 : i32}
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
// CANON:         aie.dma_bd(%arg0 : memref<131072xbf16>, 0, 131072)
// CANON-NOT:         [<
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_4d_match (%tile_0_0, MM2S, 0)
    aie.runtime_sequence @iter_bd_4d_len_matches_linearizes(%arg0 : memref<131072xbf16>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<131072xbf16>, 0, 131072,
          [<size = 32, stride = 4096>, <size = 64, stride = 64>,
           <size = 64, stride = 1>, <size = 1, stride = 1>]) {bd_id = 0 : i32}
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
// CANON:         aie.dma_bd(%arg0 : memref<4096xbf16>, 0, 4096)
// CANON-NOT:         [<
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_2d_match (%tile_0_0, MM2S, 0)
    aie.runtime_sequence @iter_bd_2d_len_matches_linearizes(%arg0 : memref<4096xbf16>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4096xbf16>, 0, 4096,
          [<size = 64, stride = 64>, <size = 64, stride = 1>]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
