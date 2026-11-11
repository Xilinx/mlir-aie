//===- bad_npu_dma_memcpy_nd_bd_id.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// A copy programs one of its tile's buffer descriptors, so its id must name
// one, static or runtime-sized.
module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.runtime_sequence(%pkts : memref<?xi32>) {
      // expected-error@+1 {{programs buffer descriptor ID 20, but tile (0, 0) has 16 buffer descriptors}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 20 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
    }
  }
}

// -----

module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.runtime_sequence(%pkts : memref<?xi32>, %n : i64) {
      // expected-error@+1 {{programs buffer descriptor ID 16, but tile (0, 0) has 16 buffer descriptors}}
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, %n][0, 0, 0, 1]) {id = 16 : i64, metadata = @ctrl} : memref<?xi32>
    }
  }
}

// -----

// The last one is fine.
module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%shim, MM2S, 0)
    aie.runtime_sequence(%pkts : memref<?xi32>) {
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 15 : i64, metadata = @ctrl} : memref<?xi32>
    }
  }
}
