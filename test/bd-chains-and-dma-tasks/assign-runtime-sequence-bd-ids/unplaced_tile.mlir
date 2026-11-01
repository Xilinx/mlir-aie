//===- unplaced_tile.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --aie-assign-runtime-sequence-bd-ids %s

module {
  aie.device(npu2) {
    %t = aie.logical_tile<MemTile>(?, ?)
    %buf = aie.buffer(%t) : memref<64xi32>
    aie.runtime_sequence @seq() {
      // expected-error @+1 {{Cannot assign buffer descriptor ids to a DMA task whose tile is not placed; run placement first.}}
      %task = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%task)
    }
  }
}
