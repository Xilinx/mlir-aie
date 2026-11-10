//===- dma_start_channel_range.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A DMA program start names one of its tile's channels; allocation and the
// placers count it there, so an index past the tile's channels is refused.

module {
  aie.device(npu2) {
    %mt = aie.logical_tile<MemTile>(?, ?)
    %b = aie.buffer(%mt) : memref<16xi32>
    %md = aie.memtile_dma(%mt) {
      // expected-error @+1 {{'aie.dma_start' op S2MM channel 6 is out of range: the tile has 6 S2MM channels}}
      %s = aie.dma_start(S2MM, 6, ^bd, ^end)
    ^bd:
      aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd
    ^end:
      aie.end
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(1, 0)
    %md = aie.shim_dma(%t) {
      // expected-error @+1 {{'aie.dma_start' op MM2S channel 2 is out of range: the tile has 2 MM2S channels}}
      %s = aie.dma_start(MM2S, 2, ^end, ^end)
    ^end:
      aie.end
    }
  }
}

// -----

// The last channel of each is accepted.
module {
  aie.device(npu2) {
    %mt = aie.logical_tile<MemTile>(?, ?)
    %b = aie.buffer(%mt) : memref<16xi32>
    %md = aie.memtile_dma(%mt) {
      %s = aie.dma_start(S2MM, 5, ^bd, ^end)
    ^bd:
      aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd
    ^end:
      aie.end
    }
  }
}
