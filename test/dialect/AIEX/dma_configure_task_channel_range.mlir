//===- dma_configure_task_channel_range.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A runtime task configures one of its tile's channels by index; an index past
// the tile's channels is refused, logical tile or placed.

module {
  aie.device(npu2) {
    %t = aie.tile(2, 0)
    aie.runtime_sequence(%a: memref<16xi32>) {
      // expected-error @+1 {{'aiex.dma_configure_task' op S2MM channel 3 is out of range: the tile has 2 S2MM channels}}
      %task = aiex.dma_configure_task(%t, S2MM, 3) {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %mt = aie.logical_tile<MemTile>(?, ?)
    %b = aie.buffer(%mt) : memref<16xi32>
    aie.runtime_sequence() {
      // expected-error @+1 {{'aiex.dma_configure_task' op MM2S channel 6 is out of range: the tile has 6 MM2S channels}}
      %task = aiex.dma_configure_task(%mt, MM2S, 6) {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
    }
  }
}

// -----

// The last channel is accepted.
module {
  aie.device(npu2) {
    %t = aie.tile(2, 0)
    aie.runtime_sequence(%a: memref<16xi32>) {
      %task = aiex.dma_configure_task(%t, S2MM, 1) {
        aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
    }
  }
}
