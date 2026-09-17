// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --verify-diagnostics %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A pool resident at home is still remote to an endpoint on another tile.
module @pinned_remote {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %remote = aie.tile(1, 1)
    %b = aie.buffer(%home) {sym_name = "b"} : memref<16xi32>
    aie.objectfifo.pool @p(%home) {
      depth = 1 : i32, buffers = [@b]
    } : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error @+1 {{pinned MM2S DMA channel 4 cannot access adjacent MemTile buffers}}
    aie.objectfifo.dma_endpoint @reader(%remote) drains @p {channelIndex = 4 : i32}
  }
}

// -----

// Just one remote object restricts the channel for the whole buffer chain.
module @pinned_split {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %left = aie.tile(0, 1)
    %right = aie.tile(1, 1)
    %a = aie.buffer(%left) {sym_name = "a"} : memref<16xi32>
    %b = aie.buffer(%right) {sym_name = "b"} : memref<16xi32>
    aie.objectfifo.pool @p(%left) {
      depth = 2 : i32, buffers = [@a, @b]
    } : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error @+1 {{pinned S2MM DMA channel 5 cannot access adjacent MemTile buffers}}
    aie.objectfifo.dma_endpoint @writer(%left) fills @p {channelIndex = 5 : i32}
  }
}

// -----

// Neither neighbor of the full middle tile is reachable by both endpoints.
module @no_common_memory {
  aie.device(npu2) {
    %left = aie.tile(0, 1)
    %middle = aie.tile(1, 1)
    %right = aie.tile(2, 1)
    %reserved = aie.buffer(%middle) : memref<524288xi8>
    // expected-error @+1 {{could not place buffers in accessible memory with available capacity}}
    aie.objectfifo.pool @p(%middle) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%left) drains @p
    aie.objectfifo.dma_endpoint @writer(%right) fills @p
  }
}
