// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --verify-diagnostics %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A known column difference requires neighbor channels even with an unknown
// row. Fixed buffers prevent locality repair from moving the pool to the reader.
module @unknown_row {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %home = aie.logical_tile<MemTile>(1, 1)
    // expected-error @+1 {{requires at least 5 MM2S channels, but capacity is 4 for adjacent MemTile access}}
    %reader = aie.logical_tile<MemTile>(0, ?)
    %b = aie.buffer(%home) {sym_name = "b"} : memref<16xi32>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    // expected-note @+1 {{DMA endpoint @r0 requires adjacent MemTile access}}
    aie.objectfifo.dma_endpoint @r0(%reader) drains @p
    // expected-note @+1 {{DMA endpoint @r1 requires adjacent MemTile access}}
    aie.objectfifo.dma_endpoint @r1(%reader) drains @p
    // expected-note @+1 {{DMA endpoint @r2 requires adjacent MemTile access}}
    aie.objectfifo.dma_endpoint @r2(%reader) drains @p
    // expected-note @+1 {{DMA endpoint @r3 requires adjacent MemTile access}}
    aie.objectfifo.dma_endpoint @r3(%reader) drains @p
    // expected-note @+1 {{DMA endpoint @r4 requires adjacent MemTile access}}
    aie.objectfifo.dma_endpoint @r4(%reader) drains @p
  }
}

// -----

// Partially constrained columns cannot establish locality either.
module @unknown_column {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    // expected-error @+1 {{requires at least 5 S2MM channels, but capacity is 4 for adjacent MemTile access}}
    %writer = aie.logical_tile<MemTile>(?, 1)
    %b = aie.buffer(%home) {sym_name = "b"} : memref<16xi32>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    // expected-note @+1 {{DMA endpoint @w0 requires adjacent MemTile access}}
    aie.objectfifo.dma_endpoint @w0(%writer) fills @p
    // expected-note @+1 {{DMA endpoint @w1 requires adjacent MemTile access}}
    aie.objectfifo.dma_endpoint @w1(%writer) fills @p
    // expected-note @+1 {{DMA endpoint @w2 requires adjacent MemTile access}}
    aie.objectfifo.dma_endpoint @w2(%writer) fills @p
    // expected-note @+1 {{DMA endpoint @w3 requires adjacent MemTile access}}
    aie.objectfifo.dma_endpoint @w3(%writer) fills @p
    // expected-note @+1 {{DMA endpoint @w4 requires adjacent MemTile access}}
    aie.objectfifo.dma_endpoint @w4(%writer) fills @p
  }
}

// -----

// Pinned channels obey the same conservative limit on wholly unplaced tiles.
module @unknown_coordinates {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %home = aie.logical_tile<MemTile>(?, ?)
    %reader = aie.logical_tile<MemTile>(?, ?)
    %b = aie.buffer(%home) {sym_name = "b"} : memref<16xi32>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error @+1 {{pinned MM2S DMA channel 4 cannot access adjacent MemTile buffers or locks}}
    aie.objectfifo.dma_endpoint @reader(%reader) drains @p {channelIndex = 4 : i32}
  }
}

// -----

// Fully constrained aliases must not reserve the same pinned channel twice.
module @co_located_pinned {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %alias = aie.logical_tile<MemTile>(0, 1)
    aie.objectfifo.pool @p(%home) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    // expected-note @+1 {{DMA endpoint @reader; occupies channel 0}}
    aie.objectfifo.dma_endpoint @reader(%home) drains @p {channelIndex = 0 : i32}
    // expected-error @+1 {{pinned MM2S DMA channel 0 is out of range or already in use on this tile}}
    aie.objectfifo.dma_endpoint @alias_reader(%alias) drains @p {channelIndex = 0 : i32}
  }
}
