//===- bad_dma_channel.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The region form of a DMA program and a shim DMA allocation name one of the
// tile's DMA channels, and a shim PL tile has no DMA. Without these checks the
// channel's registers are computed past the end of its register block, e.g.
// memtile S2MM 6 programs the MM2S 0 control and queue registers.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu1) {
  %t = aie.tile(0, 1)
  %b = aie.buffer(%t) : memref<16xi32>
  %l0 = aie.lock(%t, 0) {init = 1 : i32}
  %l1 = aie.lock(%t, 1) {init = 0 : i32}
  aie.memtile_dma(%t) {
    // expected-error@+1 {{'aie.dma' op S2MM channel 6 is out of range: the tile has 6 S2MM channels}}
    %0 = aie.dma(S2MM, 6) [{
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b : memref<16xi32>)
      aie.use_lock(%l1, Release, %c1)
    }]
    aie.end
  }
}

// -----

aie.device(npu1) {
  %t = aie.tile(0, 0)
  // expected-error@+1 {{'aie.shim_dma_allocation' op MM2S channel 2 is out of range: the tile has 2 MM2S channels}}
  aie.shim_dma_allocation @a(%t, MM2S, 2)
}

// -----

aie.device(npu1) {
  %t = aie.tile(0, 0)
  // expected-error@+1 {{'aie.shim_dma_allocation' op S2MM channel -1 is out of range: the tile has 2 S2MM channels}}
  aie.shim_dma_allocation @a(%t, S2MM, -1)
}

// -----

// The allocation index is 64-bit; a value that truncates to 0 is rejected.
aie.device(npu1) {
  %t = aie.tile(0, 0)
  // expected-error@+1 {{'aie.shim_dma_allocation' op MM2S channel 4294967296 is out of range: the tile has 2 MM2S channels}}
  aie.shim_dma_allocation @a(%t, MM2S, 4294967296)
}

// -----

// xcve2802 column 0 is a shim PL tile.
aie.device(xcve2802) {
  %t = aie.tile(0, 0)
  // expected-error@+1 {{'aie.shim_dma_allocation' op tile (0, 0) is a shim PL tile and has no DMA}}
  aie.shim_dma_allocation @a(%t, MM2S, 0)
}

// -----

// A logical shim PL tile answers by its type.
aie.device(xcve2802) {
  %t = aie.logical_tile<ShimPLTile>(?, ?)
  // expected-error@+1 {{'aie.shim_dma_allocation' op the tile is a shim PL tile and has no DMA}}
  aie.shim_dma_allocation @a(%t, MM2S, 0)
}
