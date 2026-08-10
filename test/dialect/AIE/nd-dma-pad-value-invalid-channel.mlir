//===- nd-dma-pad-value-invalid-channel.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

// The constant pad value is a MemTile MM2S channel register, so a nonzero
// pad_value must be rejected on any channel that cannot carry it: a non-memtile
// (core-tile) channel, or an S2MM channel (the register is MM2S-only).

module {
  // Terminator form: aie.dma_start on a core tile.
  aie.device(npu2) {
    %t = aie.tile(1, 3)
    %buf = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
      // expected-error@+1 {{'aie.dma_start' op pad_value is only supported on memtile DMA channels}}
      aie.dma_start("MM2S", 0, ^bd0, ^end) {pad_value = 7 : i32}
      ^bd0:
        aie.dma_bd(%buf : memref<8xi32> offset = 0 len = 8)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
  // Region form: aie.dma on a core tile.
  aie.device(npu2) {
    %t = aie.tile(1, 3)
    %buf = aie.buffer(%t) : memref<8xi32>
    %pl = aie.lock(%t, 0) {init = 1 : i32}
    %cl = aie.lock(%t, 1) {init = 0 : i32}
    aie.mem(%t) {
      %c1 = arith.constant 1 : i32
      // expected-error@+1 {{'aie.dma' op pad_value is only supported on memtile DMA channels}}
      %0 = aie.dma(MM2S, 0) {pad_value = 7 : i32} [{
        aie.use_lock(%cl, AcquireGreaterEqual, %c1)
        aie.dma_bd(%buf : memref<8xi32> offset = 0 len = 8)
        aie.use_lock(%pl, Release, %c1)
      }]
      aie.end
    }
  }
  // Terminator form: aie.dma_start on an S2MM memtile channel (register is MM2S-only).
  aie.device(npu2) {
    %t = aie.tile(1, 1)
    %buf = aie.buffer(%t) : memref<8xi32>
    aie.memtile_dma(%t) {
      // expected-error@+1 {{'aie.dma_start' op pad_value is only supported on MM2S DMA channels}}
      aie.dma_start("S2MM", 0, ^bd0, ^end) {pad_value = 7 : i32}
      ^bd0:
        aie.dma_bd(%buf : memref<8xi32> offset = 0 len = 8)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
  // Region form: aie.dma on an S2MM memtile channel.
  aie.device(npu2) {
    %t = aie.tile(1, 1)
    %buf = aie.buffer(%t) : memref<8xi32>
    %pl = aie.lock(%t, 0) {init = 1 : i32}
    %cl = aie.lock(%t, 1) {init = 0 : i32}
    aie.memtile_dma(%t) {
      %c1 = arith.constant 1 : i32
      // expected-error@+1 {{'aie.dma' op pad_value is only supported on MM2S DMA channels}}
      %0 = aie.dma(S2MM, 0) {pad_value = 7 : i32} [{
        aie.use_lock(%cl, AcquireGreaterEqual, %c1)
        aie.dma_bd(%buf : memref<8xi32> offset = 0 len = 8)
        aie.use_lock(%pl, Release, %c1)
      }]
      aie.end
    }
  }
}
