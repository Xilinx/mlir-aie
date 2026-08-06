//===- nd-dma-pad-value-non-memtile.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

// The constant pad value is a MemTile MM2S channel register. A nonzero
// pad_value on a non-memtile (core-tile) channel has no hardware backing and
// must be rejected by the verifier.

module {
  aie.device(xcve2802) {
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
}
