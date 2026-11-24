//===- nd-dma-pad-field-widths.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A MemTile BD's pad fields are 6, 5 and 4 bits wide for d0 (in 32-bit words),
// d1 and d2.

aie.device(npu2) {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) : memref<64xi32>
  %mem = aie.memtile_dma(%t) {
    aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      // expected-error@+1 {{D0 pad_before/pad_after exceeds the [0:63] range of 32-bit words.}}
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 144 sizes = [2, 8] strides = [8, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 64>])
      aie.next_bd ^end
    ^end:
      aie.end
  }
}

// -----

aie.device(npu2) {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) : memref<64xi32>
  %mem = aie.memtile_dma(%t) {
    aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      // expected-error@+1 {{D1 pad_before/pad_after exceeds the [0:31] range.}}
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 272 sizes = [2, 8] strides = [8, 1] pad [<const_pad_before = 32, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>])
      aie.next_bd ^end
    ^end:
      aie.end
  }
}

// -----

aie.device(npu1) {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) : memref<64xi32>
  %mem = aie.memtile_dma(%t) {
    aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      // expected-error@+1 {{D2 pad_before/pad_after exceeds the [0:15] range.}}
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 288 sizes = [2, 2, 8] strides = [16, 8, 1] pad [<const_pad_before = 16, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>])
      aie.next_bd ^end
    ^end:
      aie.end
  }
}

// -----

// Each field at its maximum.
aie.device(npu2) {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) : memref<64xi32>
  %mem = aie.memtile_dma(%t) {
    aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 39831 sizes = [2, 2, 8] strides = [16, 8, 1] pad [<const_pad_before = 15, const_pad_after = 0>, <const_pad_before = 31, const_pad_after = 0>, <const_pad_before = 63, const_pad_after = 0>])
      aie.next_bd ^end
    ^end:
      aie.end
  }
}

// -----

// 252 i8 elements are 63 words.
aie.device(npu2) {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) : memref<64xi8>
  %mem = aie.memtile_dma(%t) {
    aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<64xi8> offset = 0 len = 520 sizes = [2, 8] strides = [8, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 252, const_pad_after = 0>])
      aie.next_bd ^end
    ^end:
      aie.end
  }
}

// -----

aie.device(npu2) {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) : memref<64xi8>
  %mem = aie.memtile_dma(%t) {
    aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      // expected-error@+1 {{D0 pad_before/pad_after exceeds the [0:63] range of 32-bit words.}}
      aie.dma_bd(%buf : memref<64xi8> offset = 0 len = 528 sizes = [2, 8] strides = [8, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 256, const_pad_after = 0>])
      aie.next_bd ^end
    ^end:
      aie.end
  }
}
