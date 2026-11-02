//===- nd-dma-pad-four-dims.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A MemTile walk over four dimensions emits its d2 padding only when d3 is
// unused, and d3 has none, so the outer two dimensions must be unpadded.

aie.device(npu2) {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) : memref<64xi32>
  %mem = aie.memtile_dma(%t) {
    aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      // expected-error@+1 {{a four-dimensional walk pads only its two innermost dimensions}}
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 96 sizes = [2, 2, 2, 8] strides = [32, 16, 8, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 1, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>])
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
      // expected-error@+1 {{a four-dimensional walk pads only its two innermost dimensions}}
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 128 sizes = [2, 2, 2, 8] strides = [32, 16, 8, 1] pad [<const_pad_before = 1, const_pad_after = 1>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>])
      aie.next_bd ^end
    ^end:
      aie.end
  }
}

// -----

// The two innermost dimensions pad as usual.
aie.device(npu2) {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) : memref<64xi32>
  %mem = aie.memtile_dma(%t) {
    aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 256 sizes = [2, 2, 2, 8] strides = [32, 16, 8, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 1, const_pad_after = 1>, <const_pad_before = 0, const_pad_after = 8>])
      aie.next_bd ^end
    ^end:
      aie.end
  }
}
