//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Sub-32-bit element types (e.g. i8) need every non-innermost dim's
// `stride * elementWidthInBytes` and the innermost dim's
// `size * elementWidthInBytes` to be a whole number of 32-bit words,
// because lib/Targets/AIERT.cpp's static/CDO lowering scales by
// elementWidthInBytes/4.0 and truncates via an unguarded static_cast
// (hardware-confirmed silent miscompilation). DMABDOp::verify now rejects
// values that are not word-granular instead of letting AIERT.cpp silently
// substitute a different, hardware-expressible value.

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// Control: non-innermost (outer) dim stride = 4 elements (4 * 1 byte =
// exactly 1 word). Expressible in hardware; must remain accepted.
module {
  aie.device(npu2) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<64xi8>
    aie.memtile_dma(%t1) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<64xi8> offset = 0 len = 8 sizes = [2, 4] strides = [4, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Negative companion: non-innermost dim stride = 1 element (0.25 words),
// not representable in the hardware stepsize register. Must be rejected.
module {
  aie.device(npu2) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<64xi8>
    aie.memtile_dma(%t1) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{non-innermost dim stride must be a multiple of 4 bytes for sub-32b element types}}
        aie.dma_bd(%buf : memref<64xi8> offset = 0 len = 8 sizes = [2, 4] strides = [1, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Control: innermost dim size = 4 elements (4 * 1 byte = exactly 1 word).
// Expressible in hardware; must remain accepted.
module {
  aie.device(npu2) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<64xi8>
    aie.memtile_dma(%t1) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<64xi8> offset = 0 len = 8 sizes = [2, 4] strides = [8, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Negative: innermost dim size = 5 elements (1.25 words). Must be rejected.
module {
  aie.device(npu2) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<64xi8>
    aie.memtile_dma(%t1) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{innermost dim size must be a multiple of 4 bytes for sub-32b element types}}
        aie.dma_bd(%buf : memref<64xi8> offset = 0 len = 16 sizes = [2, 5] strides = [8, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Negative: innermost dim size = 6 elements (1.5 words). This is the exact
// case hardware-confirmed to silently corrupt data (both declared sizes 4
// and 6 encode to the same 1-word hardware register). Must be rejected.
module {
  aie.device(npu2) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<64xi8>
    aie.memtile_dma(%t1) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{innermost dim size must be a multiple of 4 bytes for sub-32b element types}}
        aie.dma_bd(%buf : memref<64xi8> offset = 0 len = 16 sizes = [2, 6] strides = [8, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Negative: innermost dim size = 7 elements (1.75 words). Must be rejected.
module {
  aie.device(npu2) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<64xi8>
    aie.memtile_dma(%t1) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{innermost dim size must be a multiple of 4 bytes for sub-32b element types}}
        aie.dma_bd(%buf : memref<64xi8> offset = 0 len = 16 sizes = [2, 7] strides = [8, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
