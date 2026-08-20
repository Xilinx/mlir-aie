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

// -----

// Positive: all dims have size 1 with a sub-32-bit element type and unit
// strides. Stride is never stepped and innermost size is 1, so word-
// granularity alignment is not required (matches test/python/dma_op.py).
module {
  aie.device(npu2) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<2xi16>
    aie.memtile_dma(%t1) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<2xi16> offset = 0 len = 2 sizes = [1, 1, 1, 1] strides = [1, 1, 1, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Positive: trailing size-1 dims after the dim that actually does the
// sub-32b work (the shape from matrix_multiplication/cascade on i16). Dim 1
// (size=32, stride=1) is the last dim with size > 1, so it is the effective
// innermost dim and is checked by size (32 * 2 bytes = 64 bytes, a whole
// number of words): 32 contiguous i16 reads, expressible in hardware. Dim 0
// (size=32, stride=32) sits outside the effective innermost dim and is
// checked by stride (32 * 2 bytes = 64 bytes, also word-aligned). Dims 2 and
// 3 have size == 1 and are skipped regardless of their position. Checking
// dim 1 by array position instead (as the non-innermost stride, 1 * 2 = 2
// bytes) would incorrectly reject this shape.
module {
  aie.device(npu2) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<1024xi16>
    aie.memtile_dma(%t1) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<1024xi16> offset = 0 len = 2048 sizes = [32, 32, 1, 1] strides = [32, 1, 32, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
