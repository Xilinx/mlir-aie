//===- normalize-dma-bd-dims.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for AIENormalizeDmaBdDims: strips size-1 degenerate dimensions from
// DMABDOp and, when the result is contiguous, linearizes the BD to scalar
// (len-only) form.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-normalize-dma-bd-dims --split-input-file %s | FileCheck %s

// -----

// Test 1: Degenerate leading dim stripped but result is NOT contiguous.
// Input:  sizes = [1, 4, 32]  strides = [128, 64, 1]
// After stripping size-1: sizes = [4, 32]  strides = [64, 1]
// Contiguous? innermost stride = 1, product = 32, outer stride = 64 != 32.
// => NOT contiguous; only strip, do not linearize.

// CHECK-LABEL: @strip_noncontiguous
// CHECK:         aie.shim_dma
// CHECK:           aie.dma_bd
// CHECK-SAME:        sizes = [4, 32] strides = [64, 1]
module @strip_noncontiguous {
  aie.device(npu1) {
    %tile = aie.tile(0, 0)
    %buf = aie.external_buffer { sym_name = "buf_t1" } : memref<256xi32>
    aie.shim_dma(%tile) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        // sizes=[1,4,32] strides=[128,64,1]: leading dim is degenerate (size=1)
        // maxIdx = (1-1)*128 + (4-1)*64 + (32-1)*1 = 0 + 192 + 31 = 223 < 256
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 128
                   sizes = [1, 4, 32] strides = [128, 64, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Test 2: Degenerate leading dim stripped and result IS contiguous → linearize.
// Input:  sizes = [1, 4, 64]  strides = [256, 64, 1]
// After stripping: sizes = [4, 64]  strides = [64, 1]
// Contiguous? innermost stride = 1, product = 64, outer stride = 64 == 64.
// product of all orig dims = 1 * 4 * 64 = 256 == len(256). Linearize to len=256.

// CHECK-LABEL: @strip_then_linearize
// CHECK:         aie.shim_dma
// CHECK:           aie.dma_bd(%{{.*}} : memref<256xi32> offset = 0 len = 256)
// CHECK-NOT:         sizes
module @strip_then_linearize {
  aie.device(npu1) {
    %tile = aie.tile(0, 0)
    %buf = aie.external_buffer { sym_name = "buf_t2" } : memref<256xi32>
    aie.shim_dma(%tile) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        // sizes=[1,4,64] strides=[256,64,1]: strip leading dim, then contiguous.
        // maxIdx = (1-1)*256 + (4-1)*64 + (64-1)*1 = 0 + 192 + 63 = 255 < 256
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256
                   sizes = [1, 4, 64] strides = [256, 64, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Test 3: No degenerate dims → pass is a no-op.
// Input:  sizes = [4, 32]  strides = [128, 1]   (non-contiguous, no size-1)
// newDims.size() == origDims.size(), so pass returns without modification.

// CHECK-LABEL: @no_degenerate_dims
// CHECK:         aie.shim_dma
// CHECK:           aie.dma_bd
// CHECK-SAME:        sizes = [4, 32] strides = [128, 1]
module @no_degenerate_dims {
  aie.device(npu1) {
    %tile = aie.tile(0, 0)
    %buf = aie.external_buffer { sym_name = "buf_t3" } : memref<512xi32>
    aie.shim_dma(%tile) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        // No size-1 dims; pass should not touch this BD.
        // maxIdx = (4-1)*128 + (32-1)*1 = 384 + 31 = 415 < 512
        aie.dma_bd(%buf : memref<512xi32> offset = 0 len = 128
                   sizes = [4, 32] strides = [128, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Test 4: All dims are degenerate (size=1) → stripped to empty → linearize.
// Input:  sizes = [1, 1, 1]  strides = [4, 2, 1]
// After stripping: newDims is empty. newDims.empty() triggers the contiguous
// branch. product = 1 * 1 * 1 = 1 == len(1). Linearize to len=1.

// CHECK-LABEL: @all_degenerate_linearize
// CHECK:         aie.shim_dma
// CHECK:           aie.dma_bd(%{{.*}} : memref<4xi32> offset = 0 len = 1)
// CHECK-NOT:         sizes
module @all_degenerate_linearize {
  aie.device(npu1) {
    %tile = aie.tile(0, 0)
    %buf = aie.external_buffer { sym_name = "buf_t4" } : memref<4xi32>
    aie.shim_dma(%tile) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        // All sizes are 1; maxIdx = 0 < 4. product = 1, len = 1.
        aie.dma_bd(%buf : memref<4xi32> offset = 0 len = 1
                   sizes = [1, 1, 1] strides = [4, 2, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Test 5: Already linear (no sizes/strides) → pass skips via getMixedSizes().empty().
// Input: len = 1024, no sizes, no strides.
// The pass returns immediately on line `if (op.getMixedSizes().empty()) return;`.

// CHECK-LABEL: @already_linear
// CHECK:         aie.shim_dma
// CHECK:           aie.dma_bd(%{{.*}} : memref<1024xi32> offset = 0 len = 1024)
// CHECK-NOT:         sizes
module @already_linear {
  aie.device(npu1) {
    %tile = aie.tile(0, 0)
    %buf = aie.external_buffer { sym_name = "buf_t5" } : memref<1024xi32>
    aie.shim_dma(%tile) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 1024)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Test 6: Cascade-like i16 BD with two trailing degenerate dims → linearize.
// Input:  sizes = [32, 32, 1, 1]  strides = [32, 1, 32, 1]  (i16 elements)
// After stripping size-1 (positions 2 and 3): sizes = [32, 32]  strides = [32, 1]
// Contiguous? innermost stride = 1, product = 32, outer stride = 32 == 32. YES.
// product = 32 * 32 * 1 * 1 = 1024 == len(1024). Linearize to len=1024.
// Needs MemTile for 4-dim BD support; xcve2802 tile(1,1) is a MemTile.

// CHECK-LABEL: @cascade_i16_strip_linearize
// CHECK:         aie.memtile_dma
// CHECK:           aie.dma_bd(%{{.*}} : memref<1024xi16> offset = 0 len = 1024)
// CHECK-NOT:         sizes
module @cascade_i16_strip_linearize {
  aie.device(xcve2802) {
    %tile = aie.tile(1, 1)
    %buf = aie.buffer(%tile) { sym_name = "buf_t6" } : memref<1024xi16>
    aie.memtile_dma(%tile) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        // sizes=[32,32,1,1] strides=[32,1,32,1] on i16:
        // maxIdx = (32-1)*32 + (32-1)*1 + 0 + 0 = 992 + 31 = 1023 < 1024
        // innermost stride for size>1 is dims[1].stride=1 (valid for i16).
        // outer non-degenerate dim: stride=32, 32*2=64 bytes is word-aligned.
        aie.dma_bd(%buf : memref<1024xi16> offset = 0 len = 1024
                   sizes = [32, 32, 1, 1] strides = [32, 1, 32, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Test 7: Iteration case — outermost dim is BD iteration (len < product).
// Input:  sizes = [8, 1, 256, 32]  strides = [8192, 32, 32, 1]  len = 8192
// product = 8 * 1 * 256 * 32 = 65536, but len = 8192 != 65536.
// The outermost dim encodes BD iteration; stripping the size-1 dim would
// change the dimension count and confuse downstream passes that interpret
// the 4th dim as iteration. Pass must leave this BD unchanged.
// maxIdx = (8-1)*8192 + 0 + (256-1)*32 + (32-1)*1 = 57344 + 8160 + 31 = 65535

// CHECK-LABEL: @iteration_case_no_strip
// CHECK:         aie.memtile_dma
// CHECK:           aie.dma_bd
// CHECK-SAME:        sizes = [8, 1, 256, 32] strides = [8192, 32, 32, 1]
module @iteration_case_no_strip {
  aie.device(xcve2802) {
    %tile = aie.tile(1, 1)
    %buf = aie.buffer(%tile) { sym_name = "buf_t7" } : memref<65536xi32>
    aie.memtile_dma(%tile) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<65536xi32> offset = 0 len = 8192
                   sizes = [8, 1, 256, 32] strides = [8192, 32, 32, 1])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
