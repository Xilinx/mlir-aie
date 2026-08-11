//===- dma_bd_iteration.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s --split-input-file | FileCheck %s

// iteration_current == 0.
// CHECK: XAie_DmaSetBdIteration(&(dma_tile21_bd0), 16, 4, 0)
module @aie_module {
  aie.device(xcve2302) {
    %t01 = aie.tile(2, 1)
    %buf = aie.buffer(%t01) { address = 8192 : i32, sym_name = "in" } : memref<256xi32>
    %m01 = aie.memtile_dma(%t01) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 64 sizes = [2, 2, 2, 2] strides = [8, 4, 2, 1]) { iteration_size = 4 : i32, iteration_stride = 16 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// iteration_current == 2.
// CHECK: XAie_DmaSetBdIteration(&(dma_tile21_bd0), 16, 4, 2)
module @aie_module {
  aie.device(xcve2302) {
    %t01 = aie.tile(2, 1)
    %buf = aie.buffer(%t01) { address = 8192 : i32, sym_name = "in" } : memref<256xi32>
    %m01 = aie.memtile_dma(%t01) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 64) { iteration_size = 4 : i32, iteration_stride = 16 : i32, iteration_current = 2 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// non-32b element type is scaled to 32b word.
// CHECK: XAie_DmaSetBdIteration(&(dma_tile21_bd0), 8, 4, 0)
module @aie_module {
  aie.device(xcve2302) {
    %t01 = aie.tile(2, 1)
    %buf = aie.buffer(%t01) { address = 8192 : i32, sym_name = "in" } : memref<256xi64>
    %m01 = aie.memtile_dma(%t01) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<256xi64> offset = 0 len = 64) { iteration_size = 4 : i32, iteration_stride = 4 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// core tile (aie.mem) also lowers iteration.
// CHECK: XAie_DmaSetBdIteration(&(dma_tile22_bd0), 16, 4, 0)
module @aie_module {
  aie.device(xcve2302) {
    %t22 = aie.tile(2, 2)
    %buf = aie.buffer(%t22) { address = 1024 : i32, sym_name = "in" } : memref<256xi32>
    %m22 = aie.mem(%t22) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 64) { iteration_size = 4 : i32, iteration_stride = 16 : i32 }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// no BD iteration -> no XAie_DmaSetBdIteration.
// CHECK-NOT: XAie_DmaSetBdIteration
module @aie_module {
  aie.device(xcve2302) {
    %t01 = aie.tile(2, 1)
    %buf = aie.buffer(%t01) { address = 8192 : i32, sym_name = "in" } : memref<256xi32>
    %m01 = aie.memtile_dma(%t01) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 64 sizes = [2, 2, 2, 2] strides = [8, 4, 2, 1]) { }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
