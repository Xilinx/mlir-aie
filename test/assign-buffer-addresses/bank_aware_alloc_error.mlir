//===- bank_aware_alloc_error.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s
// CHECK: warning: Not all requested buffers fit in the available memory.
// CHECK: note: see current operation: %tile_3_3 = aie.tile(3, 3)
// CHECK: note: Current configuration of buffers in bank(s) : MemoryMap:
// CHECK:         bank : 0        0x0-0x1FFF
// CHECK:                 (stack)         : 0x0-0x3FF     (1024 bytes)
// CHECK:         bank : 1        0x2000-0x3FFF
// CHECK:         bank : 2        0x4000-0x5FFF
// CHECK:         bank : 3        0x6000-0x7FFF

// CHECK: error: {{.*}}could not be placed: buffer "b" needs 32768 bytes and this tile has no room left for it

module @test {
  aie.device(xcvc1902) {
    %0 = aie.tile(3, 3)
    %b1 = aie.buffer(%0) { sym_name = "a" } : memref<16xi8>
    %1 = aie.buffer(%0) { sym_name = "b" } : memref<8192xi32>
    %b2 = aie.buffer(%0) { sym_name = "c" } : memref<16xi16>
    %3 = aie.tile(4, 4)
    %4 = aie.buffer(%3) : memref<500xi32>
    aie.core(%0) {
      aie.end
    }
    aie.core(%3) {
      aie.end
    }
  }
}
