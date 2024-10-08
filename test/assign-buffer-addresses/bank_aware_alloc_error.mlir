//===- bank_aware_alloc_error.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s 2>&1 | FileCheck %s
// CHECK:   error: Failed to allocate buffer: "b" with size: 32768 bytes.
// CHECK:   %1 = aie.buffer(%0) { sym_name = "b" } : memref<8192xi32>
// CHECK:        ^
// CHECK: note: see current operation: %2 = "aie.buffer"(%0) <{sym_name = "b"}> : (index) -> memref<8192xi32>
// CHECK: error: 'aie.tile' op All requested buffers don't fit in the available memory: Bank aware

// CHECK:   %0 = aie.tile(3, 3)
// CHECK:        ^
// CHECK: note: see current operation: %0 = "aie.tile"() <{col = 3 : i32, row = 3 : i32}> : () -> index
// CHECK: note: Current configuration of buffers in bank(s) : MemoryMap:
// CHECK:                 (stack)         : 0x0-0x3FF     (1024 bytes)
// CHECK:         bank : 0        0x0-0x1FFF
// CHECK:         bank : 1        0x2000-0x3FFF
// CHECK:         bank : 2        0x4000-0x5FFF
// CHECK:         bank : 3        0x6000-0x7FFF

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