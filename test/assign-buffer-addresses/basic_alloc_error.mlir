//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-assign-buffer-addresses="basicAlloc" %s 2>&1 | FileCheck %s
// CHECK:   error: 'aie.tile' op allocated buffers exceeded available memory
// CHECK:   (stack) : 0x0-0x3FF     (1024 bytes)
// CHECK:   b       : 0x400-0x83FF         (32768 bytes)
// CHECK:   c       : 0x8400-0x841F         (32 bytes)
// CHECK:   a       : 0x8420-0x842F         (16 bytes)

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
