//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s
// CHECK:   error: 'aie.tile' op allocated buffers exceeded available memory
// CHECK:   Error in bank(s) : 3 
// CHECK:   MemoryMap:
// CHECK:   	bank : 0	  0x0-0x1FFF
// CHECK:   		(stack) 	: 0x0-0x3FF 	(1024 bytes)
// CHECK:   		      c 	: 0x400-0x41F 	(32 bytes)
// CHECK:   	bank : 1	  0x2000-0x3FFF
// CHECK:   		      a 	: 0x2000-0x200F 	(16 bytes)
// CHECK:   	bank : 2	  0x4000-0x5FFF
// CHECK:   	bank : 3	  0x6000-0x7FFF
// CHECK:   		      b 	: 0x6000-0xDFFF 	(32768 bytes)

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
