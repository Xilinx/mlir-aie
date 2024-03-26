//===- memtile_error.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s
// CHECK:   error: 'aie.tile' op allocated buffers exceeded available memory
// CHECK:   Error in bank(s) : 0 
// CHECK:   MemoryMap:
// CHECK:   	bank : 0	  0x0-0x7FFFF
// CHECK:   		    a 	: 0x0-0x80E7F 	(528000 bytes)


module @test {
 aie.device(xcve2302) {
  %0 = aie.tile(3, 1)
  %b1 = aie.buffer(%0) { sym_name = "a" } : memref<132000xi32>
  aie.memtile_dma(%0) {
    aie.end
  }
 }
}
