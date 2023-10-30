//===- allocation_error.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aiecc.py --xchesscc --xbridge %s 2>&1 | FileCheck %s --check-prefix=CHESS
// RUN: not aiecc.py --no-xchesscc --no-xbridge %s 2>&1 | FileCheck %s --check-prefix=PEANO

// CHESS: Error: could not find free space for SpaceSymbol x in memory DMb
// PEANO: ld.lld: error: section '.bss' will not fit in region 'data': overflowed by 4 bytes

// REQUIRES: chess
// REQUIRES: peano
// If we use all of the local memory, then linking the AIE executable should fail.

module @example0 {
 AIE.device(xcvc1902) {
  memref.global @x : memref<4xi8> = uninitialized
  func.func @test (%i: index, %v: i8) -> i8 {
      %x = memref.get_global @x : memref<4xi8>
      memref.store %v, %x[%i] : memref<4xi8>
      %r = memref.load %x[%i] : memref<4xi8>
      func.return %r : i8
  }

  %t33 = AIE.tile(3, 3)

  // Use all the local memory for buffers, combined with the 1024 byte stack size.
  %buf33 = AIE.buffer(%t33) : memref<31744xi8>

  %c33 = AIE.core(%t33) {
    %idx1 = arith.constant 3 : index
    %val1 = arith.constant 7 : i8
    memref.store %val1, %buf33[%idx1] : memref<31744xi8>
    func.call @test(%idx1, %val1) : (index, i8) -> i8
    AIE.end
  }
 }
}
