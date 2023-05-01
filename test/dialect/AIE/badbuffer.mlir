//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aiecc.py %s |& FileCheck %s
// CHECK: error: 'AIE.buffer' op in Column 1 and Row 1 is accessed from an unreachable tile in Column 4 and Row 4

module @test {
  %t1 = AIE.tile(1, 1)
  %t2 = AIE.tile(4, 4)
  %b1 = AIE.buffer(%t1) { sym_name = "a" } : memref<16xi32>
  %core = AIE.core(%t2) {
    %val1 = arith.constant 1 : i32
    %idx1 = arith.constant 3 : index
    memref.store %val1, %b1[%idx1] : memref<16xi32>
    AIE.end
  }
}
