//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.buffer' op in Column 1 and Row 1 is accessed from an unreachable tile in Column 4 and Row 4

module @test {
  %t1 = aie.tile(1, 1)
  %t2 = aie.tile(4, 4)
  %b1 = aie.buffer(%t1) { sym_name = "a" } : memref<16xi32>
  %core = aie.core(%t2) {
    %val1 = arith.constant 1 : i32
    %idx1 = arith.constant 3 : index
    memref.store %val1, %b1[%idx1] : memref<16xi32>
    aie.end
  }
}
