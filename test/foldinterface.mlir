//===- foldinterface.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// Check that the arith.constants in AIE.core are not moved out of the region by pass

// RUN: aie-opt %s -canonicalize | FileCheck %s
// CHECK: AIE.core
// CHECK: arith.constant
module @aie.herd_0  {
  %0 = AIE.tile(8, 3)
  %1 = AIE.buffer(%0) {sym_name = "b2"} : memref<32x32xi32>
  %2 = AIE.buffer(%0) {sym_name = "b1"} : memref<32x32xi32>
  %3 = AIE.buffer(%0) {sym_name = "b0"} : memref<32x32xi32>
  %4 = AIE.core(%0)  {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    scf.for %arg0 = %c0 to %c64 step %c32 {
      affine.for %arg1 = 0 to 32 {
        affine.for %arg2 = 0 to 32 {
          affine.for %arg3 = 0 to 32 {
            %5 = affine.load %3[%arg1, %arg3] : memref<32x32xi32>
            %6 = affine.load %2[%arg3, %arg2] : memref<32x32xi32>
            %7 = affine.load %1[%arg1, %arg2] : memref<32x32xi32>
            %8 = arith.muli %5, %6 : i32
            %9 = arith.addi %7, %8 : i32
            affine.store %9, %1[%arg1, %arg2] : memref<32x32xi32>
          }
        }
      }
    }
    AIE.end
  }
}
