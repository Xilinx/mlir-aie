//===- memory-affinity-vc2302.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// Check that we can legally access certain buffers/locks

// CHECK-LABEL: module @example0 {
// CHECK:       }

module @example0 {
 AIE.device(xcve2302) {

  // All  AIE rows: Local Memory on the East

  // (2, 3)<(3, 3) (4, 3) (5, 3)
  //          v
  // (2, 2) (3, 2) (4, 2) (5, 2)

  %t33 = AIE.tile(3, 3)
  %t32 = AIE.tile(3, 2)
  // %t34 = AIE.tile(3, 4)
  %t23 = AIE.tile(2, 3)
  // %t35 = AIE.tile(3, 5)
  // %t24 = AIE.tile(2, 4)

  %buf33 = AIE.buffer(%t33) : memref<256xi32>
  %buf32 = AIE.buffer(%t32) : memref<256xi32>
  // %buf34 = AIE.buffer(%t34) : memref<256xi32>
  %buf23 = AIE.buffer(%t23) : memref<256xi32>
  // %buf35 = AIE.buffer(%t35) : memref<256xi32>
  // %buf24 = AIE.buffer(%t24) : memref<256xi32>

  %c33 = AIE.core(%t33) {
    %idx1 = arith.constant 3 : index
    %val1 = arith.constant 7 : i32
    memref.store %val1, %buf33[%idx1] : memref<256xi32>
    memref.store %val1, %buf32[%idx1] : memref<256xi32>
    // memref.store %val1, %buf34[%idx1] : memref<256xi32>
    memref.store %val1, %buf23[%idx1] : memref<256xi32>
    AIE.end
  }
 }
}
