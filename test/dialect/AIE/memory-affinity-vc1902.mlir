//===- example0.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// Check that we can legally access certain buffers/locks

// CHECK-LABEL: module @example0 {
// CHECK:       }

module @example0 {
 aie.device(xcvc1902) {

  // Odd  AIE rows: DMem on the East
  // Even AIE rows: DMem on the West

  // (2, 5) (3, 5) (4, 5) (5, 5)
  //           ^
  // (2, 4) (3, 4)>(4, 4) (5, 4)
  //          V^
  // (2, 3)<(3, 3) (4, 3) (5, 3)
  //          v
  // (2, 2) (3, 2) (4, 2) (5, 2)

  %t33 = aie.tile(3, 3)
  %t32 = aie.tile(3, 2)
  %t34 = aie.tile(3, 4)
  %t23 = aie.tile(2, 3)
  %t35 = aie.tile(3, 5)
  %t44 = aie.tile(4, 4)

  aie.buffer @buf33(%t33) : memref<256xi32>
  aie.buffer @buf32(%t32) : memref<256xi32>
  aie.buffer @buf34(%t34) : memref<256xi32>
  aie.buffer @buf23(%t23) : memref<256xi32>
  aie.buffer @buf35(%t35) : memref<256xi32>
  aie.buffer @buf44(%t44) : memref<256xi32>

  %c33 = aie.core(%t33) {
    %idx1 = arith.constant 3 : index
    %val1 = arith.constant 7 : i32
    %buf33 = aie.get_buffer @buf33 : memref<256xi32>
    memref.store %val1, %buf33[%idx1] : memref<256xi32>
    %buf32 = aie.get_buffer @buf32 : memref<256xi32>
    memref.store %val1, %buf32[%idx1] : memref<256xi32>
    %buf34 = aie.get_buffer @buf34 : memref<256xi32>
    memref.store %val1, %buf34[%idx1] : memref<256xi32>
    %buf23 = aie.get_buffer @buf23 : memref<256xi32>
    memref.store %val1, %buf23[%idx1] : memref<256xi32>
    aie.end
  }
  %c34 = aie.core(%t34) {
    %idx1 = arith.constant 3 : index
    %val1 = arith.constant 7 : i32
    %buf34_1 = aie.get_buffer @buf34 : memref<256xi32>
    memref.store %val1, %buf34_1[%idx1] : memref<256xi32>
    %buf33_1 = aie.get_buffer @buf33 : memref<256xi32>
    memref.store %val1, %buf33_1[%idx1] : memref<256xi32>
    %buf35 = aie.get_buffer @buf35 : memref<256xi32>
    memref.store %val1, %buf35[%idx1] : memref<256xi32>
    %buf44 = aie.get_buffer @buf44 : memref<256xi32>
    memref.store %val1, %buf44[%idx1] : memref<256xi32>
    aie.end
  }
 }
}
