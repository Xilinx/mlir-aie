//===- base_test_1.aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: February 9th 2022
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-register-objectFifos %s | FileCheck %s

// CHECK-LABEL: module @arrays {
// CHECK-NEXT:    %0 = AIE.tile(1, 2)
// CHECK-NEXT:    %1 = AIE.buffer(%0) : memref<16xi32>
// CHECK-NEXT:    %2 = AIE.buffer(%0) : memref<16xi32>
// CHECK-NEXT:    %3 = AIE.buffer(%0) : memref<16xi32>
// CHECK-NEXT:    %4 = AIE.lock(%0, 0)
// CHECK-NEXT:    %5 = AIE.lock(%0, 1)
// CHECK-NEXT:    %6 = AIE.lock(%0, 2)
// CHECK-NEXT:    %7 = AIE.createArray(%1, %2, %3 : memref<16xi32>, memref<16xi32>, memref<16xi32>) : !AIE.array<memref<16xi32>>
// CHECK-NEXT:    %8 = AIE.createArray(%4, %5, %6 : index, index, index) : !AIE.array<index>
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %9 = AIE.array.load(%7 : !AIE.array<memref<16xi32>>, %c1) : memref<16xi32>
// CHECK-NEXT:    %10 = AIE.array.load(%8 : !AIE.array<index>, %c1) : index
// CHECK-NEXT:  }

module @arrays {
    %tile12 = AIE.tile(1, 2)

    %buff0 = AIE.buffer(%tile12) :  memref<16xi32>
    %buff1 = AIE.buffer(%tile12) :  memref<16xi32>
    %buff2 = AIE.buffer(%tile12) :  memref<16xi32>

    %lock0 = AIE.lock(%tile12, 0)
    %lock1 = AIE.lock(%tile12, 1)
    %lock2 = AIE.lock(%tile12, 2)

    %buff_array = AIE.createArray(%buff0, %buff1, %buff2 : memref<16xi32>, memref<16xi32>, memref<16xi32>) : !AIE.array<memref<16xi32>>
    %lock_array = AIE.createArray(%lock0, %lock1, %lock2 : index, index, index) : !AIE.array<index>

    %index = arith.constant 1 : index
    %buff = AIE.array.load(%buff_array : !AIE.array<memref<16xi32>>, %index) : memref<16xi32>
    %lock = AIE.array.load(%lock_array : !AIE.array<index>, %index) : index
}