//===- test-cast.mlir ------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -convert-aievec-to-llvm | FileCheck %s --check-prefix=CHECK-AIE2
// RUN: aie-opt %s --convert-aievec-to-llvm='aie-target=aie2p' | FileCheck %s --check-prefix=CHECK-AIE2P

// CHECK-AIE2-LABEL: @test_cast_zero_acc_f32
// CHECK-AIE2P-LABEL: @test_cast_zero_acc_f32
func.func @test_cast_zero_acc_f32() -> vector<16xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
  // AIE2: uses vbroadcast.zero.acc1024 intrinsic for zero accumulator
  // CHECK-AIE2: %[[ZERO_ACC:.*]] = "xllvm.intr.aie2.vbroadcast.zero.acc1024"() : () -> vector<16xi64>
  // CHECK-AIE2-NEXT: %[[SHUFFLE:.*]] = vector.shuffle %[[ZERO_ACC]], %[[ZERO_ACC]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<16xi64>, vector<16xi64>
  // CHECK-AIE2-NEXT: %[[RESULT:.*]] = llvm.bitcast %[[SHUFFLE]] : vector<8xi64> to vector<16xf32>
  
  // AIE2P: simply folds the cast
  // CHECK-AIE2P: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
  %0 = aievec.cast %cst {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK-AIE2: return %[[RESULT]] : vector<16xf32>
  // CHECK-AIE2P: return %[[CST]] : vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-AIE2-LABEL: @test_cast_nonzero_acc_f32
// CHECK-AIE2P-LABEL: @test_cast_nonzero_acc_f32
func.func @test_cast_nonzero_acc_f32(%arg0: vector<16xf32>) -> vector<16xf32> {
  // For non-zero constants, the cast should just fold away for both AIE2 and AIE2P
  // CHECK-AIE2-NOT: vbroadcast.zero.acc1024
  // CHECK-AIE2: return %arg0 : vector<16xf32>
  // CHECK-AIE2P: return %arg0 : vector<16xf32>
  %0 = aievec.cast %arg0 {isResAcc = true} : vector<16xf32>, vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-AIE2-LABEL: @test_cast_non_acc
// CHECK-AIE2P-LABEL: @test_cast_non_acc
func.func @test_cast_non_acc() -> vector<16xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
  // When isResAcc = false, should just fold away for both AIE2 and AIE2P
  // CHECK-AIE2-NOT: vbroadcast.zero.acc1024
  // CHECK-AIE2: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
  // CHECK-AIE2: return %[[CST]] : vector<16xf32>
  // CHECK-AIE2P: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
  // CHECK-AIE2P: return %[[CST]] : vector<16xf32>
  %0 = aievec.cast %cst {isResAcc = false} : vector<16xf32>, vector<16xf32>
  return %0 : vector<16xf32>
}
