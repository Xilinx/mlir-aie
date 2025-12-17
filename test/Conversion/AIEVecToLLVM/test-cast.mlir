//===- test-cast.mlir ------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -convert-aievec-to-llvm | FileCheck %s

// CHECK-LABEL: @test_cast_zero_acc_f32
func.func @test_cast_zero_acc_f32() -> vector<16xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
  // CHECK: %[[ZERO_ACC:.*]] = "xllvm.intr.aie2.vbroadcast.zero.acc1024"() : () -> vector<16xi64>
  // CHECK-NEXT: %[[SHUFFLE:.*]] = vector.shuffle %[[ZERO_ACC]], %[[ZERO_ACC]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<16xi64>, vector<16xi64>
  // CHECK-NEXT: %[[RESULT:.*]] = llvm.bitcast %[[SHUFFLE]] : vector<8xi64> to vector<16xf32>
  %0 = aievec.cast %cst {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK: return %[[RESULT]] : vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @test_cast_nonzero_acc_f32
func.func @test_cast_nonzero_acc_f32(%arg0: vector<16xf32>) -> vector<16xf32> {
  // For non-zero constants, the cast should just fold away
  // CHECK-NOT: vbroadcast.zero.acc1024
  // CHECK: return %arg0 : vector<16xf32>
  %0 = aievec.cast %arg0 {isResAcc = true} : vector<16xf32>, vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @test_cast_non_acc
func.func @test_cast_non_acc() -> vector<16xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
  // When isResAcc = false, should just fold away
  // CHECK-NOT: vbroadcast.zero.acc1024
  // CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
  // CHECK: return %[[CST]] : vector<16xf32>
  %0 = aievec.cast %cst {isResAcc = false} : vector<16xf32>, vector<16xf32>
  return %0 : vector<16xf32>
}
