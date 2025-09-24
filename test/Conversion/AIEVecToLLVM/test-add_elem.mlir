//===- test-add_elem.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -convert-aievec-to-llvm -split-input-file | FileCheck %s

// CHECK-LABEL: add_elem_flat_fp32
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<16xf32>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<16xf32>
func.func @add_elem_flat_fp32(%lhs : vector<16xf32>, %rhs : vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[CONF:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[LHS_BC:.*]] = llvm.bitcast %[[LHS]] : vector<16xf32> to vector<8xi64>
  // CHECK: %[[RHS_BC:.*]] = llvm.bitcast %[[RHS]] : vector<16xf32> to vector<8xi64>
  // CHECK: %[[ADD:.*]] = "xllvm.intr.aie2.add.accfloat"(%[[LHS_BC]], %[[RHS_BC]], %[[CONF]]) : (vector<8xi64>, vector<8xi64>, i32) -> vector<8xi64>
  // CHECK: %[[RES:.*]] = llvm.bitcast %[[ADD]] : vector<8xi64> to vector<16xf32>
  %0 = aievec.add_elem %lhs, %rhs : vector<16xf32>
  return %0 : vector<16xf32>
}
