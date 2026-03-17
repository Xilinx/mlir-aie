//===- test-32lane-aie2.mlir - Tests for 32-lane vector ops ---*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2 target-backend=llvmir" | FileCheck %s

// Tests for 32-lane vector operations on AIE2
// These operations are split into two 16-lane operations and concatenated back together

//===----------------------------------------------------------------------===//
// v32bf16 add tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @vecaddf_v32bf16(
// CHECK-SAME: %[[LHS:.*]]: vector<32xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xbf16>)
func.func @vecaddf_v32bf16(%arg0: vector<32xbf16>, %arg1: vector<32xbf16>) -> vector<32xbf16> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // Extract lower and upper halves of LHS
  // CHECK: %[[LHS_LOW:.*]] = aievec.ext %[[LHS]] {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
  // CHECK: %[[LHS_HIGH:.*]] = aievec.ext %[[LHS]] {index = 1 : i8} : vector<32xbf16>, vector<16xbf16>
  // Extract lower and upper halves of RHS
  // CHECK: %[[RHS_LOW:.*]] = aievec.ext %[[RHS]] {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
  // CHECK: %[[RHS_HIGH:.*]] = aievec.ext %[[RHS]] {index = 1 : i8} : vector<32xbf16>, vector<16xbf16>
  // Process lower half: ups -> add_elem -> srs
  // CHECK: %[[UPS_LHS_LOW:.*]] = aievec.ups %[[LHS_LOW]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK: %[[UPS_RHS_LOW:.*]] = aievec.ups %[[RHS_LOW]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK: %[[ADD_LOW:.*]] = aievec.add_elem %[[UPS_LHS_LOW]], %[[UPS_RHS_LOW]] : vector<16xf32>
  // CHECK: %[[SRS_LOW:.*]] = aievec.srs %[[ADD_LOW]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
  // Process upper half: ups -> add_elem -> srs
  // CHECK: %[[UPS_LHS_HIGH:.*]] = aievec.ups %[[LHS_HIGH]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK: %[[UPS_RHS_HIGH:.*]] = aievec.ups %[[RHS_HIGH]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK: %[[ADD_HIGH:.*]] = aievec.add_elem %[[UPS_LHS_HIGH]], %[[UPS_RHS_HIGH]] : vector<16xf32>
  // CHECK: %[[SRS_HIGH:.*]] = aievec.srs %[[ADD_HIGH]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
  // Concat back together
  // CHECK: %[[RESULT:.*]] = aievec.concat %[[SRS_LOW]], %[[SRS_HIGH]] : vector<16xbf16>, vector<32xbf16>
  %0 = arith.addf %arg0, %arg1 : vector<32xbf16>
  // CHECK: return %[[RESULT]] : vector<32xbf16>
  return %0 : vector<32xbf16>
}

//===----------------------------------------------------------------------===//
// v32bf16 sub tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @vecsubf_v32bf16(
// CHECK-SAME: %[[LHS:.*]]: vector<32xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xbf16>)
func.func @vecsubf_v32bf16(%arg0: vector<32xbf16>, %arg1: vector<32xbf16>) -> vector<32xbf16> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // Extract lower and upper halves of LHS
  // CHECK: %[[LHS_LOW:.*]] = aievec.ext %[[LHS]] {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
  // CHECK: %[[LHS_HIGH:.*]] = aievec.ext %[[LHS]] {index = 1 : i8} : vector<32xbf16>, vector<16xbf16>
  // Extract lower and upper halves of RHS
  // CHECK: %[[RHS_LOW:.*]] = aievec.ext %[[RHS]] {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
  // CHECK: %[[RHS_HIGH:.*]] = aievec.ext %[[RHS]] {index = 1 : i8} : vector<32xbf16>, vector<16xbf16>
  // Process lower half: ups -> sub_elem -> srs
  // CHECK: %[[UPS_LHS_LOW:.*]] = aievec.ups %[[LHS_LOW]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK: %[[UPS_RHS_LOW:.*]] = aievec.ups %[[RHS_LOW]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK: %[[SUB_LOW:.*]] = aievec.sub_elem %[[UPS_LHS_LOW]], %[[UPS_RHS_LOW]] : vector<16xf32>
  // CHECK: %[[SRS_LOW:.*]] = aievec.srs %[[SUB_LOW]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
  // Process upper half: ups -> sub_elem -> srs
  // CHECK: %[[UPS_LHS_HIGH:.*]] = aievec.ups %[[LHS_HIGH]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK: %[[UPS_RHS_HIGH:.*]] = aievec.ups %[[RHS_HIGH]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK: %[[SUB_HIGH:.*]] = aievec.sub_elem %[[UPS_LHS_HIGH]], %[[UPS_RHS_HIGH]] : vector<16xf32>
  // CHECK: %[[SRS_HIGH:.*]] = aievec.srs %[[SUB_HIGH]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
  // Concat back together
  // CHECK: %[[RESULT:.*]] = aievec.concat %[[SRS_LOW]], %[[SRS_HIGH]] : vector<16xbf16>, vector<32xbf16>
  %0 = arith.subf %arg0, %arg1 : vector<32xbf16>
  // CHECK: return %[[RESULT]] : vector<32xbf16>
  return %0 : vector<32xbf16>
}

//===----------------------------------------------------------------------===//
// v32bf16 mul tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @vecmulf_v32bf16(
// CHECK-SAME: %[[LHS:.*]]: vector<32xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xbf16>)
func.func @vecmulf_v32bf16(%arg0: vector<32xbf16>, %arg1: vector<32xbf16>) -> vector<32xbf16> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // Extract lower and upper halves of LHS
  // CHECK: %[[LHS_LOW:.*]] = aievec.ext %[[LHS]] {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
  // CHECK: %[[LHS_HIGH:.*]] = aievec.ext %[[LHS]] {index = 1 : i8} : vector<32xbf16>, vector<16xbf16>
  // Extract lower and upper halves of RHS
  // CHECK: %[[RHS_LOW:.*]] = aievec.ext %[[RHS]] {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
  // CHECK: %[[RHS_HIGH:.*]] = aievec.ext %[[RHS]] {index = 1 : i8} : vector<32xbf16>, vector<16xbf16>
  // Process lower half: mul_elem -> srs
  // CHECK: %[[MUL_LOW:.*]] = aievec.mul_elem %[[LHS_LOW]], %[[RHS_LOW]] : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
  // CHECK: %[[SRS_LOW:.*]] = aievec.srs %[[MUL_LOW]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
  // Process upper half: mul_elem -> srs
  // CHECK: %[[MUL_HIGH:.*]] = aievec.mul_elem %[[LHS_HIGH]], %[[RHS_HIGH]] : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
  // CHECK: %[[SRS_HIGH:.*]] = aievec.srs %[[MUL_HIGH]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
  // Concat back together
  // CHECK: %[[RESULT:.*]] = aievec.concat %[[SRS_LOW]], %[[SRS_HIGH]] : vector<16xbf16>, vector<32xbf16>
  %0 = arith.mulf %arg0, %arg1 : vector<32xbf16>
  // CHECK: return %[[RESULT]] : vector<32xbf16>
  return %0 : vector<32xbf16>
}

//===----------------------------------------------------------------------===//
// v32f32 add tests (split into two v16f32)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @vecaddf_v32f32(
// CHECK-SAME: %[[LHS:.*]]: vector<32xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xf32>)
func.func @vecaddf_v32f32(%arg0: vector<32xf32>, %arg1: vector<32xf32>) -> vector<32xf32> {
  // Extract lower and upper halves of LHS
  // CHECK: %[[LHS_LOW:.*]] = aievec.ext %[[LHS]] {index = 0 : i8} : vector<32xf32>, vector<16xf32>
  // CHECK: %[[LHS_HIGH:.*]] = aievec.ext %[[LHS]] {index = 1 : i8} : vector<32xf32>, vector<16xf32>
  // Extract lower and upper halves of RHS
  // CHECK: %[[RHS_LOW:.*]] = aievec.ext %[[RHS]] {index = 0 : i8} : vector<32xf32>, vector<16xf32>
  // CHECK: %[[RHS_HIGH:.*]] = aievec.ext %[[RHS]] {index = 1 : i8} : vector<32xf32>, vector<16xf32>
  // Process lower half: cast -> add_elem -> cast
  // CHECK: %[[CAST_LHS_LOW:.*]] = aievec.cast %[[LHS_LOW]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[CAST_RHS_LOW:.*]] = aievec.cast %[[RHS_LOW]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[ADD_LOW:.*]] = aievec.add_elem %[[CAST_LHS_LOW]], %[[CAST_RHS_LOW]] : vector<16xf32>
  // CHECK: %[[RESULT_LOW:.*]] = aievec.cast %[[ADD_LOW]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  // Process upper half: cast -> add_elem -> cast
  // CHECK: %[[CAST_LHS_HIGH:.*]] = aievec.cast %[[LHS_HIGH]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[CAST_RHS_HIGH:.*]] = aievec.cast %[[RHS_HIGH]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[ADD_HIGH:.*]] = aievec.add_elem %[[CAST_LHS_HIGH]], %[[CAST_RHS_HIGH]] : vector<16xf32>
  // CHECK: %[[RESULT_HIGH:.*]] = aievec.cast %[[ADD_HIGH]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  // Concat back together
  // CHECK: %[[RESULT:.*]] = aievec.concat %[[RESULT_LOW]], %[[RESULT_HIGH]] : vector<16xf32>, vector<32xf32>
  %0 = arith.addf %arg0, %arg1 : vector<32xf32>
  // CHECK: return %[[RESULT]] : vector<32xf32>
  return %0 : vector<32xf32>
}

//===----------------------------------------------------------------------===//
// v32f32 sub tests (split into two v16f32)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @vecsubf_v32f32(
// CHECK-SAME: %[[LHS:.*]]: vector<32xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<32xf32>)
func.func @vecsubf_v32f32(%arg0: vector<32xf32>, %arg1: vector<32xf32>) -> vector<32xf32> {
  // Extract lower and upper halves of LHS
  // CHECK: %[[LHS_LOW:.*]] = aievec.ext %[[LHS]] {index = 0 : i8} : vector<32xf32>, vector<16xf32>
  // CHECK: %[[LHS_HIGH:.*]] = aievec.ext %[[LHS]] {index = 1 : i8} : vector<32xf32>, vector<16xf32>
  // Extract lower and upper halves of RHS
  // CHECK: %[[RHS_LOW:.*]] = aievec.ext %[[RHS]] {index = 0 : i8} : vector<32xf32>, vector<16xf32>
  // CHECK: %[[RHS_HIGH:.*]] = aievec.ext %[[RHS]] {index = 1 : i8} : vector<32xf32>, vector<16xf32>
  // Process lower half: cast -> sub_elem -> cast
  // CHECK: %[[CAST_LHS_LOW:.*]] = aievec.cast %[[LHS_LOW]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[CAST_RHS_LOW:.*]] = aievec.cast %[[RHS_LOW]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[SUB_LOW:.*]] = aievec.sub_elem %[[CAST_LHS_LOW]], %[[CAST_RHS_LOW]] : vector<16xf32>
  // CHECK: %[[RESULT_LOW:.*]] = aievec.cast %[[SUB_LOW]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  // Process upper half: cast -> sub_elem -> cast
  // CHECK: %[[CAST_LHS_HIGH:.*]] = aievec.cast %[[LHS_HIGH]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[CAST_RHS_HIGH:.*]] = aievec.cast %[[RHS_HIGH]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK: %[[SUB_HIGH:.*]] = aievec.sub_elem %[[CAST_LHS_HIGH]], %[[CAST_RHS_HIGH]] : vector<16xf32>
  // CHECK: %[[RESULT_HIGH:.*]] = aievec.cast %[[SUB_HIGH]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  // Concat back together
  // CHECK: %[[RESULT:.*]] = aievec.concat %[[RESULT_LOW]], %[[RESULT_HIGH]] : vector<16xf32>, vector<32xf32>
  %0 = arith.subf %arg0, %arg1 : vector<32xf32>
  // CHECK: return %[[RESULT]] : vector<32xf32>
  return %0 : vector<32xf32>
}
