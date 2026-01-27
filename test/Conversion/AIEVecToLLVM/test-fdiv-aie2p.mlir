//===- test-fdiv-aie2p.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

// CHECK-LABEL: @scalar_f32_fdiv
// CHECK-SAME: %[[LHS:.*]]: f32, %[[RHS:.*]]: f32
func.func @scalar_f32_fdiv(%lhs : f32, %rhs : f32) -> f32 {
  // CHECK: %[[RES:.*]] = arith.divf %[[LHS]], %[[RHS]] : f32
  // CHECK: return %[[RES]] : f32
  %0 = arith.divf %lhs, %rhs : f32
  return %0 : f32
}

// -----

// CHECK: llvm.func @__aie2p_scalar_fdiv
// CHECK-LABEL: @vector_4xf32_fdiv
// CHECK-SAME: %[[LHS:.*]]: vector<4xf32>, %[[RHS:.*]]: vector<4xf32>
func.func @vector_4xf32_fdiv(%lhs : vector<4xf32>, %rhs : vector<4xf32>) -> vector<4xf32> {
  // CHECK: %[[POISON:.*]] = llvm.mlir.poison : vector<4xf32>
  // CHECK-COUNT-4: llvm.call @__aie2p_scalar_fdiv
  %0 = arith.divf %lhs, %rhs : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK: llvm.func @__aie2p_scalar_fdiv
// CHECK-LABEL: @vector_16xf32_fdiv
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>, %[[RHS:.*]]: vector<16xf32>
func.func @vector_16xf32_fdiv(%lhs : vector<16xf32>, %rhs : vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[POISON:.*]] = llvm.mlir.poison : vector<16xf32>
  // CHECK-COUNT-16: llvm.call @__aie2p_scalar_fdiv
  %0 = arith.divf %lhs, %rhs : vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// CHECK: llvm.func @__aie2p_scalar_fdiv
// CHECK-LABEL: @vector_8xf32_fdiv
// CHECK-SAME: %[[LHS:.*]]: vector<8xf32>, %[[RHS:.*]]: vector<8xf32>
func.func @vector_8xf32_fdiv(%lhs : vector<8xf32>, %rhs : vector<8xf32>) -> vector<8xf32> {
  // CHECK: %[[POISON:.*]] = llvm.mlir.poison : vector<8xf32>
  // CHECK-COUNT-8: llvm.call @__aie2p_scalar_fdiv
  %0 = arith.divf %lhs, %rhs : vector<8xf32>
  return %0 : vector<8xf32>
}
