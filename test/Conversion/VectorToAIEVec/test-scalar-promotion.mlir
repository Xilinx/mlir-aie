//===- test-scalar-promotion.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Tests for scalar arith op promotion to vector aievec ops.
// Scalar ops are promoted to 512-bit vector ops to prevent LLVM's SLP
// vectorizer from creating sub-512-bit vectors that crash the AIE2 backend.
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s --check-prefix=AIE2P

// Test 1: scalar maxsi i32

// CHECK-LABEL: func.func @scalar_maxsi_i32(
// CHECK: aievec.broadcast_scalar
// CHECK: aievec.broadcast_scalar
// CHECK: aievec.max
// CHECK: aievec.ext_elem
// CHECK-NOT: arith.maxsi
// AIE2P-LABEL: func.func @scalar_maxsi_i32(
// AIE2P: aievec.broadcast_scalar
// AIE2P: aievec.broadcast_scalar
// AIE2P: aievec.max
// AIE2P: aievec.ext_elem
// AIE2P-NOT: arith.maxsi
func.func @scalar_maxsi_i32(%a: i32, %b: i32) -> i32 {
  %0 = arith.maxsi %a, %b : i32
  return %0 : i32
}

// Test 2: scalar minsi i32

// CHECK-LABEL: func.func @scalar_minsi_i32(
// CHECK: aievec.broadcast_scalar
// CHECK: aievec.broadcast_scalar
// CHECK: aievec.min
// CHECK: aievec.ext_elem
// CHECK-NOT: arith.minsi
// AIE2P-LABEL: func.func @scalar_minsi_i32(
// AIE2P: aievec.broadcast_scalar
// AIE2P: aievec.broadcast_scalar
// AIE2P: aievec.min
// AIE2P: aievec.ext_elem
// AIE2P-NOT: arith.minsi
func.func @scalar_minsi_i32(%a: i32, %b: i32) -> i32 {
  %0 = arith.minsi %a, %b : i32
  return %0 : i32
}

// Test 3: scalar shrsi i32

// CHECK-LABEL: func.func @scalar_shrsi_i32(
// CHECK: aievec.broadcast_scalar
// CHECK: aievec.ups
// CHECK: aievec.srs
// CHECK: aievec.ext_elem
// CHECK-NOT: arith.shrsi
// AIE2P-LABEL: func.func @scalar_shrsi_i32(
// AIE2P: aievec.broadcast_scalar
// AIE2P: aievec.ups
// AIE2P: aievec.srs
// AIE2P: aievec.ext_elem
// AIE2P-NOT: arith.shrsi
func.func @scalar_shrsi_i32(%a: i32, %b: i32) -> i32 {
  %0 = arith.shrsi %a, %b : i32
  return %0 : i32
}

// Test 4: scalar maxsi i16

// CHECK-LABEL: func.func @scalar_maxsi_i16(
// CHECK: aievec.broadcast_scalar
// CHECK: aievec.broadcast_scalar
// CHECK: aievec.max
// CHECK: aievec.ext_elem
// AIE2P-LABEL: func.func @scalar_maxsi_i16(
// AIE2P: aievec.broadcast_scalar
// AIE2P: aievec.broadcast_scalar
// AIE2P: aievec.max
// AIE2P: aievec.ext_elem
func.func @scalar_maxsi_i16(%a: i16, %b: i16) -> i16 {
  %0 = arith.maxsi %a, %b : i16
  return %0 : i16
}

// Test 5: scalar i64 should NOT be promoted (passthrough)

// CHECK-LABEL: func.func @scalar_maxsi_i64(
// CHECK: arith.maxsi
// AIE2P-LABEL: func.func @scalar_maxsi_i64(
// AIE2P: arith.maxsi
func.func @scalar_maxsi_i64(%a: i64, %b: i64) -> i64 {
  %0 = arith.maxsi %a, %b : i64
  return %0 : i64
}
