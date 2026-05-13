//===- test-scalarize-int-fp-conv.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Verify that vector forms of arith.{sitofp,uitofp,fptosi,fptoui} are
// scalarized into per-lane {vector.extract -> scalar conv -> reassemble}
// for the LLVMIR backend (AIE2 and AIE2P), and left untouched for the
// CPP backend or for scalar inputs.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p target-backend=llvmir" | FileCheck %s --check-prefixes=CHECK,LLVMIR
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2 target-backend=llvmir"  | FileCheck %s --check-prefixes=CHECK,LLVMIR
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2 target-backend=cpp"     | FileCheck %s --check-prefixes=CHECK,CPP

// -----------------------------------------------------------------------------
// fptosi: vector<16xbf16> -> vector<16xi32>  (the motivating mask use case)
// -----------------------------------------------------------------------------
// CHECK-LABEL: func @vec_fptosi_bf16_i32(
// CHECK-SAME:    %[[A:.*]]: vector<16xbf16>
func.func @vec_fptosi_bf16_i32(%a: vector<16xbf16>) -> vector<16xi32> {
  // LLVMIR: %[[E0:.*]] = vector.extract %[[A]][0] : bf16 from vector<16xbf16>
  // LLVMIR: %[[C0:.*]] = arith.fptosi %[[E0]] : bf16 to i32
  // LLVMIR: %[[E15:.*]] = vector.extract %[[A]][15] : bf16 from vector<16xbf16>
  // LLVMIR: %[[C15:.*]] = arith.fptosi %[[E15]] : bf16 to i32
  // LLVMIR: %[[R:.*]] = vector.from_elements {{.*}} : vector<16xi32>
  // LLVMIR: return %[[R]] : vector<16xi32>
  // CPP: %[[R:.*]] = arith.fptosi %[[A]] : vector<16xbf16> to vector<16xi32>
  // CPP: return %[[R]] : vector<16xi32>
  %0 = arith.fptosi %a : vector<16xbf16> to vector<16xi32>
  return %0 : vector<16xi32>
}

// -----------------------------------------------------------------------------
// fptoui: vector<8xf32> -> vector<8xi32>
// -----------------------------------------------------------------------------
// CHECK-LABEL: func @vec_fptoui_f32_i32(
// CHECK-SAME:    %[[A:.*]]: vector<8xf32>
func.func @vec_fptoui_f32_i32(%a: vector<8xf32>) -> vector<8xi32> {
  // LLVMIR-COUNT-8: arith.fptoui %{{.*}} : f32 to i32
  // LLVMIR: vector.from_elements {{.*}} : vector<8xi32>
  // CPP: arith.fptoui %[[A]] : vector<8xf32> to vector<8xi32>
  %0 = arith.fptoui %a : vector<8xf32> to vector<8xi32>
  return %0 : vector<8xi32>
}

// -----------------------------------------------------------------------------
// sitofp: vector<8xi32> -> vector<8xf32>
// -----------------------------------------------------------------------------
// CHECK-LABEL: func @vec_sitofp_i32_f32(
// CHECK-SAME:    %[[A:.*]]: vector<8xi32>
func.func @vec_sitofp_i32_f32(%a: vector<8xi32>) -> vector<8xf32> {
  // LLVMIR-COUNT-8: arith.sitofp %{{.*}} : i32 to f32
  // LLVMIR: vector.from_elements {{.*}} : vector<8xf32>
  // CPP: arith.sitofp %[[A]] : vector<8xi32> to vector<8xf32>
  %0 = arith.sitofp %a : vector<8xi32> to vector<8xf32>
  return %0 : vector<8xf32>
}

// -----------------------------------------------------------------------------
// uitofp: vector<8xi32> -> vector<8xf32>
// -----------------------------------------------------------------------------
// CHECK-LABEL: func @vec_uitofp_i32_f32(
// CHECK-SAME:    %[[A:.*]]: vector<8xi32>
func.func @vec_uitofp_i32_f32(%a: vector<8xi32>) -> vector<8xf32> {
  // LLVMIR-COUNT-8: arith.uitofp %{{.*}} : i32 to f32
  // LLVMIR: vector.from_elements {{.*}} : vector<8xf32>
  // CPP: arith.uitofp %[[A]] : vector<8xi32> to vector<8xf32>
  %0 = arith.uitofp %a : vector<8xi32> to vector<8xf32>
  return %0 : vector<8xf32>
}

// -----------------------------------------------------------------------------
// Scalar form must pass through untouched on every backend.
// -----------------------------------------------------------------------------
// CHECK-LABEL: func @scalar_fptosi(
// CHECK-SAME:    %[[A:.*]]: bf16
func.func @scalar_fptosi(%a: bf16) -> i32 {
  // CHECK-NEXT: %[[R:.*]] = arith.fptosi %[[A]] : bf16 to i32
  // CHECK-NEXT: return %[[R]] : i32
  %0 = arith.fptosi %a : bf16 to i32
  return %0 : i32
}

// -----------------------------------------------------------------------------
// Multi-dimensional vector forms are out of scope for the scalarize pattern.
// They must pass through this conversion unchanged (clearer failure later)
// rather than aborting here as "explicitly marked illegal".
// -----------------------------------------------------------------------------
// CHECK-LABEL: func @vec_2d_fptosi_passthrough(
// CHECK-SAME:    %[[A:.*]]: vector<2x4xf32>
func.func @vec_2d_fptosi_passthrough(%a: vector<2x4xf32>) -> vector<2x4xi32> {
  // CHECK-NEXT: %[[R:.*]] = arith.fptosi %[[A]] : vector<2x4xf32> to vector<2x4xi32>
  // CHECK-NEXT: return %[[R]] : vector<2x4xi32>
  %0 = arith.fptosi %a : vector<2x4xf32> to vector<2x4xi32>
  return %0 : vector<2x4xi32>
}
