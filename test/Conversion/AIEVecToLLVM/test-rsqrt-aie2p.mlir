//===- test-rsqrt-aie2p.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

// CHECK-LABEL: @scalar_f32_rsqrt
// CHECK-SAME: %[[ARG0:.*]]: f32
func.func @scalar_f32_rsqrt(%arg0 : f32) -> f32 {
  // CHECK: %[[RES:.*]] = "xllvm.intr.aie2p.invsqrt"(%[[ARG0]]) : (f32) -> f32
  // CHECK: return %[[RES]] : f32
  %0 = math.rsqrt %arg0 : f32
  return %0 : f32
}

// -----

// CHECK-LABEL: @vector_4xf32_rsqrt
// CHECK-SAME: %[[ARG0:.*]]: vector<4xf32>
func.func @vector_4xf32_rsqrt(%arg0 : vector<4xf32>) -> vector<4xf32> {
  // CHECK: %[[POISON:.*]] = llvm.mlir.poison : vector<4xf32>
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[ELEM0:.*]] = llvm.extractelement %[[ARG0]][%[[C0]] : i64] : vector<4xf32>
  // CHECK: %[[RSQRT0:.*]] = "xllvm.intr.aie2p.invsqrt"(%[[ELEM0]]) : (f32) -> f32
  // CHECK: %[[VEC0:.*]] = llvm.insertelement %[[RSQRT0]], %[[POISON]][%[[C0]] : i64] : vector<4xf32>
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[ELEM1:.*]] = llvm.extractelement %[[ARG0]][%[[C1]] : i64] : vector<4xf32>
  // CHECK: %[[RSQRT1:.*]] = "xllvm.intr.aie2p.invsqrt"(%[[ELEM1]]) : (f32) -> f32
  // CHECK: %[[VEC1:.*]] = llvm.insertelement %[[RSQRT1]], %[[VEC0]][%[[C1]] : i64] : vector<4xf32>
  // CHECK: %[[C2:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[ELEM2:.*]] = llvm.extractelement %[[ARG0]][%[[C2]] : i64] : vector<4xf32>
  // CHECK: %[[RSQRT2:.*]] = "xllvm.intr.aie2p.invsqrt"(%[[ELEM2]]) : (f32) -> f32
  // CHECK: %[[VEC2:.*]] = llvm.insertelement %[[RSQRT2]], %[[VEC1]][%[[C2]] : i64] : vector<4xf32>
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: %[[ELEM3:.*]] = llvm.extractelement %[[ARG0]][%[[C3]] : i64] : vector<4xf32>
  // CHECK: %[[RSQRT3:.*]] = "xllvm.intr.aie2p.invsqrt"(%[[ELEM3]]) : (f32) -> f32
  // CHECK: %[[VEC3:.*]] = llvm.insertelement %[[RSQRT3]], %[[VEC2]][%[[C3]] : i64] : vector<4xf32>
  // CHECK: return %[[VEC3]] : vector<4xf32>
  %0 = math.rsqrt %arg0 : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: @vector_16xf32_rsqrt
// CHECK-SAME: %[[ARG0:.*]]: vector<16xf32>
func.func @vector_16xf32_rsqrt(%arg0 : vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[POISON:.*]] = llvm.mlir.poison : vector<16xf32>
  // CHECK-COUNT-16: "xllvm.intr.aie2p.invsqrt"
  %0 = math.rsqrt %arg0 : vector<16xf32>
  return %0 : vector<16xf32>
}
