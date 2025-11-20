//===- test-max-min.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -convert-aievec-to-llvm -split-input-file | FileCheck %s
// RUN: aie-opt %s -convert-aievec-to-llvm="aie-target=aie2p" -split-input-file | FileCheck %s --check-prefix=AIE2P

// CHECK-LABEL: max_i32
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<16xi32>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<16xi32>
// AIE2P-LABEL: max_i32
// AIE2P-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<16xi32>
// AIE2P-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<16xi32>
func.func @max_i32(%lhs : vector<16xi32>, %rhs : vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[CMP:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[MAX:.*]] = "xllvm.intr.aie2.vmax.lt32"(%[[LHS]], %[[RHS]], %[[CMP]]) : (vector<16xi32>, vector<16xi32>, i32) -> !llvm.struct<(vector<16xi32>, i32)>
  // CHECK: %[[RES:.*]] = llvm.extractvalue %[[MAX]][0] : !llvm.struct<(vector<16xi32>, i32)>
  // AIE2P: %[[CMP:.*]] = llvm.mlir.constant(1 : i32) : i32
  // AIE2P: %[[MAX:.*]] = "xllvm.intr.aie2p.vmax.lt32"(%[[LHS]], %[[RHS]], %[[CMP]]) : (vector<16xi32>, vector<16xi32>, i32) -> !llvm.struct<(vector<16xi32>, i32)>
  // AIE2P: %[[RES:.*]] = llvm.extractvalue %[[MAX]][0] : !llvm.struct<(vector<16xi32>, i32)>
  %0 = aievec.max %lhs, %rhs : vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: max_i16
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<32xi16>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<32xi16>
// AIE2P-LABEL: max_i16
// AIE2P-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<32xi16>
// AIE2P-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<32xi16>
func.func @max_i16(%lhs : vector<32xi16>, %rhs : vector<32xi16>) -> vector<32xi16> {
  // CHECK: %[[CMP:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[MAX:.*]] = "xllvm.intr.aie2.vmax.lt16"(%[[LHS]], %[[RHS]], %[[CMP]]) : (vector<32xi16>, vector<32xi16>, i32) -> !llvm.struct<(vector<32xi16>, i32)>
  // CHECK: %[[RES:.*]] = llvm.extractvalue %[[MAX]][0] : !llvm.struct<(vector<32xi16>, i32)>
  // AIE2P: %[[CMP:.*]] = llvm.mlir.constant(1 : i32) : i32
  // AIE2P: %[[MAX:.*]] = "xllvm.intr.aie2p.vmax.lt16"(%[[LHS]], %[[RHS]], %[[CMP]]) : (vector<32xi16>, vector<32xi16>, i32) -> !llvm.struct<(vector<32xi16>, i32)>
  // AIE2P: %[[RES:.*]] = llvm.extractvalue %[[MAX]][0] : !llvm.struct<(vector<32xi16>, i32)>
  %0 = aievec.max %lhs, %rhs : vector<32xi16>
  return %0 : vector<32xi16>
}

// -----

// CHECK-LABEL: max_bf16
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<32xbf16>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<32xbf16>
// AIE2P-LABEL: max_bf16
// AIE2P-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<32xbf16>
// AIE2P-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<32xbf16>
func.func @max_bf16(%lhs : vector<32xbf16>, %rhs : vector<32xbf16>) -> vector<32xbf16> {
  // CHECK: %[[MAX:.*]] = "xllvm.intr.aie2.vmax.ltbf16"(%[[LHS]], %[[RHS]]) : (vector<32xbf16>, vector<32xbf16>) -> !llvm.struct<(vector<32xbf16>, i32)>
  // CHECK: %[[RES:.*]] = llvm.extractvalue %[[MAX]][0] : !llvm.struct<(vector<32xbf16>, i32)>
  // AIE2P: %[[MAX:.*]] = "xllvm.intr.aie2p.vmax.ltbf16"(%[[LHS]], %[[RHS]]) : (vector<32xbf16>, vector<32xbf16>) -> !llvm.struct<(vector<32xbf16>, i32)>
  // AIE2P: %[[RES:.*]] = llvm.extractvalue %[[MAX]][0] : !llvm.struct<(vector<32xbf16>, i32)>
  %0 = aievec.max %lhs, %rhs : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// CHECK-LABEL: min_i32
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<16xi32>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<16xi32>
// AIE2P-LABEL: min_i32
// AIE2P-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<16xi32>
// AIE2P-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<16xi32>
func.func @min_i32(%lhs : vector<16xi32>, %rhs : vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[CMP:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[MIN:.*]] = "xllvm.intr.aie2.vmin.ge32"(%[[LHS]], %[[RHS]], %[[CMP]]) : (vector<16xi32>, vector<16xi32>, i32) -> !llvm.struct<(vector<16xi32>, i32)>
  // CHECK: %[[RES:.*]] = llvm.extractvalue %[[MIN]][0] : !llvm.struct<(vector<16xi32>, i32)>
  // AIE2P: %[[CMP:.*]] = llvm.mlir.constant(1 : i32) : i32
  // AIE2P: %[[MIN:.*]] = "xllvm.intr.aie2p.vmin.ge32"(%[[LHS]], %[[RHS]], %[[CMP]]) : (vector<16xi32>, vector<16xi32>, i32) -> !llvm.struct<(vector<16xi32>, i32)>
  // AIE2P: %[[RES:.*]] = llvm.extractvalue %[[MIN]][0] : !llvm.struct<(vector<16xi32>, i32)>
  %0 = aievec.min %lhs, %rhs : vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: min_i16
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<32xi16>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<32xi16>
// AIE2P-LABEL: min_i16
// AIE2P-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<32xi16>
// AIE2P-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<32xi16>
func.func @min_i16(%lhs : vector<32xi16>, %rhs : vector<32xi16>) -> vector<32xi16> {
  // CHECK: %[[CMP:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[MIN:.*]] = "xllvm.intr.aie2.vmin.ge16"(%[[LHS]], %[[RHS]], %[[CMP]]) : (vector<32xi16>, vector<32xi16>, i32) -> !llvm.struct<(vector<32xi16>, i32)>
  // CHECK: %[[RES:.*]] = llvm.extractvalue %[[MIN]][0] : !llvm.struct<(vector<32xi16>, i32)>
  // AIE2P: %[[CMP:.*]] = llvm.mlir.constant(1 : i32) : i32
  // AIE2P: %[[MIN:.*]] = "xllvm.intr.aie2p.vmin.ge16"(%[[LHS]], %[[RHS]], %[[CMP]]) : (vector<32xi16>, vector<32xi16>, i32) -> !llvm.struct<(vector<32xi16>, i32)>
  // AIE2P: %[[RES:.*]] = llvm.extractvalue %[[MIN]][0] : !llvm.struct<(vector<32xi16>, i32)>
  %0 = aievec.min %lhs, %rhs : vector<32xi16>
  return %0 : vector<32xi16>
}

// -----

// CHECK-LABEL: min_bf16
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<32xbf16>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<32xbf16>
// AIE2P-LABEL: min_bf16
// AIE2P-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<32xbf16>
// AIE2P-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<32xbf16>
func.func @min_bf16(%lhs : vector<32xbf16>, %rhs : vector<32xbf16>) -> vector<32xbf16> {
  // CHECK: %[[MIN:.*]] = "xllvm.intr.aie2.vmin.gebf16"(%[[LHS]], %[[RHS]]) : (vector<32xbf16>, vector<32xbf16>) -> !llvm.struct<(vector<32xbf16>, i32)>
  // CHECK: %[[RES:.*]] = llvm.extractvalue %[[MIN]][0] : !llvm.struct<(vector<32xbf16>, i32)>
  // AIE2P: %[[MIN:.*]] = "xllvm.intr.aie2p.vmin.gebf16"(%[[LHS]], %[[RHS]]) : (vector<32xbf16>, vector<32xbf16>) -> !llvm.struct<(vector<32xbf16>, i32)>
  // AIE2P: %[[RES:.*]] = llvm.extractvalue %[[MIN]][0] : !llvm.struct<(vector<32xbf16>, i32)>
  %0 = aievec.min %lhs, %rhs : vector<32xbf16>
  return %0 : vector<32xbf16>
}
