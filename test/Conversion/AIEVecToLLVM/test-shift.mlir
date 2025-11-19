//===- test-shift.mlir ------------------------------------------*- MLIR -*-===//
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

// CHECK-LABEL: shift_i32
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<16xi32>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<16xi32>
// CHECK-SAME: %[[SHIFT:[a-zA-Z0-9]+]]: i32
// AIE2P-LABEL: shift_i32
// AIE2P-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<16xi32>
// AIE2P-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<16xi32>
// AIE2P-SAME: %[[SHIFT:[a-zA-Z0-9]+]]: i32
func.func @shift_i32(%lhs : vector<16xi32>, %rhs : vector<16xi32>, %shift : i32) -> vector<16xi32> {
  // CHECK: %[[STEP:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[RES:.*]] = "xllvm.intr.aie2.vshift.I512.I512"(%[[LHS]], %[[RHS]], %[[STEP]], %[[SHIFT]]) : (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
  // AIE2P: %[[STEP:.*]] = llvm.mlir.constant(0 : i32) : i32
  // AIE2P: %[[RES:.*]] = "xllvm.intr.aie2p.vshift.I512.I512"(%[[LHS]], %[[RHS]], %[[STEP]], %[[SHIFT]]) : (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
  %0 = aievec.shift %lhs, %rhs, %shift {isAcc = false} : vector<16xi32>, vector<16xi32>, i32, vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: shift_bf16
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<32xbf16>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<32xbf16>
// CHECK-SAME: %[[SHIFT:[a-zA-Z0-9]+]]: i32
// AIE2P-LABEL: shift_bf16
// AIE2P-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<32xbf16>
// AIE2P-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<32xbf16>
// AIE2P-SAME: %[[SHIFT:[a-zA-Z0-9]+]]: i32
func.func @shift_bf16(%lhs : vector<32xbf16>, %rhs : vector<32xbf16>, %shift : i32) -> vector<32xbf16> {
  // CHECK: %[[STEP:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[RES:.*]] = "xllvm.intr.aie2.vshift.bf512.bf512"(%[[LHS]], %[[RHS]], %[[STEP]], %[[SHIFT]]) : (vector<32xbf16>, vector<32xbf16>, i32, i32) -> vector<32xbf16>
  // AIE2P: %[[STEP:.*]] = llvm.mlir.constant(0 : i32) : i32
  // AIE2P: %[[RES:.*]] = "xllvm.intr.aie2p.vshift.bf512.bf512"(%[[LHS]], %[[RHS]], %[[STEP]], %[[SHIFT]]) : (vector<32xbf16>, vector<32xbf16>, i32, i32) -> vector<32xbf16>
  %0 = aievec.shift %lhs, %rhs, %shift {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// CHECK-LABEL: shift_i16
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<32xi16>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<32xi16>
// CHECK-SAME: %[[SHIFT:[a-zA-Z0-9]+]]: i32
// AIE2P-LABEL: shift_i16
// AIE2P-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<32xi16>
// AIE2P-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<32xi16>
// AIE2P-SAME: %[[SHIFT:[a-zA-Z0-9]+]]: i32
func.func @shift_i16(%lhs : vector<32xi16>, %rhs : vector<32xi16>, %shift : i32) -> vector<32xi16> {
  // CHECK: %[[STEP:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[LHS_BC:.*]] = llvm.bitcast %[[LHS]] : vector<32xi16> to vector<16xi32>
  // CHECK: %[[RHS_BC:.*]] = llvm.bitcast %[[RHS]] : vector<32xi16> to vector<16xi32>
  // CHECK: %[[RES:.*]] = "xllvm.intr.aie2.vshift.I512.I512"(%[[LHS_BC]], %[[RHS_BC]], %[[STEP]], %[[SHIFT]]) : (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
  // CHECK: %[[RES_BC:.*]] = llvm.bitcast %[[RES]] : vector<16xi32> to vector<32xi16>
  // AIE2P: %[[STEP:.*]] = llvm.mlir.constant(0 : i32) : i32
  // AIE2P: %[[LHS_BC:.*]] = llvm.bitcast %[[LHS]] : vector<32xi16> to vector<16xi32>
  // AIE2P: %[[RHS_BC:.*]] = llvm.bitcast %[[RHS]] : vector<32xi16> to vector<16xi32>
  // AIE2P: %[[RES:.*]] = "xllvm.intr.aie2p.vshift.I512.I512"(%[[LHS_BC]], %[[RHS_BC]], %[[STEP]], %[[SHIFT]]) : (vector<16xi32>, vector<16xi32>, i32, i32) -> vector<16xi32>
  // AIE2P: %[[RES_BC:.*]] = llvm.bitcast %[[RES]] : vector<16xi32> to vector<32xi16>
  %0 = aievec.shift %lhs, %rhs, %shift {isAcc = false} : vector<32xi16>, vector<32xi16>, i32, vector<32xi16>
  return %0 : vector<32xi16>
}
