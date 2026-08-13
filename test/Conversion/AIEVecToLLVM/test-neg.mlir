//===- test-neg.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie2" | FileCheck %s

// CHECK-LABEL: neg_v16f32
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: vector<16xf32>
func.func @neg_v16f32(%src : vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[CONF:.*]] = llvm.mlir.constant(28 : i32) : i32
  // CHECK: %[[BC0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xf32> to vector<8xi64>
  // CHECK: %[[NEG:.*]] = "xllvm.intr.aie2.ACC512.accfloat.neg.conf"(%[[BC0]], %[[CONF]]) : (vector<8xi64>, i32) -> vector<8xi64>
  // CHECK: %[[BC1:.*]] = llvm.bitcast %[[NEG]] : vector<8xi64> to vector<16xf32>
  // CHECK: return %[[BC1]] : vector<16xf32>
  %0 = aievec.neg %src : vector<16xf32>
  return %0 : vector<16xf32>
}
