//===- test-neg-aie2p.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

// CHECK-LABEL: neg_v16f32
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: vector<16xf32>
func.func @neg_v16f32(%src : vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[SHUF0:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] : vector<16xf32>, vector<16xf32>
  // CHECK: %[[CONF:.*]] = llvm.mlir.constant(60 : i32) : i32
  // CHECK: %[[NEG:.*]] = "xllvm.intr.aie2p.ACC2048.accfloat.neg.conf"(%[[SHUF0]], %[[CONF]]) : (vector<64xf32>, i32) -> vector<64xf32>
  // CHECK: %[[SHUF1:.*]] = vector.shuffle %[[NEG]], %[[NEG]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<64xf32>, vector<64xf32>
  // CHECK: return %[[SHUF1]] : vector<16xf32>
  %0 = aievec.neg %src : vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: neg_v32f32
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: vector<32xf32>
func.func @neg_v32f32(%src : vector<32xf32>) -> vector<32xf32> {
  // CHECK: %[[SHUF0:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] : vector<32xf32>, vector<32xf32>
  // CHECK: %[[CONF:.*]] = llvm.mlir.constant(60 : i32) : i32
  // CHECK: %[[NEG:.*]] = "xllvm.intr.aie2p.ACC2048.accfloat.neg.conf"(%[[SHUF0]], %[[CONF]]) : (vector<64xf32>, i32) -> vector<64xf32>
  // CHECK: %[[SHUF1:.*]] = vector.shuffle %[[NEG]], %[[NEG]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<64xf32>, vector<64xf32>
  // CHECK: return %[[SHUF1]] : vector<32xf32>
  %0 = aievec.neg %src : vector<32xf32>
  return %0 : vector<32xf32>
}
