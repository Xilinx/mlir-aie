//===- test-exp2-aie2p.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

func.func @v16bf16_exp2(%arg0 : vector<16xbf16>) -> vector<16xbf16> {
  %0 = aievec.exp2 %arg0 : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// CHECK-LABEL: @v16bf16_exp2
// CHECK-SAME: %[[ARG0:.*]]: vector<16xbf16>
// CHECK: %[[F32:.*]] = "xllvm.intr.aie2p.v16bf16.to.v16accfloat"(%[[ARG0]]) : (vector<16xbf16>) -> vector<16xf32>
// CHECK: %[[EXP2:.*]] = "xllvm.intr.aie2p.exp2"(%[[F32]]) : (vector<16xf32>) -> vector<16xbf16>
// CHECK: return %[[EXP2]] : vector<16xbf16>

// -----

func.func @v32bf16_exp2(%arg0 : vector<32xbf16>) -> vector<32xbf16> {
  %0 = aievec.exp2 %arg0 : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL: @v32bf16_exp2
// CHECK-SAME: %[[ARG0:.*]]: vector<32xbf16>
// CHECK: %[[LOWER_BF16:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xbf16>, vector<32xbf16>
// CHECK: %[[UPPER_BF16:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xbf16>, vector<32xbf16>
// CHECK: %[[LOWER_F32:.*]] = "xllvm.intr.aie2p.v16bf16.to.v16accfloat"(%[[LOWER_BF16]]) : (vector<16xbf16>) -> vector<16xf32>
// CHECK: %[[UPPER_F32:.*]] = "xllvm.intr.aie2p.v16bf16.to.v16accfloat"(%[[UPPER_BF16]]) : (vector<16xbf16>) -> vector<16xf32>
// CHECK: %[[EXP20:.*]] = "xllvm.intr.aie2p.exp2"(%[[LOWER_F32]]) : (vector<16xf32>) -> vector<16xbf16>
// CHECK: %[[EXP21:.*]] = "xllvm.intr.aie2p.exp2"(%[[UPPER_F32]]) : (vector<16xf32>) -> vector<16xbf16>
// CHECK: %[[RESULT:.*]] = vector.shuffle %[[EXP20]], %[[EXP21]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xbf16>, vector<16xbf16>
// CHECK: return %[[RESULT]] : vector<32xbf16>

// -----

func.func @v16f32_exp2(%arg0 : vector<16xf32>) -> vector<16xf32> {
  %0 = aievec.exp2 %arg0 : vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @v16f32_exp2
// CHECK-SAME: %[[ARG0:.*]]: vector<16xf32>
// CHECK: %[[EXP2:.*]] = "xllvm.intr.aie2p.exp2"(%[[ARG0]]) : (vector<16xf32>) -> vector<16xbf16>
// CHECK: %[[F32:.*]] = "xllvm.intr.aie2p.v16bf16.to.v16accfloat"(%[[EXP2]]) : (vector<16xbf16>) -> vector<16xf32>
// CHECK: return %[[F32]] : vector<16xf32>

// -----

func.func @v32f32_exp2(%arg0 : vector<32xf32>) -> vector<32xf32> {
  %0 = aievec.exp2 %arg0 : vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @v32f32_exp2
// CHECK-SAME: %[[ARG0:.*]]: vector<32xf32>
// CHECK: %[[LOWER:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xf32>, vector<32xf32>
// CHECK: %[[UPPER:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xf32>, vector<32xf32>
// CHECK: %[[EXP20:.*]] = "xllvm.intr.aie2p.exp2"(%[[LOWER]]) : (vector<16xf32>) -> vector<16xbf16>
// CHECK: %[[EXP21:.*]] = "xllvm.intr.aie2p.exp2"(%[[UPPER]]) : (vector<16xf32>) -> vector<16xbf16>
// CHECK: %[[F32_0:.*]] = "xllvm.intr.aie2p.v16bf16.to.v16accfloat"(%[[EXP20]]) : (vector<16xbf16>) -> vector<16xf32>
// CHECK: %[[F32_1:.*]] = "xllvm.intr.aie2p.v16bf16.to.v16accfloat"(%[[EXP21]]) : (vector<16xbf16>) -> vector<16xf32>
// CHECK: %[[RESULT:.*]] = vector.shuffle %[[F32_0]], %[[F32_1]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: return %[[RESULT]] : vector<32xf32>
