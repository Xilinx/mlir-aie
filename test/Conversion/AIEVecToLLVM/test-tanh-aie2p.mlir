//===- test-tanh-aie2p.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

func.func @v16bf16_tanh(%arg0 : vector<16xbf16>) -> vector<16xbf16> {
  %0 = aievec.tanh %arg0 : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// CHECK-LABEL: @v16bf16_tanh
// CHECK-SAME: %[[ARG0:.*]]: vector<16xbf16>
// CHECK: %[[F32:.*]] = "xllvm.intr.aie2p.v16bf16.to.v16accfloat"(%[[ARG0]]) : (vector<16xbf16>) -> vector<16xf32>
// CHECK: %[[TANH:.*]] = "xllvm.intr.aie2p.tanh"(%[[F32]]) : (vector<16xf32>) -> vector<16xbf16>
// CHECK: return %[[TANH]] : vector<16xbf16>

// -----

func.func @v32bf16_tanh(%arg0 : vector<32xbf16>) -> vector<32xbf16> {
  %0 = aievec.tanh %arg0 : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL: @v32bf16_tanh
// CHECK-SAME: %[[ARG0:.*]]: vector<32xbf16>
// CHECK: %[[LOWER_BF16:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xbf16>, vector<32xbf16>
// CHECK: %[[UPPER_BF16:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xbf16>, vector<32xbf16>
// CHECK: %[[LOWER_F32:.*]] = "xllvm.intr.aie2p.v16bf16.to.v16accfloat"(%[[LOWER_BF16]]) : (vector<16xbf16>) -> vector<16xf32>
// CHECK: %[[UPPER_F32:.*]] = "xllvm.intr.aie2p.v16bf16.to.v16accfloat"(%[[UPPER_BF16]]) : (vector<16xbf16>) -> vector<16xf32>
// CHECK: %[[TANH0:.*]] = "xllvm.intr.aie2p.tanh"(%[[LOWER_F32]]) : (vector<16xf32>) -> vector<16xbf16>
// CHECK: %[[TANH1:.*]] = "xllvm.intr.aie2p.tanh"(%[[UPPER_F32]]) : (vector<16xf32>) -> vector<16xbf16>
// CHECK: %[[RESULT:.*]] = vector.shuffle %[[TANH0]], %[[TANH1]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xbf16>, vector<16xbf16>
// CHECK: return %[[RESULT]] : vector<32xbf16>
