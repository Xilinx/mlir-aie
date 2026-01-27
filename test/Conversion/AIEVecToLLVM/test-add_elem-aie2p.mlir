//===- test-add_elem-aie2p.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

// CHECK-LABEL: add_elem_flat_v32f32
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: vector<32xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: vector<32xf32>
func.func @add_elem_flat_v32f32(%lhs : vector<32xf32>, %rhs : vector<32xf32>) -> vector<32xf32> {
  // CHECK: %[[SHUF0:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] : vector<32xf32>, vector<32xf32>
  // CHECK: %[[SHUF1:.*]] = vector.shuffle %[[ARG1]], %[[ARG1]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] : vector<32xf32>, vector<32xf32>
  // CHECK: %[[CONF:.*]] = llvm.mlir.constant(60 : i32) : i32
  // CHECK: %[[ADD:.*]] = "xllvm.intr.aie2p.ACC2048.accfloat.add.conf"(%[[SHUF0]], %[[SHUF1]], %[[CONF]]) : (vector<64xf32>, vector<64xf32>, i32) -> vector<64xf32>
  // CHECK: %[[SHUF2:.*]] = vector.shuffle %[[ADD]], %[[ADD]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<64xf32>, vector<64xf32>
  // CHECK: return %[[SHUF2]] : vector<32xf32>
  %0 = aievec.add_elem %lhs, %rhs : vector<32xf32>
  return %0 : vector<32xf32>
}
