//===- test-add_elem-i32-aie2p.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie2p" -split-input-file | FileCheck %s

// i32 aievec.add_elem on AIE2p is the 2048-bit accumulator's integer add, its
// 64 lanes carried as 32 of i64. Fewer lanes are padded with poison, and the
// low lanes kept.

// CHECK-LABEL: add_elem_v32i32
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: vector<32xi32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: vector<32xi32>
// CHECK: %[[BC0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xi32> to vector<16xi64>
// CHECK: %[[BC1:.*]] = llvm.bitcast %[[ARG1]] : vector<32xi32> to vector<16xi64>
// CHECK: %[[PAD0:.*]] = vector.shuffle %[[BC0]], %[[BC0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, {{(-1, )+}}-1] : vector<16xi64>, vector<16xi64>
// CHECK: %[[PAD1:.*]] = vector.shuffle %[[BC1]], %[[BC1]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, {{(-1, )+}}-1] : vector<16xi64>, vector<16xi64>
// CHECK: %[[CONF:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.aie2p.ACC2048.add.conf"(%[[PAD0]], %[[PAD1]], %[[CONF]]) : (vector<32xi64>, vector<32xi64>, i32) -> vector<32xi64>
// CHECK: %[[LOW:.*]] = vector.shuffle %[[R]], %[[R]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xi64>, vector<32xi64>
// CHECK: %[[RES:.*]] = llvm.bitcast %[[LOW]] : vector<16xi64> to vector<32xi32>
// CHECK: return %[[RES]] : vector<32xi32>
func.func @add_elem_v32i32(%lhs : vector<32xi32>, %rhs : vector<32xi32>) -> vector<32xi32> {
  %0 = aievec.add_elem %lhs, %rhs : vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

// CHECK-LABEL: add_elem_v64i32
// CHECK: llvm.bitcast %{{.*}} : vector<64xi32> to vector<32xi64>
// CHECK-NOT: vector.shuffle
// CHECK: "xllvm.intr.aie2p.ACC2048.add.conf"
// CHECK-NOT: vector.shuffle
// CHECK: llvm.bitcast %{{.*}} : vector<32xi64> to vector<64xi32>
func.func @add_elem_v64i32(%lhs : vector<64xi32>, %rhs : vector<64xi32>) -> vector<64xi32> {
  %0 = aievec.add_elem %lhs, %rhs : vector<64xi32>
  return %0 : vector<64xi32>
}
