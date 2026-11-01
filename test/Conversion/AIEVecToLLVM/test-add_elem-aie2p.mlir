//===- test-add_elem-aie2p.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie2p" -split-input-file | FileCheck %s

// f32 aievec.add_elem on AIE2p is the 64-lane ACC2048 accfloat intrinsic. Fewer
// lanes are padded with poison through i64 pairs, and the low lanes kept.

// CHECK-LABEL: add_elem_v8f32
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: vector<8xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: vector<8xf32>
// CHECK: %[[BC0:.*]] = llvm.bitcast %[[ARG0]] : vector<8xf32> to vector<4xi64>
// CHECK: %[[BC1:.*]] = llvm.bitcast %[[ARG1]] : vector<8xf32> to vector<4xi64>
// CHECK: %[[PAD0:.*]] = vector.shuffle %[[BC0]], %[[BC0]] [0, 1, 2, 3, -1, {{(-1, )+}}-1] : vector<4xi64>, vector<4xi64>
// CHECK: %[[PAD1:.*]] = vector.shuffle %[[BC1]], %[[BC1]] [0, 1, 2, 3, -1, {{(-1, )+}}-1] : vector<4xi64>, vector<4xi64>
// CHECK: %[[ACC0:.*]] = llvm.bitcast %[[PAD0]] : vector<32xi64> to vector<64xf32>
// CHECK: %[[ACC1:.*]] = llvm.bitcast %[[PAD1]] : vector<32xi64> to vector<64xf32>
// CHECK: %[[CONF:.*]] = llvm.mlir.constant(60 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.aie2p.ACC2048.accfloat.add.conf"(%[[ACC0]], %[[ACC1]], %[[CONF]]) : (vector<64xf32>, vector<64xf32>, i32) -> vector<64xf32>
// CHECK: %[[R64:.*]] = llvm.bitcast %[[R]] : vector<64xf32> to vector<32xi64>
// CHECK: %[[LOW:.*]] = vector.shuffle %[[R64]], %[[R64]] [0, 1, 2, 3] : vector<32xi64>, vector<32xi64>
// CHECK: %[[RES:.*]] = llvm.bitcast %[[LOW]] : vector<4xi64> to vector<8xf32>
// CHECK: return %[[RES]] : vector<8xf32>
func.func @add_elem_v8f32(%lhs : vector<8xf32>, %rhs : vector<8xf32>) -> vector<8xf32> {
  %0 = aievec.add_elem %lhs, %rhs : vector<8xf32>
  return %0 : vector<8xf32>
}

// -----

// CHECK-LABEL: add_elem_v16f32
// CHECK: llvm.bitcast %{{.*}} : vector<16xf32> to vector<8xi64>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, -1, {{(-1, )+}}-1] : vector<8xi64>, vector<8xi64>
// CHECK: "xllvm.intr.aie2p.ACC2048.accfloat.add.conf"
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<32xi64>, vector<32xi64>
// CHECK: llvm.bitcast %{{.*}} : vector<8xi64> to vector<16xf32>
func.func @add_elem_v16f32(%lhs : vector<16xf32>, %rhs : vector<16xf32>) -> vector<16xf32> {
  %0 = aievec.add_elem %lhs, %rhs : vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// CHECK-LABEL: add_elem_v32f32
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: vector<32xf32>
// CHECK: %[[BC0:.*]] = llvm.bitcast %[[ARG0]] : vector<32xf32> to vector<16xi64>
// CHECK: vector.shuffle %[[BC0]], %[[BC0]] [0, 1, 2, {{([0-9]+, )+}}15, -1, {{(-1, )+}}-1] : vector<16xi64>, vector<16xi64>
// CHECK: %[[R:.*]] = "xllvm.intr.aie2p.ACC2048.accfloat.add.conf"
// CHECK: %[[R64:.*]] = llvm.bitcast %[[R]] : vector<64xf32> to vector<32xi64>
// CHECK: %[[LOW:.*]] = vector.shuffle %[[R64]], %[[R64]] [0, 1, 2, {{([0-9]+, )+}}15] : vector<32xi64>, vector<32xi64>
// CHECK: %[[RES:.*]] = llvm.bitcast %[[LOW]] : vector<16xi64> to vector<32xf32>
// CHECK: return %[[RES]] : vector<32xf32>
func.func @add_elem_v32f32(%lhs : vector<32xf32>, %rhs : vector<32xf32>) -> vector<32xf32> {
  %0 = aievec.add_elem %lhs, %rhs : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

// At 64 lanes the intrinsic is used as is.
// CHECK-LABEL: add_elem_v64f32
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: vector<64xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: vector<64xf32>
// CHECK-NOT: vector.shuffle
// CHECK: %[[CONF:.*]] = llvm.mlir.constant(60 : i32) : i32
// CHECK: %[[R:.*]] = "xllvm.intr.aie2p.ACC2048.accfloat.add.conf"(%[[ARG0]], %[[ARG1]], %[[CONF]]) : (vector<64xf32>, vector<64xf32>, i32) -> vector<64xf32>
// CHECK-NOT: vector.shuffle
// CHECK: return %[[R]] : vector<64xf32>
func.func @add_elem_v64f32(%lhs : vector<64xf32>, %rhs : vector<64xf32>) -> vector<64xf32> {
  %0 = aievec.add_elem %lhs, %rhs : vector<64xf32>
  return %0 : vector<64xf32>
}
