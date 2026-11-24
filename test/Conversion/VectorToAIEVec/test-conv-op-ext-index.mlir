//===- test-conv-op-ext-index.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2 shift=10" | FileCheck %s

// The top half of the signal (aievec.ext index 1) starts at element 16 of
// the source, so with a shift to element 15 it forms one convolution.

// CHECK-LABEL: func.func @conv_top_half
//       CHECK:   %[[C30:.*]] = arith.constant 30 : i32
//       CHECK:   %[[V:.*]] = aievec.upd
//   CHECK-NOT:   arith.muli
//       CHECK:   %[[S:.*]] = aievec.shift %[[V]], %[[V]], %[[C30]] {isAcc = false} : vector<32xi16>
//       CHECK:   aievec.mul_conv %[[S]], %{{.*}} {M = 16 : i32, N = 4 : i32}
//   CHECK-NOT:   arith.muli
func.func @conv_top_half(%arg0: memref<64xi16>, %arg1: memref<16xi16>, %arg2: memref<16xi16>) {
  %c0 = arith.constant 0 : index
  %c30_i32 = arith.constant 30 : i32
  %0 = aievec.upd %arg0[%c0] {index = 0 : i8, offset = 0 : i32} : memref<64xi16>, vector<32xi16>
  %sbh = aievec.ext %0 {index = 0 : i8} : vector<32xi16>, vector<16xi16>
  %sth = aievec.ext %0 {index = 1 : i8} : vector<32xi16>, vector<16xi16>
  %1 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<16xi16>, vector<16xi16>
  %s15 = aievec.shift %sbh, %sth, %c30_i32 {isAcc = false} : vector<16xi16>, vector<16xi16>, i32, vector<16xi16>
  %b0 = aievec.broadcast %1 {idx = 0 : i8} : vector<16xi16>, vector<16xi16>
  %m0 = arith.muli %s15, %b0 : vector<16xi16>
  %b1 = aievec.broadcast %1 {idx = 1 : i8} : vector<16xi16>, vector<16xi16>
  %m1 = arith.muli %sth, %b1 : vector<16xi16>
  %r = arith.addi %m0, %m1 : vector<16xi16>
  vector.transfer_write %r, %arg2[%c0] {in_bounds = [true]} : vector<16xi16>, memref<16xi16>
  return
}

// The top half at element 16, then a shift to element 1: not consecutive, so
// not one convolution from element 0.

// CHECK-LABEL: func.func @conv_top_half_not_consecutive
//       CHECK:   %[[C32:.*]] = arith.constant 32 : i32
//       CHECK:   %[[V:.*]] = aievec.upd
//       CHECK:   arith.muli
//       CHECK:   %[[S:.*]] = aievec.shift %[[V]], %[[V]], %[[C32]] {isAcc = false} : vector<32xi16>
//       CHECK:   aievec.fma_conv %[[S]]
func.func @conv_top_half_not_consecutive(%arg0: memref<64xi16>, %arg1: memref<16xi16>, %arg2: memref<16xi16>) {
  %c0 = arith.constant 0 : index
  %c2_i32 = arith.constant 2 : i32
  %0 = aievec.upd %arg0[%c0] {index = 0 : i8, offset = 0 : i32} : memref<64xi16>, vector<32xi16>
  %sbh = aievec.ext %0 {index = 0 : i8} : vector<32xi16>, vector<16xi16>
  %sth = aievec.ext %0 {index = 1 : i8} : vector<32xi16>, vector<16xi16>
  %1 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : i32} : memref<16xi16>, vector<16xi16>
  %b0 = aievec.broadcast %1 {idx = 0 : i8} : vector<16xi16>, vector<16xi16>
  %m0 = arith.muli %sth, %b0 : vector<16xi16>
  %s1 = aievec.shift %sbh, %sth, %c2_i32 {isAcc = false} : vector<16xi16>, vector<16xi16>, i32, vector<16xi16>
  %b1 = aievec.broadcast %1 {idx = 1 : i8} : vector<16xi16>, vector<16xi16>
  %m1 = arith.muli %s1, %b1 : vector<16xi16>
  %r = arith.addi %m0, %m1 : vector<16xi16>
  vector.transfer_write %r, %arg2[%c0] {in_bounds = [true]} : vector<16xi16>, memref<16xi16>
  return
}
