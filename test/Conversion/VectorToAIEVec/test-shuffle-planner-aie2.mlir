//===- test-shuffle-planner-aie2.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec='aie-target=aie2' | FileCheck %s

// The shuffle planner is for AIE2P only: on AIE2, shuffles and broadcasts are
// left as they are and 512-bit transposes keep their single mode.

// CHECK-LABEL: @transpose_mask_f32(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.shuffle
func.func @transpose_mask_f32(%a: vector<64xf32>) -> vector<64xf32> {
  %r = vector.shuffle %a, %a [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63] : vector<64xf32>, vector<64xf32>
  return %r : vector<64xf32>
}

// -----

// CHECK-LABEL: @repeat_v4i32(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.shuffle
func.func @repeat_v4i32(%a: vector<4xi32>) -> vector<16xi32> {
  %r = vector.shuffle %a, %a [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
  return %r : vector<16xi32>
}

// -----

// CHECK-LABEL: @interleave(
// CHECK: vector.interleave
// CHECK-NOT: aievec.shuffle
func.func @interleave(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<64xbf16> {
  %r = vector.interleave %a, %b : vector<32xbf16> -> vector<64xbf16>
  return %r : vector<64xbf16>
}

// -----

// CHECK-LABEL: @transpose_i8(
// CHECK: aievec.shuffle %{{.*}} [t8_8x8] : vector<64xi8>
// CHECK-NOT: vector.shuffle
func.func @transpose_i8(%a: vector<8x8xi8>) -> vector<8x8xi8> {
  %r = vector.transpose %a, [1, 0] : vector<8x8xi8> to vector<8x8xi8>
  return %r : vector<8x8xi8>
}

// -----

// CHECK-LABEL: @broadcast_rows(
// CHECK-NOT: vector.shuffle
// CHECK: vector.from_elements
func.func @broadcast_rows(%a: vector<8xf32>) -> vector<8x8xf32> {
  %r = vector.broadcast %a : vector<8xf32> to vector<8x8xf32>
  return %r : vector<8x8xf32>
}
