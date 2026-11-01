//===- test-replicate-shuffle-aie2p.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec='aie-target=aie2p' -cse -canonicalize | FileCheck %s

// Repetitions and broadcasts on AIE2P, through the shuffle planner.

// A 128-bit vector repeated to 512 bits: two interleaves of 128-bit lanes.
// CHECK-LABEL: @repeat_v4f32_to_v16(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t128_4x2_lo] : vector<4xi128>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t128_4x2_lo] : vector<4xi128>
// CHECK: return
func.func @repeat_v4f32_to_v16(%a: vector<4xf32>) -> vector<16xf32> {
  %r = vector.shuffle %a, %a [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xf32>, vector<4xf32>
  return %r : vector<16xf32>
}

// -----

// To 1024 bits: the 512-bit register is used twice.
// CHECK-LABEL: @repeat_v8bf16_to_v64(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t128_4x2_lo] : vector<4xi128>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t128_4x2_lo] : vector<4xi128>
// CHECK: return
func.func @repeat_v8bf16_to_v64(%a: vector<8xbf16>) -> vector<64xbf16> {
  %r = vector.shuffle %a, %a [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<8xbf16>, vector<8xbf16>
  return %r : vector<64xbf16>
}

// -----

// The same for 8-bit elements.
// CHECK-LABEL: @repeat_v16i8_to_v128(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t128_4x2_lo] : vector<4xi128>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t128_4x2_lo] : vector<4xi128>
// CHECK: return
func.func @repeat_v16i8_to_v128(%a: vector<16xi8>) -> vector<128xi8> {
  %r = vector.shuffle %a, %a [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<16xi8>, vector<16xi8>
  return %r : vector<128xi8>
}

// -----

// To 256 bits: the lower half of the planned register.
// CHECK-LABEL: @repeat_v8bf16_to_v16(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t128_2x4_lo] : vector<4xi128>
// CHECK: return
func.func @repeat_v8bf16_to_v16(%a: vector<8xbf16>) -> vector<16xbf16> {
  %r = vector.shuffle %a, %a [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<8xbf16>, vector<8xbf16>
  return %r : vector<16xbf16>
}

// -----

// A poison lane may hold the repeated value too.
// CHECK-LABEL: @repeat_with_poison(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t128_4x2_lo] : vector<4xi128>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t128_4x2_lo] : vector<4xi128>
// CHECK: return
func.func @repeat_with_poison(%a: vector<4xi32>) -> vector<16xi32> {
  %r = vector.shuffle %a, %a [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, -1] : vector<4xi32>, vector<4xi32>
  return %r : vector<16xi32>
}

// -----

// A 256-bit vector repeated: LLVM copies the register half, as cheap as a
// shuffle, so the shuffle stays.
// CHECK-LABEL: @repeat_256_to_512(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.
// CHECK: return
func.func @repeat_256_to_512(%a: vector<16xbf16>) -> vector<32xbf16> {
  %r = vector.shuffle %a, %a [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<16xbf16>, vector<16xbf16>
  return %r : vector<32xbf16>
}

// -----

// The same register repeated four times stays too.
// CHECK-LABEL: @repeat_256_to_2048(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.
// CHECK: return
func.func @repeat_256_to_2048(%a: vector<8xf32>) -> vector<64xf32> {
  %r = vector.shuffle %a, %a [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<8xf32>, vector<8xf32>
  return %r : vector<64xf32>
}

// -----

// Almost a repetition: two lanes swapped first.
// CHECK-LABEL: @not_repeat(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.
// CHECK: return
func.func @not_repeat(%a: vector<8xbf16>) -> vector<32xbf16> {
  %r = vector.shuffle %a, %a [1, 0, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<8xbf16>, vector<8xbf16>
  return %r : vector<32xbf16>
}

// -----

// A broadcast that adds leading dims becomes a repeating shuffle, which LLVM
// lowers by register copies.
// CHECK-LABEL: @broadcast_rows(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.
// CHECK: return
func.func @broadcast_rows(%a: vector<8xf32>) -> vector<8x8xf32> {
  %r = vector.broadcast %a : vector<8xf32> to vector<8x8xf32>
  return %r : vector<8x8xf32>
}

// -----

// A 64-bit vector broadcast over rows: the repetition is planned.
// CHECK-LABEL: @broadcast_rows_64bit(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_lo] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_lo] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_lo] : vector<8xi64>
// CHECK: return
func.func @broadcast_rows_64bit(%a: vector<2xi32>) -> vector<8x2xi32> {
  %r = vector.broadcast %a : vector<2xi32> to vector<8x2xi32>
  return %r : vector<8x2xi32>
}

// -----

// Two values broadcast over 8 rows and read transposed: the broadcast and the
// transpose become one gather, each value filling 8 lanes.
// CHECK-LABEL: @broadcast_transposed(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_lo] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_lo] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_lo] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t32_8x2] : vector<16xi32>
// CHECK: return
func.func @broadcast_transposed(%v: vector<2xi32>) -> vector<16xi32> {
  %b = vector.broadcast %v : vector<2xi32> to vector<8x2xi32>
  %f = vector.shape_cast %b : vector<8x2xi32> to vector<16xi32>
  %t = vector.shuffle %f, %f [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi32>, vector<16xi32>
  return %t : vector<16xi32>
}

// -----

// Eight values: each step of the plan doubles every value.
// CHECK-LABEL: @broadcast_transposed_k8(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi] : vector<16xi32>
// CHECK: return
func.func @broadcast_transposed_k8(%v: vector<8xi32>) -> vector<64xi32> {
  %b = vector.broadcast %v : vector<8xi32> to vector<8x8xi32>
  %f = vector.shape_cast %b : vector<8x8xi32> to vector<64xi32>
  %t = vector.shuffle %f, %f [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63] : vector<64xi32>, vector<64xi32>
  return %t : vector<64xi32>
}
