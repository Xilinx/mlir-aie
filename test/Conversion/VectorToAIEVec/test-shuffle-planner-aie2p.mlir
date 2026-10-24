//===- test-shuffle-planner-aie2p.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec='aie-target=aie2p' -cse -canonicalize | FileCheck %s

// Static shuffles lowered to vshuffle modes on AIE2P, and those left to LLVM.

// Even lanes of two registers: one deinterleave.
// CHECK-LABEL: @deinterleave_bf16(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_32x2_lo] : vector<32xbf16>
// CHECK: return
func.func @deinterleave_bf16(%a: vector<64xbf16>) -> vector<32xbf16> {
  %r = vector.shuffle %a, %a [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62] : vector<64xbf16>, vector<64xbf16>
  return %r : vector<32xbf16>
}

// -----

// Two registers interleaved: the lo and hi halves of one mode.
// CHECK-LABEL: @interleave_bf16(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_2x32_lo] : vector<32xbf16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_2x32_hi] : vector<32xbf16>
// CHECK: return
func.func @interleave_bf16(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<64xbf16> {
  %r = vector.shuffle %a, %b [0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63] : vector<32xbf16>, vector<32xbf16>
  return %r : vector<64xbf16>
}

// -----

// Reversal: 32-bit lanes reversed, then the 16-bit pairs swapped.
// CHECK-LABEL: @reverse_bf16(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t32_1x16_flip] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t16_1x2_flip] : vector<32xbf16>
// CHECK: return
func.func @reverse_bf16(%a: vector<32xbf16>) -> vector<32xbf16> {
  %r = vector.shuffle %a, %a [31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] : vector<32xbf16>, vector<32xbf16>
  return %r : vector<32xbf16>
}

// -----

// Reversal of bytes: 32-bit lanes reversed, then bytes within them.
// CHECK-LABEL: @reverse_i8(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t32_1x16_flip] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t8_1x4_flip] : vector<64xi8>
// CHECK: return
func.func @reverse_i8(%a: vector<64xi8>) -> vector<64xi8> {
  %r = vector.shuffle %a, %a [63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] : vector<64xi8>, vector<64xi8>
  return %r : vector<64xi8>
}

// -----

// Reversal of 32-bit lanes: one AIE2P mode.
// CHECK-LABEL: @reverse_f32(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t32_1x16_flip] : vector<16xf32>
// CHECK: return
func.func @reverse_f32(%a: vector<16xf32>) -> vector<16xf32> {
  %r = vector.shuffle %a, %a [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] : vector<16xf32>, vector<16xf32>
  return %r : vector<16xf32>
}

// -----

// f16 has no aievec.shuffle type of its own: the modes run on i16.
// CHECK-LABEL: @reverse_f16(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t32_1x16_flip] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t16_1x2_flip] : vector<32xi16>
// CHECK: return
func.func @reverse_f16(%a: vector<32xf16>) -> vector<32xf16> {
  %r = vector.shuffle %a, %a [31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] : vector<32xf16>, vector<32xf16>
  return %r : vector<32xf16>
}

// -----

// A repetition read transposed, so each element fills a row of 8 lanes.
// The two shuffles are planned as one gather: each step doubles every value.
// CHECK-LABEL: @row_splats(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi] : vector<16xf32>
// CHECK: return
func.func @row_splats(%a: vector<8xf32>) -> vector<64xf32> {
  %r = vector.shuffle %a, %a [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<8xf32>, vector<8xf32>
  %t = vector.shuffle %r, %r [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63] : vector<64xf32>, vector<64xf32>
  return %t : vector<64xf32>
}

// -----

// Two splats concatenated: both scalars broadcast to 512 bits and one
// shuffle takes the lower halves.
// CHECK-LABEL: @concat_splats(
// CHECK: %{{.*}} = aievec.broadcast_scalar %{{.*}} : i32, vector<16xi32>
// CHECK: %{{.*}} = aievec.broadcast_scalar %{{.*}} : i32, vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: return
func.func @concat_splats(%a: i32, %b: i32) -> vector<16xi32> {
  %sa = vector.broadcast %a : i32 to vector<8xi32>
  %sb = vector.broadcast %b : i32 to vector<8xi32>
  %r = vector.shuffle %sa, %sb [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xi32>, vector<8xi32>
  return %r : vector<16xi32>
}

// -----

// A splat and a vector concatenated. LLVM would build this a lane at a time.
// CHECK-LABEL: @concat_splat_and_vector(
// CHECK: %{{.*}} = aievec.broadcast_scalar %{{.*}} : i32, vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t256_2x2_lo] : vector<2xi256>
// CHECK: return
func.func @concat_splat_and_vector(%a: i32, %v: vector<8xi32>) -> vector<16xi32> {
  %sa = vector.broadcast %a : i32 to vector<8xi32>
  %r = vector.shuffle %sa, %v [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xi32>, vector<8xi32>
  return %r : vector<16xi32>
}

// -----

// Poison lanes may hold anything.
// CHECK-LABEL: @poison_lanes(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_32x2_lo] : vector<32xbf16>
// CHECK: return
func.func @poison_lanes(%a: vector<64xbf16>) -> vector<32xbf16> {
  %r = vector.shuffle %a, %a [-1, 2, 4, 6, 8, -1, 12, 14, 16, 18, -1, 22, 24, 26, 28, -1, 32, 34, 36, 38, -1, 42, 44, 46, 48, -1, 52, 54, 56, 58, -1, 62] : vector<64xbf16>, vector<64xbf16>
  return %r : vector<32xbf16>
}

// -----

// A 256-bit result is the lower half of a planned register.
// CHECK-LABEL: @half_result(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t32_8x2] : vector<16xf32>
// CHECK: return
func.func @half_result(%a: vector<16xf32>) -> vector<8xf32> {
  %r = vector.shuffle %a, %a [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xf32>, vector<16xf32>
  return %r : vector<8xf32>
}

// -----

// No short plan exists: left to LLVM.
// CHECK-LABEL: @no_plan(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.
// CHECK: return
func.func @no_plan(%a: vector<16xf32>) -> vector<16xf32> {
  %r = vector.shuffle %a, %a [4, 11, 10, 13, 12, 3, 6, 0, 1, 15, 14, 5, 2, 8, 9, 7] : vector<16xf32>, vector<16xf32>
  return %r : vector<16xf32>
}

// -----

// Each lane from its own position in one of two sources: a select for LLVM.
// CHECK-LABEL: @select(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.
// CHECK: return
func.func @select(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %r = vector.shuffle %a, %b [16, 1, 2, 19, 4, 5, 22, 7, 8, 25, 10, 11, 28, 13, 14, 31] : vector<16xf32>, vector<16xf32>
  return %r : vector<16xf32>
}

// -----

// A register half of the source: left to LLVM.
// CHECK-LABEL: @extract_half(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.
// CHECK: return
func.func @extract_half(%a: vector<64xi16>) -> vector<32xi16> {
  %r = vector.shuffle %a, %a [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xi16>, vector<64xi16>
  return %r : vector<32xi16>
}

// -----

// Whole registers concatenated: left to LLVM.
// CHECK-LABEL: @concat(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.
// CHECK: return
func.func @concat(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<64xbf16> {
  %r = vector.shuffle %a, %b [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xbf16>, vector<32xbf16>
  return %r : vector<64xbf16>
}

// -----

// Padding with poison: left to LLVM.
// CHECK-LABEL: @pad(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.
// CHECK: return
func.func @pad(%a: vector<8xf32>) -> vector<16xf32> {
  %r = vector.shuffle %a, %a [0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1] : vector<8xf32>, vector<8xf32>
  return %r : vector<16xf32>
}

// -----

// A rotation: LLVM shifts it.
// CHECK-LABEL: @rotate(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.
// CHECK: return
func.func @rotate(%a: vector<16xf32>) -> vector<16xf32> {
  %r = vector.shuffle %a, %a [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2] : vector<16xf32>, vector<16xf32>
  return %r : vector<16xf32>
}

// -----

// Index elements have no fixed width: left to LLVM.
// CHECK-LABEL: @index_elements(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.
// CHECK: return
func.func @index_elements(%a: vector<16xindex>) -> vector<16xindex> {
  %r = vector.shuffle %a, %a [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] : vector<16xindex>, vector<16xindex>
  return %r : vector<16xindex>
}

// -----

// 64-bit lanes swapped in pairs.
// CHECK-LABEL: @swap_pairs_i64(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_hi] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t64_4x2] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_2x8_lo] : vector<8xi64>
// CHECK: return
func.func @swap_pairs_i64(%a: vector<8xi64>) -> vector<8xi64> {
  %r = vector.shuffle %a, %a [1, 0, 3, 2, 5, 4, 7, 6] : vector<8xi64>, vector<8xi64>
  return %r : vector<8xi64>
}

// -----

// A 3x4 transpose of 12 lanes: no plan found, left to LLVM.
// CHECK-LABEL: @odd_size(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.
// CHECK: return
func.func @odd_size(%a: vector<12xf32>) -> vector<12xf32> {
  %r = vector.shuffle %a, %a [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11] : vector<12xf32>, vector<12xf32>
  return %r : vector<12xf32>
}

// -----

// vector.deinterleave: both results planned.
// CHECK-LABEL: @deinterleave_op(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_32x2_lo] : vector<32xbf16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_32x2_hi] : vector<32xbf16>
// CHECK: return
func.func @deinterleave_op(%a: vector<64xbf16>) -> (vector<32xbf16>, vector<32xbf16>) {
  %e, %o = vector.deinterleave %a : vector<64xbf16> -> vector<32xbf16>
  return %e, %o : vector<32xbf16>, vector<32xbf16>
}

// -----

// vector.interleave of two 256-bit vectors.
// CHECK-LABEL: @interleave_f32(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xf32>
// CHECK: return
func.func @interleave_f32(%a: vector<8xf32>, %b: vector<8xf32>) -> vector<16xf32> {
  %r = vector.interleave %a, %b : vector<8xf32> -> vector<16xf32>
  return %r : vector<16xf32>
}

// -----

// The 8x8 transpose has an 8-shuffle plan with four modes. It reuses the two
// modes the first plan loaded instead, for 12 shuffles.
// CHECK-LABEL: @modes_reused(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_2x16_hi] : vector<16xf32>
// CHECK: return
func.func @modes_reused(%a: vector<8xf32>, %b: vector<64xf32>) -> (vector<64xf32>, vector<64xf32>) {
  %r = vector.shuffle %a, %a [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<8xf32>, vector<8xf32>
  %t = vector.shuffle %r, %r [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63] : vector<64xf32>, vector<64xf32>
  %u = vector.shuffle %b, %b [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63] : vector<64xf32>, vector<64xf32>
  return %t, %u : vector<64xf32>, vector<64xf32>
}
