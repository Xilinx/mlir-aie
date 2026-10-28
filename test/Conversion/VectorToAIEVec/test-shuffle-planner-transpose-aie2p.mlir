//===- test-shuffle-planner-transpose-aie2p.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec='aie-target=aie2p' -cse -canonicalize | FileCheck %s

// vector.transpose on AIE2P becomes a shuffle of the flattened vector, which
// the shuffle planner lowers to vshuffle modes or leaves to LLVM.

// One 512-bit transpose: one mode.
// CHECK-LABEL: @t_i8_8x8(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t8_8x8] : vector<64xi8>
// CHECK: return
func.func @t_i8_8x8(%a: vector<8x8xi8>) -> vector<8x8xi8> {
  %r = vector.transpose %a, [1, 0] : vector<8x8xi8> to vector<8x8xi8>
  return %r : vector<8x8xi8>
}

// -----

// 2048 bits of 8-bit elements.
// CHECK-LABEL: @t_i8_16x16(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t8_16x4] : vector<64xi8>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t8_16x4] : vector<64xi8>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t8_16x4] : vector<64xi8>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t8_16x4] : vector<64xi8>
// CHECK: return
func.func @t_i8_16x16(%a: vector<16x16xi8>) -> vector<16x16xi8> {
  %r = vector.transpose %a, [1, 0] : vector<16x16xi8> to vector<16x16xi8>
  return %r : vector<16x16xi8>
}

// -----

// 32x8 i8 transpose: 12 shuffles.
// CHECK-LABEL: @t_i8_32x8(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t8_64x2_lo] : vector<64xi8>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t8_64x2_lo] : vector<64xi8>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t8_64x2_lo] : vector<64xi8>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t8_64x2_hi] : vector<64xi8>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t8_64x2_hi] : vector<64xi8>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t8_64x2_lo] : vector<64xi8>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t8_64x2_lo] : vector<64xi8>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t8_64x2_hi] : vector<64xi8>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t8_64x2_hi] : vector<64xi8>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t8_64x2_lo] : vector<64xi8>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t8_64x2_hi] : vector<64xi8>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t8_64x2_hi] : vector<64xi8>
// CHECK: return
func.func @t_i8_32x8(%a: vector<32x8xi8>) -> vector<8x32xi8> {
  %r = vector.transpose %a, [1, 0] : vector<32x8xi8> to vector<8x32xi8>
  return %r : vector<8x32xi8>
}

// -----

// The 8x8 16-bit transpose across two registers (AIE2P modes).
// CHECK-LABEL: @t_bf16_8x8(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_8x8_lo] : vector<32xbf16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_8x8_hi] : vector<32xbf16>
// CHECK: return
func.func @t_bf16_8x8(%a: vector<8x8xbf16>) -> vector<8x8xbf16> {
  %r = vector.transpose %a, [1, 0] : vector<8x8xbf16> to vector<8x8xbf16>
  return %r : vector<8x8xbf16>
}

// -----

// 4x8 bf16 transpose: one shuffle.
// CHECK-LABEL: @t_bf16_4x8(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t16_4x8] : vector<32xbf16>
// CHECK: return
func.func @t_bf16_4x8(%a: vector<4x8xbf16>) -> vector<8x4xbf16> {
  %r = vector.transpose %a, [1, 0] : vector<4x8xbf16> to vector<8x4xbf16>
  return %r : vector<8x4xbf16>
}

// -----

// 16x16 bf16 transpose: 24 shuffles.
// CHECK-LABEL: @t_bf16_16x16(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_16x4_lo] : vector<32xbf16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_16x4_lo] : vector<32xbf16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_16x4_lo] : vector<32xbf16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_16x4_lo] : vector<32xbf16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_16x4_hi] : vector<32xbf16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_16x4_hi] : vector<32xbf16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_16x4_hi] : vector<32xbf16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_16x4_hi] : vector<32xbf16>
// CHECK: return
func.func @t_bf16_16x16(%a: vector<16x16xbf16>) -> vector<16x16xbf16> {
  %r = vector.transpose %a, [1, 0] : vector<16x16xbf16> to vector<16x16xbf16>
  return %r : vector<16x16xbf16>
}

// -----

// 8x16 i16 transpose: 8 shuffles.
// CHECK-LABEL: @t_i16_8x16(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_lo] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_lo] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_8x8_lo] : vector<32xi16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_hi] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_hi] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_8x8_lo] : vector<32xi16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_8x8_hi] : vector<32xi16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_8x8_hi] : vector<32xi16>
// CHECK: return
func.func @t_i16_8x16(%a: vector<8x16xi16>) -> vector<16x8xi16> {
  %r = vector.transpose %a, [1, 0] : vector<8x16xi16> to vector<16x8xi16>
  return %r : vector<16x8xi16>
}

// -----

// 4x4 f32 transpose: one shuffle.
// CHECK-LABEL: @t_f32_4x4(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t32_4x4] : vector<16xf32>
// CHECK: return
func.func @t_f32_4x4(%a: vector<4x4xf32>) -> vector<4x4xf32> {
  %r = vector.transpose %a, [1, 0] : vector<4x4xf32> to vector<4x4xf32>
  return %r : vector<4x4xf32>
}

// -----

// 8x8 32-bit: 64-bit pairs first, then 32-bit lanes; 8 shuffles.
// CHECK-LABEL: @t_f32_8x8(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_lo] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_lo] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_8x4_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_hi] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_hi] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_8x4_lo] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_8x4_hi] : vector<16xf32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_8x4_hi] : vector<16xf32>
// CHECK: return
func.func @t_f32_8x8(%a: vector<8x8xf32>) -> vector<8x8xf32> {
  %r = vector.transpose %a, [1, 0] : vector<8x8xf32> to vector<8x8xf32>
  return %r : vector<8x8xf32>
}

// -----

// 16x8 i32 transpose: 24 shuffles.
// CHECK-LABEL: @t_i32_16x8(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_lo] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t32_16x2_hi] : vector<16xi32>
// CHECK: return
func.func @t_i32_16x8(%a: vector<16x8xi32>) -> vector<8x16xi32> {
  %r = vector.transpose %a, [1, 0] : vector<16x8xi32> to vector<8x16xi32>
  return %r : vector<8x16xi32>
}

// -----

// 2x8 f32 transpose: one shuffle.
// CHECK-LABEL: @t_f32_2x8(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}} [t32_2x8] : vector<16xf32>
// CHECK: return
func.func @t_f32_2x8(%a: vector<2x8xf32>) -> vector<8x2xf32> {
  %r = vector.transpose %a, [1, 0] : vector<2x8xf32> to vector<8x2xf32>
  return %r : vector<8x2xf32>
}

// -----

// A transpose of the two inner dims of a rank-3 vector.
// CHECK-LABEL: @t_rank3(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_8x8_lo] : vector<32xbf16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_8x8_hi] : vector<32xbf16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_8x8_lo] : vector<32xbf16>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t16_8x8_hi] : vector<32xbf16>
// CHECK: return
func.func @t_rank3(%a: vector<2x8x8xbf16>) -> vector<2x8x8xbf16> {
  %r = vector.transpose %a, [0, 2, 1] : vector<2x8x8xbf16> to vector<2x8x8xbf16>
  return %r : vector<2x8x8xbf16>
}

// -----

// 15 lanes: no plan, the shuffle is left to LLVM.
// CHECK-LABEL: @t_f32_3x5(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.
// CHECK: return
func.func @t_f32_3x5(%a: vector<3x5xf32>) -> vector<5x3xf32> {
  %r = vector.transpose %a, [1, 0] : vector<3x5xf32> to vector<5x3xf32>
  return %r : vector<5x3xf32>
}

// -----

// 4x4 i64 transpose: 4 shuffles.
// CHECK-LABEL: @t_i64_4x4(
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_lo] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_hi] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_lo] : vector<8xi64>
// CHECK: %{{.*}} = aievec.shuffle %{{.*}}, %{{.*}} [t64_8x2_hi] : vector<8xi64>
// CHECK: return
func.func @t_i64_4x4(%a: vector<4x4xi64>) -> vector<4x4xi64> {
  %r = vector.transpose %a, [1, 0] : vector<4x4xi64> to vector<4x4xi64>
  return %r : vector<4x4xi64>
}

// -----

// Index elements: the transpose becomes a shuffle for LLVM.
// CHECK-LABEL: @t_index(
// CHECK: vector.shuffle
// CHECK-NOT: aievec.
// CHECK: return
func.func @t_index(%a: vector<4x4xindex>) -> vector<4x4xindex> {
  %r = vector.transpose %a, [1, 0] : vector<4x4xindex> to vector<4x4xindex>
  return %r : vector<4x4xindex>
}
