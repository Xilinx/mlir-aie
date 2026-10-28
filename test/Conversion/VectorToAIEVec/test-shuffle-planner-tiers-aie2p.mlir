//===- test-shuffle-planner-tiers-aie2p.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: asserts
// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec='aie-target=aie2p' -debug-only=aie2p-shuffle-planner -o /dev/null 2>&1 | FileCheck %s

// Which search of the shuffle planner finds which plan.

// Moves of whole register halves are left to LLVM without a search.
// CHECK-NOT: planning
func.func @pad(%a: vector<8xf32>) -> vector<16xf32> {
  %r = vector.shuffle %a, %a [0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1] : vector<8xf32>, vector<8xf32>
  return %r : vector<16xf32>
}

// -----

// One mode: the per-register search finds it, so the round search over all
// modes is skipped.
// CHECK: planning {{.*}}[0, 2, 4,
// CHECK-NEXT: LLVM estimate: 62
// CHECK-NEXT: per-register search: 1 instructions, modes t16_32x2_lo
// CHECK-NOT: round search:
// CHECK: lowered with 1 instructions, modes t16_32x2_lo
func.func @deinterleave(%a: vector<64xbf16>) -> vector<32xbf16> {
  %r = vector.shuffle %a, %a [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62] : vector<64xbf16>, vector<64xbf16>
  return %r : vector<32xbf16>
}

// -----

// Both searches find 8 shuffles with four modes; plans with one lo/hi pair
// need 12.
// CHECK: planning {{.*}}[0, 8, 16, 24,
// CHECK-NEXT: LLVM estimate: 112
// CHECK-NEXT: per-register search: 8 instructions
// CHECK-NEXT: round search: 8 instructions
// CHECK-NEXT: round search with one pair: 12 instructions, modes t32_16x2_lo t32_16x2_hi
// CHECK: lowered with 8 instructions
func.func @transpose_f32(%a: vector<64xf32>) -> vector<64xf32> {
  %r = vector.shuffle %a, %a [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63] : vector<64xf32>, vector<64xf32>
  return %r : vector<64xf32>
}

// -----

// The per-register search reuses the registers built for earlier result
// registers: 7 shuffles for 4 registers.
// CHECK: planning {{.*}}[0, 0, 0, 0, 0, 0, 0, 0, 1,
// CHECK-NEXT: LLVM estimate: 72
// CHECK-NEXT: per-register search: 7 instructions, modes t32_2x16_lo t32_2x16_hi
// CHECK: lowered with 7 instructions, modes t32_2x16_lo t32_2x16_hi
func.func @row_splats(%a: vector<8xf32>) -> vector<64xf32> {
  %r = vector.shuffle %a, %a [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7] : vector<8xf32>, vector<8xf32>
  return %r : vector<64xf32>
}

// -----

// No plan: nothing is lowered.
// CHECK: planning {{.*}}[4, 11, 10,
// CHECK-NOT: search
// CHECK-NOT: lowered
func.func @no_plan(%a: vector<16xf32>) -> vector<16xf32> {
  %r = vector.shuffle %a, %a [4, 11, 10, 13, 12, 3, 6, 0, 1, 15, 14, 5, 2, 8, 9, 7] : vector<16xf32>, vector<16xf32>
  return %r : vector<16xf32>
}
