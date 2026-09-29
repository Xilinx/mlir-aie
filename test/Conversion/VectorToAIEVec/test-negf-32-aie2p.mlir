//===- test-negf-32-aie2p.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NegOpAIE2pConversion widens the accumulator to ACC2048 for 16 and 32 lanes
// alike, but ComputeNegOpPattern stopped at 16 and the AIE2P legality
// predicate agreed with it -- so a v32 negate was declared legal, nothing
// converted it, and it reached the backend as `G_FNEG <32 x s32>`.

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s

// The motivating case: a sigmoid's exp(-z) on a v32 of bf16, which takes the
// UPS/neg/SRS branch.
// CHECK-LABEL: func @neg_32_bf16
// CHECK: %[[UPS:.*]] = aievec.ups
// CHECK: %[[NEG:.*]] = aievec.neg %[[UPS]]
// CHECK: aievec.srs %[[NEG]]
// CHECK-NOT: arith.negf
func.func @neg_32_bf16(%arg0: vector<32xbf16>) -> vector<32xbf16> {
  %0 = arith.negf %arg0 : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// f32 at 32 lanes takes the cast branch instead.
// CHECK-LABEL: func @neg_32_f32
// CHECK: aievec.neg {{.*}} : vector<32xf32>
// CHECK-NOT: arith.negf
func.func @neg_32_f32(%arg0: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.negf %arg0 : vector<32xf32>
  return %0 : vector<32xf32>
}

// The 16-lane paths are unchanged.
// CHECK-LABEL: func @neg_16_bf16
// CHECK: aievec.neg
// CHECK-NOT: arith.negf
func.func @neg_16_bf16(%arg0: vector<16xbf16>) -> vector<16xbf16> {
  %0 = arith.negf %arg0 : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// CHECK-LABEL: func @neg_16_f32
// CHECK: aievec.neg {{.*}} : vector<16xf32>
// CHECK-NOT: arith.negf
func.func @neg_16_f32(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.negf %arg0 : vector<16xf32>
  return %0 : vector<16xf32>
}

// f16 is left alone: it would take the same 16-bit UPS path as bf16 and be
// reinterpreted, so it stays legal and reaches the backend as arith.negf.
// CHECK-LABEL: func @neg_32_f16
// CHECK: arith.negf
// CHECK-NOT: aievec.neg
func.func @neg_32_f16(%arg0: vector<32xf16>) -> vector<32xf16> {
  %0 = arith.negf %arg0 : vector<32xf16>
  return %0 : vector<32xf16>
}
