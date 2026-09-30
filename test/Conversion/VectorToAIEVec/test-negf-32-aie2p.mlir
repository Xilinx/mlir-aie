//===- test-negf-32-aie2p.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// arith.negf on AIE2P, at 16 and 32 lanes.

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s

// 32-lane bf16 takes the UPS/neg/SRS branch.
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

// f16 stays legal: it would take the bf16 UPS path and be reinterpreted.
// CHECK-LABEL: func @neg_32_f16
// CHECK: arith.negf
// CHECK-NOT: aievec.neg
func.func @neg_32_f16(%arg0: vector<32xf16>) -> vector<32xf16> {
  %0 = arith.negf %arg0 : vector<32xf16>
  return %0 : vector<32xf16>
}

// `getVectorLaneSize` is the product of every dimension, so this counts 32
// too. It is left alone: NegOpAIE2pConversion builds its shuffle masks per
// scalar lane from the shaped operand, so taking it here would emit indices
// that do not verify. Legal, and reaching the backend as arith.negf, is what
// it did before 32 lanes were taken at all.
// CHECK-LABEL: func @neg_2x16_f32
// CHECK: arith.negf
// CHECK-NOT: aievec.neg
func.func @neg_2x16_f32(%arg0: vector<2x16xf32>) -> vector<2x16xf32> {
  %0 = arith.negf %arg0 : vector<2x16xf32>
  return %0 : vector<2x16xf32>
}
