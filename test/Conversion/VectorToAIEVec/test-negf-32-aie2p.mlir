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

// CHECK-LABEL: func @neg_32_f32
// CHECK: aievec.neg {{.*}} : vector<32xf32>
// CHECK-NOT: arith.negf
func.func @neg_32_f32(%arg0: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.negf %arg0 : vector<32xf32>
  return %0 : vector<32xf32>
}

// The 16-lane path is unchanged.
// CHECK-LABEL: func @neg_16_f32
// CHECK: aievec.neg {{.*}} : vector<16xf32>
// CHECK-NOT: arith.negf
func.func @neg_16_f32(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.negf %arg0 : vector<16xf32>
  return %0 : vector<16xf32>
}
