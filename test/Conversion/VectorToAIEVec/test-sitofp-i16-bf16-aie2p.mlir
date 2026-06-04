//===- test-sitofp-i16-bf16-aie2p.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Verify the magic-number lowering for `arith.sitofp vector<Nxi16> to
// vector<Nxbf16>` on AIE2P. AIE2P has no vector int->fp instruction; the
// pattern emits the classic "magic constant" sequence used by aie_api
// (see detail/aie2p/elementary.hpp) on top of existing aievec ops:
//
//   UPS(int16 -> acc32) -> add(magic.as_acc32) -> cast(acc32 -> accfloat)
//   -> sub(magic.as_accfloat) -> SRS(accfloat -> bf16)
//
// 64-wide inputs are split into two 32-wide halves via vector.shuffle.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec='aie-target=aie2p target-backend=llvmir' -cse | FileCheck %s

// CHECK-LABEL: @sitofp_v32_i16_bf16(
// CHECK-SAME:  %[[A:.*]]: vector<32xi16>
func.func @sitofp_v32_i16_bf16(%v: vector<32xi16>) -> vector<32xbf16> {
  // CHECK:  %[[CST:.*]] = arith.constant dense<19201> : vector<32xi16>
  // CHECK:  %[[BC:.*]] = vector.bitcast %[[CST]] : vector<32xi16> to vector<32xbf16>
  // CHECK:  %[[MAGIC_F:.*]] = aievec.ups %[[BC]] {{.*}} : vector<32xbf16>, vector<32xf32>
  // CHECK:  %[[MAGIC_I:.*]] = aievec.cast %[[MAGIC_F]] {isResAcc = true} : vector<32xf32>, vector<32xi32>
  // CHECK:  %[[V_I:.*]] = aievec.ups %[[A]] {{.*}} : vector<32xi16>, vector<32xi32>
  // CHECK:  %[[SUM_I:.*]] = arith.addi %[[V_I]], %[[MAGIC_I]] : vector<32xi32>
  // CHECK:  %[[SUM_F:.*]] = aievec.cast %[[SUM_I]] {isResAcc = true} : vector<32xi32>, vector<32xf32>
  // CHECK:  %[[DIFF:.*]] = aievec.sub_elem %[[SUM_F]], %[[MAGIC_F]] : vector<32xf32>
  // CHECK:  %[[BF:.*]] = aievec.srs %[[DIFF]], {{.*}} : vector<32xf32>, i32, vector<32xbf16>
  %r = arith.sitofp %v : vector<32xi16> to vector<32xbf16>
  // CHECK: return %[[BF]] : vector<32xbf16>
  return %r : vector<32xbf16>
}

// 16-wide: same pattern, narrower vector type.
// CHECK-LABEL: @sitofp_v16_i16_bf16
func.func @sitofp_v16_i16_bf16(%v: vector<16xi16>) -> vector<16xbf16> {
  // CHECK:  aievec.ups %{{.*}} : vector<16xi16>, vector<16xi32>
  // CHECK:  arith.addi %{{.*}}, %{{.*}} : vector<16xi32>
  // CHECK:  aievec.cast %{{.*}} {isResAcc = true} : vector<16xi32>, vector<16xf32>
  // CHECK:  aievec.sub_elem %{{.*}}, %{{.*}} : vector<16xf32>
  // CHECK:  aievec.srs %{{.*}} : vector<16xf32>, i32, vector<16xbf16>
  %r = arith.sitofp %v : vector<16xi16> to vector<16xbf16>
  return %r : vector<16xbf16>
}

// 64-wide: split into two 32-wide halves via vector.shuffle, run the magic
// sequence on each, then concat the bf16 results with vector.shuffle.
// CHECK-LABEL: @sitofp_v64_i16_bf16(
// CHECK-SAME:  %[[A:.*]]: vector<64xi16>
func.func @sitofp_v64_i16_bf16(%v: vector<64xi16>) -> vector<64xbf16> {
  // The second operand of each split shuffle is unused; the canonicalizer
  // (run by -convert-vector-to-aievec) rewrites it to ub.poison.
  // CHECK:  %[[POISON:.*]] = ub.poison : vector<64xi16>
  // CHECK:  %[[LO:.*]] = vector.shuffle %[[A]], %[[POISON]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<64xi16>, vector<64xi16>
  // CHECK:  %[[HI:.*]] = vector.shuffle %[[A]], %[[POISON]] [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xi16>, vector<64xi16>
  // CHECK:  aievec.ups %[[LO]] {{.*}} : vector<32xi16>, vector<32xi32>
  // CHECK:  aievec.ups %[[HI]] {{.*}} : vector<32xi16>, vector<32xi32>
  // CHECK:  vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xbf16>, vector<32xbf16>
  %r = arith.sitofp %v : vector<64xi16> to vector<64xbf16>
  return %r : vector<64xbf16>
}
