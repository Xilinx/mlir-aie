//===- test-srs-compound-patterns.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Tests for compound shift+clamp+truncate → SRS pattern lowering.
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s --check-prefix=AIE2P

// Test 1: Full unsigned clamp pattern (shrsi + maxsi(0) + minsi(255) + trunci)
// 32×i32 → 32×i8 with unsigned saturation (sign=0)
// The maxsi/minsi clamp is absorbed into the SRS.

// CHECK-LABEL: func.func @test_srs_clamp_unsigned(
// CHECK-NOT: arith.shrsi
// CHECK-NOT: arith.maxsi
// CHECK-NOT: arith.minsi
// CHECK-NOT: arith.trunci
// CHECK: %[[SHIFT:.*]] = arith.constant 4 : i32
// CHECK: %[[CAST:.*]] = aievec.cast %{{.*}} {isResAcc = true} : vector<32xi32>, vector<32xi32>
// CHECK: %[[SRS:.*]] = aievec.srs %[[CAST]], %[[SHIFT]] {sign = 0 : i32} : vector<32xi32>, i32, vector<32xi8>
// CHECK: return %[[SRS]]
// AIE2P-LABEL: func.func @test_srs_clamp_unsigned(
// AIE2P-NOT: arith.shrsi
// AIE2P-NOT: arith.maxsi
// AIE2P-NOT: arith.minsi
// AIE2P-NOT: arith.trunci
// AIE2P: %[[SHIFT:.*]] = arith.constant 4 : i32
// AIE2P: %[[CAST:.*]] = aievec.cast %{{.*}} {isResAcc = true} : vector<32xi32>, vector<32xi32>
// AIE2P: %[[SRS:.*]] = aievec.srs %[[CAST]], %[[SHIFT]] {sign = 0 : i32} : vector<32xi32>, i32, vector<32xi8>
// AIE2P: return %[[SRS]]
func.func @test_srs_clamp_unsigned(%arg0: vector<32xi32>) -> vector<32xi8> {
  %c0 = arith.constant dense<0> : vector<32xi32>
  %c255 = arith.constant dense<255> : vector<32xi32>
  %shift_splat = arith.constant dense<4> : vector<32xi32>
  %shifted = arith.shrsi %arg0, %shift_splat : vector<32xi32>
  %clamped0 = arith.maxsi %shifted, %c0 : vector<32xi32>
  %clamped = arith.minsi %clamped0, %c255 : vector<32xi32>
  %result = arith.trunci %clamped : vector<32xi32> to vector<32xi8>
  return %result : vector<32xi8>
}

// Test 2: No clamp pattern (shrsi + trunci only)
// 16×i32 → 16×i16, sign defaults to 1 (not printed)

// CHECK-LABEL: func.func @test_srs_no_clamp(
// CHECK-NOT: arith.shrsi
// CHECK-NOT: arith.trunci
// CHECK: %[[SHIFT:.*]] = arith.constant 8 : i32
// CHECK: %[[CAST:.*]] = aievec.cast %{{.*}} {isResAcc = true} : vector<16xi32>, vector<16xi32>
// CHECK: %[[SRS:.*]] = aievec.srs %[[CAST]], %[[SHIFT]] : vector<16xi32>, i32, vector<16xi16>
// CHECK: return %[[SRS]]
// AIE2P-LABEL: func.func @test_srs_no_clamp(
// AIE2P-NOT: arith.shrsi
// AIE2P-NOT: arith.trunci
// AIE2P: %[[SHIFT:.*]] = arith.constant 8 : i32
// AIE2P: %[[CAST:.*]] = aievec.cast %{{.*}} {isResAcc = true} : vector<16xi32>, vector<16xi32>
// AIE2P: %[[SRS:.*]] = aievec.srs %[[CAST]], %[[SHIFT]] : vector<16xi32>, i32, vector<16xi16>
// AIE2P: return %[[SRS]]
func.func @test_srs_no_clamp(%arg0: vector<16xi32>) -> vector<16xi16> {
  %shift_splat = arith.constant dense<8> : vector<16xi32>
  %shifted = arith.shrsi %arg0, %shift_splat : vector<16xi32>
  %result = arith.trunci %shifted : vector<16xi32> to vector<16xi16>
  return %result : vector<16xi16>
}

// Test 3: Signed clamp pattern (shrsi + maxsi(-128) + minsi(127) + trunci)
// 32×i32 → 32×i8 with signed saturation (sign=1, default, not printed)

// CHECK-LABEL: func.func @test_srs_clamp_signed(
// CHECK-NOT: arith.shrsi
// CHECK-NOT: arith.maxsi
// CHECK-NOT: arith.minsi
// CHECK-NOT: arith.trunci
// CHECK: %[[SHIFT:.*]] = arith.constant 4 : i32
// CHECK: %[[CAST:.*]] = aievec.cast %{{.*}} {isResAcc = true} : vector<32xi32>, vector<32xi32>
// CHECK: %[[SRS:.*]] = aievec.srs %[[CAST]], %[[SHIFT]] : vector<32xi32>, i32, vector<32xi8>
// CHECK: return %[[SRS]]
// AIE2P-LABEL: func.func @test_srs_clamp_signed(
// AIE2P-NOT: arith.shrsi
// AIE2P-NOT: arith.maxsi
// AIE2P-NOT: arith.minsi
// AIE2P-NOT: arith.trunci
// AIE2P: %[[SHIFT:.*]] = arith.constant 4 : i32
// AIE2P: %[[CAST:.*]] = aievec.cast %{{.*}} {isResAcc = true} : vector<32xi32>, vector<32xi32>
// AIE2P: %[[SRS:.*]] = aievec.srs %[[CAST]], %[[SHIFT]] : vector<32xi32>, i32, vector<32xi8>
// AIE2P: return %[[SRS]]
func.func @test_srs_clamp_signed(%arg0: vector<32xi32>) -> vector<32xi8> {
  %c_neg128 = arith.constant dense<-128> : vector<32xi32>
  %c127 = arith.constant dense<127> : vector<32xi32>
  %shift_splat = arith.constant dense<4> : vector<32xi32>
  %shifted = arith.shrsi %arg0, %shift_splat : vector<32xi32>
  %clamped0 = arith.maxsi %shifted, %c_neg128 : vector<32xi32>
  %clamped = arith.minsi %clamped0, %c127 : vector<32xi32>
  %result = arith.trunci %clamped : vector<32xi32> to vector<32xi8>
  return %result : vector<32xi8>
}

// Test 4: Scalar shrsi is promoted to vector (broadcast + UPS/SRS + extract)
// The trunci remains scalar since it's i32→i8 scalar narrowing.

// CHECK-LABEL: func.func @test_scalar_shrsi_promoted(
// CHECK: aievec.broadcast_scalar
// CHECK: arith.trunci
// AIE2P-LABEL: func.func @test_scalar_shrsi_promoted(
// AIE2P: aievec.broadcast_scalar
// AIE2P: arith.trunci
func.func @test_scalar_shrsi_promoted(%arg0: i32) -> i8 {
  %shift = arith.constant 4 : i32
  %shifted = arith.shrsi %arg0, %shift : i32
  %result = arith.trunci %shifted : i32 to i8
  return %result : i8
}
