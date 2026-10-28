//===- test-f32-mul-bf16-inputs.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// f32 multiplies and fmas whose inputs hold bf16 values. AIE2P lowers 32 lanes
// (and a 64-lane multiply) in one op; AIE2 in two 16-lane halves.

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s --check-prefixes=CHECK,AIE2P
// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s --check-prefixes=CHECK,AIE2

// CHECK-LABEL: func @mulf_32
// CHECK-SAME: (%[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>)
// AIE2P: %[[M:.*]] = aievec.mul_elem %[[A]], %[[B]] : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// AIE2P: %[[R:.*]] = aievec.cast %[[M]] {isResAcc = false} : vector<32xf32>, vector<32xf32>
// AIE2P: return %[[R]]
// AIE2-COUNT-2: aievec.mul_elem {{.*}} : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
// AIE2: aievec.concat {{.*}} : vector<16xf32>, vector<32xf32>
func.func @mulf_32(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<32xf32> {
  %0 = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %1 = arith.extf %b : vector<32xbf16> to vector<32xf32>
  %2 = arith.mulf %0, %1 : vector<32xf32>
  return %2 : vector<32xf32>
}

// -----

// 64 lanes: one op on AIE2P, left alone on AIE2.
// CHECK-LABEL: func @mulf_64
// AIE2P: aievec.mul_elem {{.*}} : vector<64xbf16>, vector<64xbf16>, vector<64xf32>
// AIE2P-NOT: aievec.concat
// AIE2-NOT: aievec.mul_elem
// AIE2: arith.mulf {{.*}} : vector<64xf32>
func.func @mulf_64(%a: vector<64xbf16>, %b: vector<64xbf16>) -> vector<64xf32> {
  %0 = arith.extf %a : vector<64xbf16> to vector<64xf32>
  %1 = arith.extf %b : vector<64xbf16> to vector<64xf32>
  %2 = arith.mulf %0, %1 : vector<64xf32>
  return %2 : vector<64xf32>
}

// -----

// CHECK-LABEL: func @fma_32
// CHECK-SAME: (%[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>, %[[C:.*]]: vector<32xf32>)
// AIE2P: %[[R:.*]] = aievec.mac_elem %[[A]], %[[B]], %[[C]] : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// AIE2P: return %[[R]]
// AIE2-COUNT-2: aievec.mac_elem {{.*}} : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
// AIE2: aievec.concat {{.*}} : vector<16xf32>, vector<32xf32>
func.func @fma_32(%a: vector<32xbf16>, %b: vector<32xbf16>, %c: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %1 = arith.extf %b : vector<32xbf16> to vector<32xf32>
  %2 = vector.fma %0, %1, %c : vector<32xf32>
  return %2 : vector<32xf32>
}

// -----

// A constant that is exact in bf16 is narrowed where it is used.
// CHECK-LABEL: func @mulf_16_constant
// CHECK-SAME: (%[[A:.*]]: vector<16xbf16>)
// CHECK-DAG: %[[C:.*]] = arith.constant dense<5.000000e-01> : vector<16xf32>
// CHECK-DAG: %[[S:.*]] = aievec.srs %[[C]], %{{.*}} : vector<16xf32>, i32, vector<16xbf16>
// CHECK: aievec.mul_elem %[[A]], %[[S]] : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
func.func @mulf_16_constant(%a: vector<16xbf16>) -> vector<16xf32> {
  %c = arith.constant dense<0.5> : vector<16xf32>
  %0 = arith.extf %a : vector<16xbf16> to vector<16xf32>
  %1 = arith.mulf %0, %c : vector<16xf32>
  return %1 : vector<16xf32>
}

// -----

// CHECK-LABEL: func @fma_32_constant
// CHECK-DAG: %[[C:.*]] = arith.constant dense<2.500000e-01> : vector<32xf32>
// CHECK-DAG: %[[S:.*]] = aievec.srs %[[C]], %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
// AIE2P: aievec.mac_elem %{{.*}}, %[[S]], %{{.*}} : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// AIE2-COUNT-2: aievec.mac_elem
func.func @fma_32_constant(%a: vector<32xbf16>, %acc: vector<32xf32>) -> vector<32xf32> {
  %c = arith.constant dense<0.25> : vector<32xf32>
  %0 = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %1 = vector.fma %0, %c, %acc : vector<32xf32>
  return %1 : vector<32xf32>
}

// -----

// A constant that is not exact in bf16 is not a bf16 input: the multiply is
// left as it is.
// CHECK-LABEL: func @mulf_32_inexact_constant
// CHECK-NOT: aievec.mul_elem
// CHECK: arith.mulf {{.*}} : vector<32xf32>
func.func @mulf_32_inexact_constant(%a: vector<32xbf16>) -> vector<32xf32> {
  %c = arith.constant dense<0.797884> : vector<32xf32>
  %0 = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %1 = arith.mulf %0, %c : vector<32xf32>
  return %1 : vector<32xf32>
}

// -----

// A multiply with bf16 inputs feeding an add is lowered on its own.
// CHECK-LABEL: func @mulf_32_into_add
// AIE2P: aievec.mul_elem {{.*}} : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// AIE2-COUNT-2: aievec.mul_elem {{.*}} : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
// CHECK-NOT: arith.mulf
func.func @mulf_32_into_add(%a: vector<32xbf16>, %b: vector<32xbf16>, %c: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %1 = arith.extf %b : vector<32xbf16> to vector<32xf32>
  %2 = arith.mulf %0, %1 : vector<32xf32>
  %3 = arith.addf %2, %c : vector<32xf32>
  return %3 : vector<32xf32>
}
