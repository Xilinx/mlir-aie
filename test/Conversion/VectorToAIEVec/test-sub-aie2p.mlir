//===- test-sub-aie2p.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" -cse | FileCheck %s

// CHECK-LABEL: func @vecsubf_f32(
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @vecsubf_f32(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xf32> {
  // CHECK:  %[[LCAST:.*]] = aievec.cast %[[LHS]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK:  %[[RCAST:.*]] = aievec.cast %[[RHS]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK:  %[[SUB:.*]] = aievec.sub_elem %[[LCAST]], %[[RCAST]] : vector<16xf32>
  // CHECK:  %[[CAST:.*]] = aievec.cast %[[SUB]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  %0 = arith.subf %arg0, %arg1 : vector<16xf32>
  // CHECK: return %[[CAST]] : vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @vecsubf_bf16(
// CHECK-SAME: %[[LHS:.*]]: vector<16xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xbf16>)
func.func @vecsubf_bf16(%arg0: vector<16xbf16>, %arg1: vector<16xbf16>) -> vector<16xbf16> {
  // CHECK:  %[[C0:.*]] = arith.constant 0 : i32
  // CHECK:  %[[LUPS:.*]] = aievec.ups %[[LHS]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[RUPS:.*]] = aievec.ups %[[RHS]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[SUB:.*]] = aievec.sub_elem %[[LUPS]], %[[RUPS]] : vector<16xf32>
  // CHECK:  %[[SRS:.*]] = aievec.srs %[[SUB]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
  %0 = arith.subf %arg0, %arg1 : vector<16xbf16>
  // CHECK: return %[[SRS]] : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// CHECK-LABEL: func @vecsubf_bf16_f32(
// CHECK-SAME: %[[LHS:.*]]: vector<16xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xbf16>)
func.func @vecsubf_bf16_f32(%arg0: vector<16xbf16>, %arg1: vector<16xbf16>) -> vector<16xf32> {
  // CHECK:  %[[LUPS:.*]] = aievec.ups %[[LHS]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[RUPS:.*]] = aievec.ups %[[RHS]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[SUB:.*]] = aievec.sub_elem %[[LUPS]], %[[RUPS]] : vector<16xf32>
  // CHECK:  %[[CAST:.*]] = aievec.cast %[[SUB]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  %1 = arith.extf %arg0 : vector<16xbf16> to vector<16xf32>
  %2 = arith.extf %arg1 : vector<16xbf16> to vector<16xf32>
  %3 = arith.subf %1, %2 : vector<16xf32>
  // CHECK: return %[[CAST]] : vector<16xf32>
  return %3 : vector<16xf32>
}

// CHECK-LABEL: func @vecsubf_bf16_f32_mixed(
// CHECK-SAME: %[[LHS:.*]]: vector<16xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @vecsubf_bf16_f32_mixed(%arg0: vector<16xbf16>, %arg1: vector<16xf32>) -> vector<16xf32> {
  // CHECK:  %[[LUPS:.*]] = aievec.ups %[[LHS]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[RCAST:.*]] = aievec.cast %[[RHS]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK:  %[[SUB:.*]] = aievec.sub_elem %[[LUPS]], %[[RCAST]] : vector<16xf32>
  // CHECK:  %[[CAST:.*]] = aievec.cast %[[SUB]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  %1 = arith.extf %arg0 : vector<16xbf16> to vector<16xf32>
  %2 = arith.subf %1, %arg1 : vector<16xf32>
  // CHECK: return %[[CAST]] : vector<16xf32>
  return %2 : vector<16xf32>
}

// 8-wide bf16 sub is below the AIE2P native lane widths (16, 32). Each
// operand is repeated to v16bf16, the standard UPS+SubElem+SRS bf16 path
// runs, and the lower 8 lanes are extracted.
// CHECK-LABEL: func @vecsubf_bf16_v8(
// CHECK-SAME: %[[LHS:.*]]: vector<8xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<8xbf16>)
func.func @vecsubf_bf16_v8(%arg0: vector<8xbf16>, %arg1: vector<8xbf16>) -> vector<8xbf16> {
  // CHECK:  %[[C0_I32:.*]] = arith.constant 0 : i32
  // CHECK:  %[[LHS16:.*]] = aievec.concat %[[LHS]], %[[LHS]] : vector<8xbf16>, vector<16xbf16>
  // CHECK:  %[[RHS16:.*]] = aievec.concat %[[RHS]], %[[RHS]] : vector<8xbf16>, vector<16xbf16>
  // CHECK:  %[[LUPS:.*]] = aievec.ups %[[LHS16]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[RUPS:.*]] = aievec.ups %[[RHS16]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[SUB:.*]] = aievec.sub_elem %[[LUPS]], %[[RUPS]] : vector<16xf32>
  // CHECK:  %[[SRS:.*]] = aievec.srs %[[SUB]], %[[C0_I32]] : vector<16xf32>, i32, vector<16xbf16>
  // CHECK:  %[[EXTLO:.*]] = aievec.ext %[[SRS]] {index = 0 : i8} : vector<16xbf16>, vector<8xbf16>
  %0 = arith.subf %arg0, %arg1 : vector<8xbf16>
  // CHECK: return %[[EXTLO]] : vector<8xbf16>
  return %0 : vector<8xbf16>
}

// 64 lanes of f32 fill the accumulator: one sub_elem.
// CHECK-LABEL: func @vecsubf_v64f32(
// CHECK-SAME: %[[LHS:.*]]: vector<64xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<64xf32>)
// CHECK: %[[L:.*]] = aievec.cast %[[LHS]] {isResAcc = true} : vector<64xf32>, vector<64xf32>
// CHECK: %[[R:.*]] = aievec.cast %[[RHS]] {isResAcc = true} : vector<64xf32>, vector<64xf32>
// CHECK: %[[S:.*]] = aievec.sub_elem %[[L]], %[[R]] : vector<64xf32>
// CHECK: %[[RES:.*]] = aievec.cast %[[S]] {isResAcc = false} : vector<64xf32>, vector<64xf32>
// CHECK: return %[[RES]]
func.func @vecsubf_v64f32(%arg0: vector<64xf32>, %arg1: vector<64xf32>) -> vector<64xf32> {
  %0 = arith.subf %arg0, %arg1 : vector<64xf32>
  return %0 : vector<64xf32>
}
