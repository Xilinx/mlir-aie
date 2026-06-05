//===- test-sub-aie2p.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
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

// 8-wide bf16 sub is below the AIE2P native lane width (16). The pattern
// pads each operand to v16bf16 by concatenating with a zero half, runs the
// standard UPS+SubElem+SRS bf16 path, then extracts the lower 8 lanes.
// CHECK-LABEL: func @vecsubf_bf16_v8(
// CHECK-SAME: %[[LHS:.*]]: vector<8xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<8xbf16>)
func.func @vecsubf_bf16_v8(%arg0: vector<8xbf16>, %arg1: vector<8xbf16>) -> vector<8xbf16> {
  // CHECK:  %[[C0_I32:.*]] = arith.constant 0 : i32
  // CHECK:  %[[C0_BF16:.*]] = arith.constant 0.000000e+00 : bf16
  // CHECK:  %[[V16Z:.*]] = aievec.broadcast_scalar %[[C0_BF16]] : bf16, vector<16xbf16>
  // CHECK:  %[[V8Z:.*]] = aievec.ext %[[V16Z]] {index = 0 : i8} : vector<16xbf16>, vector<8xbf16>
  // CHECK:  %[[LHS16:.*]] = aievec.concat %[[LHS]], %[[V8Z]] : vector<8xbf16>, vector<16xbf16>
  // CHECK:  %[[RHS16:.*]] = aievec.concat %[[RHS]], %[[V8Z]] : vector<8xbf16>, vector<16xbf16>
  // CHECK:  %[[LUPS:.*]] = aievec.ups %[[LHS16]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[RUPS:.*]] = aievec.ups %[[RHS16]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[SUB:.*]] = aievec.sub_elem %[[LUPS]], %[[RUPS]] : vector<16xf32>
  // CHECK:  %[[SRS:.*]] = aievec.srs %[[SUB]], %[[C0_I32]] : vector<16xf32>, i32, vector<16xbf16>
  // CHECK:  %[[EXTLO:.*]] = aievec.ext %[[SRS]] {index = 0 : i8} : vector<16xbf16>, vector<8xbf16>
  %0 = arith.subf %arg0, %arg1 : vector<8xbf16>
  // CHECK: return %[[EXTLO]] : vector<8xbf16>
  return %0 : vector<8xbf16>
}
