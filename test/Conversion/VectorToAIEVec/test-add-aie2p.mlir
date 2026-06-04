//===- test-add-aie2p.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" -cse | FileCheck %s

// CHECK-LABEL: func @vecaddf_f32(
// CHECK-SAME: %[[LHS:.*]]: vector<16xf32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xf32>)
func.func @vecaddf_f32(%arg0: vector<16xf32>, %arg1: vector<16xf32>) -> vector<16xf32> {
  // CHECK:  %[[LCAST:.*]] = aievec.cast %[[LHS]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK:  %[[RCAST:.*]] = aievec.cast %[[RHS]] {isResAcc = true} : vector<16xf32>, vector<16xf32>
  // CHECK:  %[[ADD:.*]] = aievec.add_elem %[[LCAST]], %[[RCAST]] : vector<16xf32>
  // CHECK:  %[[CAST:.*]] = aievec.cast %[[ADD]] {isResAcc = false} : vector<16xf32>, vector<16xf32>
  %0 = arith.addf %arg0, %arg1 : vector<16xf32>
  // CHECK: return %[[CAST]] : vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @vecaddf_bf16(
// CHECK-SAME: %[[LHS:.*]]: vector<16xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xbf16>)
func.func @vecaddf_bf16(%arg0: vector<16xbf16>, %arg1: vector<16xbf16>) -> vector<16xbf16> {
  // CHECK:  %[[C0:.*]] = arith.constant 0 : i32
  // CHECK:  %[[LUPS:.*]] = aievec.ups %[[LHS]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[RUPS:.*]] = aievec.ups %[[RHS]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[ADD:.*]] = aievec.add_elem %[[LUPS]], %[[RUPS]] : vector<16xf32>
  // CHECK:  %[[SRS:.*]] = aievec.srs %[[ADD]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
  %0 = arith.addf %arg0, %arg1 : vector<16xbf16>
  // CHECK: return %[[SRS]] : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// 8-wide bf16 add is below the AIE2P native lane width (16). The pattern
// pads each operand to v16bf16 by concatenating with a zero half, runs the
// standard UPS+AddElem+SRS bf16 path, then extracts the lower 8 lanes.
// CHECK-LABEL: func @vecaddf_bf16_v8(
// CHECK-SAME: %[[LHS:.*]]: vector<8xbf16>,
// CHECK-SAME: %[[RHS:.*]]: vector<8xbf16>)
func.func @vecaddf_bf16_v8(%arg0: vector<8xbf16>, %arg1: vector<8xbf16>) -> vector<8xbf16> {
  // CHECK:  %[[C0_I32:.*]] = arith.constant 0 : i32
  // CHECK:  %[[C0_BF16:.*]] = arith.constant 0.000000e+00 : bf16
  // CHECK:  %[[V16Z:.*]] = aievec.broadcast_scalar %[[C0_BF16]] : bf16, vector<16xbf16>
  // CHECK:  %[[V8Z:.*]] = aievec.ext %[[V16Z]] {index = 0 : i8} : vector<16xbf16>, vector<8xbf16>
  // CHECK:  %[[LHS16:.*]] = aievec.concat %[[LHS]], %[[V8Z]] : vector<8xbf16>, vector<16xbf16>
  // CHECK:  %[[RHS16:.*]] = aievec.concat %[[RHS]], %[[V8Z]] : vector<8xbf16>, vector<16xbf16>
  // CHECK:  %[[LUPS:.*]] = aievec.ups %[[LHS16]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[RUPS:.*]] = aievec.ups %[[RHS16]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
  // CHECK:  %[[ADD:.*]] = aievec.add_elem %[[LUPS]], %[[RUPS]] : vector<16xf32>
  // CHECK:  %[[SRS:.*]] = aievec.srs %[[ADD]], %[[C0_I32]] : vector<16xf32>, i32, vector<16xbf16>
  // CHECK:  %[[EXTLO:.*]] = aievec.ext %[[SRS]] {index = 0 : i8} : vector<16xbf16>, vector<8xbf16>
  %0 = arith.addf %arg0, %arg1 : vector<8xbf16>
  // CHECK: return %[[EXTLO]] : vector<8xbf16>
  return %0 : vector<8xbf16>
}
