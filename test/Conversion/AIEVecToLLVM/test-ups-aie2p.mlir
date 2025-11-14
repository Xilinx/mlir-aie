//===- test-ups-aie2p.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

func.func @v64f32_ups_v64bf16(%arg0 : vector<64xbf16>) {
  %0 = aievec.ups %arg0 {shift = 0 : i8} : vector<64xbf16>, vector<64xf32>
  return
}

// CHECK-LABEL: @v64f32_ups_v64bf16
// CHECK-SAME: %[[ARG0:.*]]: vector<64xbf16>
// CHECK-DAG: %[[INDEX0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG: %[[INDEX1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<64xbf16> to vector<32xi32>
// CHECK-NEXT: %[[SHUFFLE0:.*]] = vector.shuffle %[[BITCAST0]], %[[BITCAST0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[SHUFFLE0]] : vector<16xi32> to vector<32xbf16>
// CHECK-NEXT: %[[UPS0:.*]] = "xllvm.intr.aie2p.v32bf16.to.v32accfloat"(
// CHECK-SAME: %[[BITCAST1]]) :
// CHECK-SAME: (vector<32xbf16>) -> vector<32xf32>
// CHECK-NEXT: %[[BITCAST2:.*]] = llvm.bitcast %[[ARG0]] : vector<64xbf16> to vector<32xi32>
// CHECK-NEXT: %[[SHUFFLE1:.*]] = vector.shuffle %[[BITCAST2]], %[[BITCAST2]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
// CHECK-NEXT: %[[BITCAST3:.*]] = llvm.bitcast %[[SHUFFLE1]] : vector<16xi32> to vector<32xbf16>
// CHECK-NEXT: %[[UPS1:.*]] = "xllvm.intr.aie2p.v32bf16.to.v32accfloat"(
// CHECK-SAME: %[[BITCAST3]]) :
// CHECK-SAME: (vector<32xbf16>) -> vector<32xf32>
// CHECK-NEXT: %[[BITCAST4:.*]] = llvm.bitcast %[[UPS0]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: %[[BITCAST5:.*]] = llvm.bitcast %[[UPS1]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = vector.shuffle %[[BITCAST4]], %[[BITCAST5]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<64xi32> to vector<64xf32>

// -----

func.func @v64i32_ups_v64i8(%arg0 : vector<64xi8>) {
  %0 = aievec.ups %arg0 {shift = 0 : i8} : vector<64xi8>, vector<64xi32>
  %1 = aievec.ups %arg0 {shift = 5 : i8} : vector<64xi8>, vector<64xi32>
  return 
}

// CHECK-LABEL: @v64i32_ups_v64i8
// CHECK-SAME: %[[ARG0:.*]]: vector<64xi8>
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[UPS0:.*]] = "xllvm.intr.aie2p.acc32.v64.I512.ups"(
// CHECK-SAME: %[[ARG0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<64xi8>, i32, i32) -> vector<64xi32>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT: %[[UPS1:.*]] = "xllvm.intr.aie2p.acc32.v64.I512.ups"(
// CHECK-SAME: %[[ARG0]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<64xi8>, i32, i32) -> vector<64xi32>

// -----

func.func @v32i64_ups_v32i16(%arg0 : vector<32xi16>) {
  %0 = aievec.ups %arg0 {shift = 0 : i8} : vector<32xi16>, vector<32xi64>
  %1 = aievec.ups %arg0 {shift = 5 : i8} : vector<32xi16>, vector<32xi64>
  return 
}

// CHECK-LABEL: @v32i64_ups_v32i16
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi16>
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[UPS0:.*]] = "xllvm.intr.aie2p.acc64.v32.I512.ups"(
// CHECK-SAME: %[[ARG0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<32xi16>, i32, i32) -> vector<32xi64>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT: %[[UPS1:.*]] = "xllvm.intr.aie2p.acc64.v32.I512.ups"(
// CHECK-SAME: %[[ARG0]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<32xi16>, i32, i32) -> vector<32xi64>
