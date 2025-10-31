//===- test-srs-aie2p.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

func.func @v64bf16_srs_v64f32(%arg0 : vector<64xf32>) {
  %c0 = arith.constant 0 : i32
  %0 = aievec.srs %arg0, %c0 : vector<64xf32>, i32, vector<64xbf16>
  return
}

// CHECK-LABEL: @v64bf16_srs_v64f32
// CHECK-SAME: %[[ARG0:.*]]: vector<64xf32>
// CHECK-NEXT: %[[SHIFT0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[INDEX0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG: %[[INDEX1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[BITCAST0:.*]] = llvm.bitcast %[[ARG0]] : vector<64xf32> to vector<64xi32>
// CHECK-NEXT: %[[SHUFFLE0:.*]] = vector.shuffle %[[BITCAST0]], %[[BITCAST0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
// CHECK-NEXT: %[[BITCAST1:.*]] = llvm.bitcast %[[SHUFFLE0]] : vector<32xi32> to vector<32xf32>
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2p.v32accfloat.to.v32bf16"(
// CHECK-SAME: %[[BITCAST1]]) :
// CHECK-SAME: (vector<32xf32>) -> vector<32xbf16>
// CHECK-NEXT: %[[BITCAST2:.*]] = llvm.bitcast %[[ARG0]] : vector<64xf32> to vector<64xi32>
// CHECK-NEXT: %[[SHUFFLE1:.*]] = vector.shuffle %[[BITCAST2]], %[[BITCAST2]] [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// CHECK-NEXT: %[[BITCAST3:.*]] = llvm.bitcast %[[SHUFFLE1]] : vector<32xi32> to vector<32xf32>
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2p.v32accfloat.to.v32bf16"(
// CHECK-SAME: %[[BITCAST3]]) :
// CHECK-SAME: (vector<32xf32>) -> vector<32xbf16>
// CHECK-NEXT: %[[BITCAST4:.*]] = llvm.bitcast %[[SRS0]] : vector<32xbf16> to vector<16xi32>
// CHECK-NEXT: %[[BITCAST5:.*]] = llvm.bitcast %[[SRS1]] : vector<32xbf16> to vector<16xi32>
// CHECK-NEXT: %[[CONCAT:.*]] = vector.shuffle %[[BITCAST4]], %[[BITCAST5]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
// CHECK-NEXT: %[[RES:.*]] = llvm.bitcast %[[CONCAT]] : vector<32xi32> to vector<64xbf16>

// -----

func.func @v32i16_srs_v32i64(%arg0 : vector<32xi64>) {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %0 = aievec.srs %arg0, %c0 : vector<32xi64>, i32, vector<32xi16>
  %1 = aievec.srs %arg0, %c5 : vector<32xi64>, i32, vector<32xi16>
  return
}

// CHECK-LABEL: @v32i16_srs_v32i64
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi64>
// CHECK-NEXT: %[[SHIFT0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = arith.constant 5 : i32
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2p.I512.v32.acc64.srs"(
// CHECK-SAME: %[[ARG0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<32xi64>, i32, i32) -> vector<32xi16>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2p.I512.v32.acc64.srs"(
// CHECK-SAME: %[[ARG0]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<32xi64>, i32, i32) -> vector<32xi16>

// -----

func.func @v64i8_srs_v64i32(%arg0 : vector<64xi32>) {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %0 = aievec.srs %arg0, %c0 : vector<64xi32>, i32, vector<64xi8>
  %1 = aievec.srs %arg0, %c5 : vector<64xi32>, i32, vector<64xi8>
  return
}

// CHECK-LABEL: @v64i8_srs_v64i32
// CHECK-SAME: %[[ARG0:.*]]: vector<64xi32>
// CHECK-NEXT: %[[SHIFT0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[SHIFT5:.*]] = arith.constant 5 : i32
// CHECK-NEXT: %[[SIGN0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SRS0:.*]] = "xllvm.intr.aie2p.I512.v64.acc32.srs"(
// CHECK-SAME: %[[ARG0]], %[[SHIFT0]], %[[SIGN0]]) : 
// CHECK-SAME: (vector<64xi32>, i32, i32) -> vector<64xi8>
// CHECK-NEXT: %[[SIGN1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[SRS1:.*]] = "xllvm.intr.aie2p.I512.v64.acc32.srs"(
// CHECK-SAME: %[[ARG0]], %[[SHIFT5]], %[[SIGN1]]) : 
// CHECK-SAME: (vector<64xi32>, i32, i32) -> vector<64xi8>

// -----

func.func @v64bfp16_from_v64f32(%arg0 : vector<64xf32>) -> !llvm.struct<(vector<64xi8>, vector<8xi8>)> {
  %0 = "xllvm.intr.aie2p.v64accfloat.to.v64bfp16ebs8"(%arg0) : (vector<64xf32>) -> !llvm.struct<(vector<64xi8>, vector<8xi8>)>
  return %0 : !llvm.struct<(vector<64xi8>, vector<8xi8>)>
}

// CHECK-LABEL: @v64bfp16_from_v64f32
// CHECK-SAME: %[[ARG0:.*]]: vector<64xf32>
// CHECK-NEXT: %[[BFP:.*]] = "xllvm.intr.aie2p.v64accfloat.to.v64bfp16ebs8"(%[[ARG0]]) : (vector<64xf32>) -> !llvm.struct<(vector<64xi8>, vector<8xi8>)>
// CHECK-NEXT: return %[[BFP]] : !llvm.struct<(vector<64xi8>, vector<8xi8>)>
