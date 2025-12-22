//===- test-ext-aie2p.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm='aie-target=aie2p' | FileCheck %s

// Test extracting lower and upper halves of v64i8
func.func @v32i8_ext_v64i8(%arg0 : vector<64xi8>) -> (vector<32xi8>, vector<32xi8>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<64xi8>, vector<32xi8>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<64xi8>, vector<32xi8>
  return %0, %1 : vector<32xi8>, vector<32xi8>
}

// CHECK-LABEL: @v32i8_ext_v64i8
// CHECK-SAME: %[[ARG0:.*]]: vector<64xi8>
// CHECK: %[[EXT0:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<64xi8>, vector<64xi8>
// CHECK: %[[EXT1:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xi8>, vector<64xi8>
// CHECK: return %[[EXT0]], %[[EXT1]] : vector<32xi8>, vector<32xi8>

// -----

// Test extracting lower and upper halves of v128i8
func.func @v64i8_ext_v128i8(%arg0 : vector<128xi8>) -> (vector<64xi8>, vector<64xi8>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<128xi8>, vector<64xi8>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<128xi8>, vector<64xi8>
  return %0, %1 : vector<64xi8>, vector<64xi8>
}

// CHECK-LABEL: @v64i8_ext_v128i8
// CHECK-SAME: %[[ARG0:.*]]: vector<128xi8>
// CHECK: %[[EXT0:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<128xi8>, vector<128xi8>
// CHECK: %[[EXT1:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xi8>, vector<128xi8>
// CHECK: return %[[EXT0]], %[[EXT1]] : vector<64xi8>, vector<64xi8>

// -----

// Test extracting lower and upper halves of v32i16
func.func @v16i16_ext_v32i16(%arg0 : vector<32xi16>) -> (vector<16xi16>, vector<16xi16>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<32xi16>, vector<16xi16>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<32xi16>, vector<16xi16>
  return %0, %1 : vector<16xi16>, vector<16xi16>
}

// CHECK-LABEL: @v16i16_ext_v32i16
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi16>
// CHECK: %[[EXT0:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xi16>, vector<32xi16>
// CHECK: %[[EXT1:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xi16>, vector<32xi16>
// CHECK: return %[[EXT0]], %[[EXT1]] : vector<16xi16>, vector<16xi16>

// -----

// Test extracting lower and upper halves of v64i16
func.func @v32i16_ext_v64i16(%arg0 : vector<64xi16>) -> (vector<32xi16>, vector<32xi16>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<64xi16>, vector<32xi16>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<64xi16>, vector<32xi16>
  return %0, %1 : vector<32xi16>, vector<32xi16>
}

// CHECK-LABEL: @v32i16_ext_v64i16
// CHECK-SAME: %[[ARG0:.*]]: vector<64xi16>
// CHECK: %[[EXT0:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<64xi16>, vector<64xi16>
// CHECK: %[[EXT1:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xi16>, vector<64xi16>
// CHECK: return %[[EXT0]], %[[EXT1]] : vector<32xi16>, vector<32xi16>

// -----

// Test extracting lower and upper halves of v16i32
func.func @v8i32_ext_v16i32(%arg0 : vector<16xi32>) -> (vector<8xi32>, vector<8xi32>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<16xi32>, vector<8xi32>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<16xi32>, vector<8xi32>
  return %0, %1 : vector<8xi32>, vector<8xi32>
}

// CHECK-LABEL: @v8i32_ext_v16i32
// CHECK-SAME: %[[ARG0:.*]]: vector<16xi32>
// CHECK: %[[EXT0:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<16xi32>, vector<16xi32>
// CHECK: %[[EXT1:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [8, 9, 10, 11, 12, 13, 14, 15] : vector<16xi32>, vector<16xi32>
// CHECK: return %[[EXT0]], %[[EXT1]] : vector<8xi32>, vector<8xi32>

// -----

// Test extracting lower and upper halves of v32i32
func.func @v16i32_ext_v32i32(%arg0 : vector<32xi32>) -> (vector<16xi32>, vector<16xi32>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<32xi32>, vector<16xi32>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<32xi32>, vector<16xi32>
  return %0, %1 : vector<16xi32>, vector<16xi32>
}

// CHECK-LABEL: @v16i32_ext_v32i32
// CHECK-SAME: %[[ARG0:.*]]: vector<32xi32>
// CHECK: %[[EXT0:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xi32>, vector<32xi32>
// CHECK: %[[EXT1:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xi32>, vector<32xi32>
// CHECK: return %[[EXT0]], %[[EXT1]] : vector<16xi32>, vector<16xi32>

// -----

// Test extracting lower and upper halves of v32bf16
func.func @v16bf16_ext_v32bf16(%arg0 : vector<32xbf16>) -> (vector<16xbf16>, vector<16xbf16>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<32xbf16>, vector<16xbf16>
  return %0, %1 : vector<16xbf16>, vector<16xbf16>
}

// CHECK-LABEL: @v16bf16_ext_v32bf16
// CHECK-SAME: %[[ARG0:.*]]: vector<32xbf16>
// CHECK: %[[EXT0:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xbf16>, vector<32xbf16>
// CHECK: %[[EXT1:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xbf16>, vector<32xbf16>
// CHECK: return %[[EXT0]], %[[EXT1]] : vector<16xbf16>, vector<16xbf16>

// -----

// Test extracting lower and upper halves of v64bf16
func.func @v32bf16_ext_v64bf16(%arg0 : vector<64xbf16>) -> (vector<32xbf16>, vector<32xbf16>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<64xbf16>, vector<32xbf16>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<64xbf16>, vector<32xbf16>
  return %0, %1 : vector<32xbf16>, vector<32xbf16>
}

// CHECK-LABEL: @v32bf16_ext_v64bf16
// CHECK-SAME: %[[ARG0:.*]]: vector<64xbf16>
// CHECK: %[[EXT0:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<64xbf16>, vector<64xbf16>
// CHECK: %[[EXT1:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xbf16>, vector<64xbf16>
// CHECK: return %[[EXT0]], %[[EXT1]] : vector<32xbf16>, vector<32xbf16>

// -----

// Test extracting lower and upper halves of v16f32
func.func @v8f32_ext_v16f32(%arg0 : vector<16xf32>) -> (vector<8xf32>, vector<8xf32>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<16xf32>, vector<8xf32>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<16xf32>, vector<8xf32>
  return %0, %1 : vector<8xf32>, vector<8xf32>
}

// CHECK-LABEL: @v8f32_ext_v16f32
// CHECK-SAME: %[[ARG0:.*]]: vector<16xf32>
// CHECK: %[[EXT0:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<16xf32>, vector<16xf32>
// CHECK: %[[EXT1:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [8, 9, 10, 11, 12, 13, 14, 15] : vector<16xf32>, vector<16xf32>
// CHECK: return %[[EXT0]], %[[EXT1]] : vector<8xf32>, vector<8xf32>

// -----

// Test extracting lower and upper halves of v32f32
func.func @v16f32_ext_v32f32(%arg0 : vector<32xf32>) -> (vector<16xf32>, vector<16xf32>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<32xf32>, vector<16xf32>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<32xf32>, vector<16xf32>
  return %0, %1 : vector<16xf32>, vector<16xf32>
}

// CHECK-LABEL: @v16f32_ext_v32f32
// CHECK-SAME: %[[ARG0:.*]]: vector<32xf32>
// CHECK: %[[EXT0:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xf32>, vector<32xf32>
// CHECK: %[[EXT1:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xf32>, vector<32xf32>
// CHECK: return %[[EXT0]], %[[EXT1]] : vector<16xf32>, vector<16xf32>

// -----

// Test extracting lower and upper halves of v64f32
func.func @v32f32_ext_v64f32(%arg0 : vector<64xf32>) -> (vector<32xf32>, vector<32xf32>) {
  %0 = aievec.ext %arg0 {index = 0 : i8} : vector<64xf32>, vector<32xf32>
  %1 = aievec.ext %arg0 {index = 1 : i8} : vector<64xf32>, vector<32xf32>
  return %0, %1 : vector<32xf32>, vector<32xf32>
}

// CHECK-LABEL: @v32f32_ext_v64f32
// CHECK-SAME: %[[ARG0:.*]]: vector<64xf32>
// CHECK: %[[EXT0:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<64xf32>, vector<64xf32>
// CHECK: %[[EXT1:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]] [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xf32>, vector<64xf32>
// CHECK: return %[[EXT0]], %[[EXT1]] : vector<32xf32>, vector<32xf32>
