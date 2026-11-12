//===- test-extract-half.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s

// A half-size vector.extract_strided_slice at offset 0 or at the half is the
// bottom or top half itself (aievec.ext); any other offset is a shift of the
// two halves.

// CHECK-LABEL: func.func @half_slices(
// CHECK-SAME: %[[V:.*]]: vector<32xbf16>
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[LO:.*]] = aievec.ext %[[V]] {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
// CHECK-DAG: %[[HI:.*]] = aievec.ext %[[V]] {index = 1 : i8} : vector<32xbf16>, vector<16xbf16>
// CHECK: %[[MID:.*]] = aievec.shift %[[LO]], %[[HI]], %[[C16]] {isAcc = false}
// CHECK-NOT: aievec.shift
// CHECK: return %[[LO]], %[[HI]], %[[MID]]
func.func @half_slices(%v: vector<32xbf16>) -> (vector<16xbf16>, vector<16xbf16>, vector<16xbf16>) {
  %lo = vector.extract_strided_slice %v offsets = [0], sizes = [16], strides = [1] : vector<32xbf16> to vector<16xbf16>
  %hi = vector.extract_strided_slice %v offsets = [16], sizes = [16], strides = [1] : vector<32xbf16> to vector<16xbf16>
  %mid = vector.extract_strided_slice %v offsets = [8], sizes = [16], strides = [1] : vector<32xbf16> to vector<16xbf16>
  return %lo, %hi, %mid : vector<16xbf16>, vector<16xbf16>, vector<16xbf16>
}
