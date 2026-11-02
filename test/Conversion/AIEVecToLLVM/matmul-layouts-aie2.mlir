//===- matmul-layouts-aie2.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm | FileCheck %s
// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm | FileCheck %s --check-prefix=COUNT

// AIE2 matmul tiles with each stored operand layout the op admits. The MAC
// reads lhs [m, k] and rhs [k, n]; an operand stored transposed is brought
// into that layout by one 512-bit vshuffle. Layouts that need any other
// transpose do not verify, see dialect/AIEVec/matmul-layouts-invalid.mlir.

// CHECK-LABEL: func.func @i8_i4_4x16x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<4x16xi8>, %[[B:.*]]: vector<16x8xi4>, %[[C:.*]]: vector<4x8xi32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<4x16xi8> to vector<64xi8>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<16x8xi4> to vector<128xi4>
// COUNT-LABEL: func.func @i8_i4_4x16x8_mk_kn(
// COUNT-NOT: "xllvm.intr.aie2.vshuffle"
func.func @i8_i4_4x16x8_mk_kn(%A : vector<4x16xi8>, %B : vector<16x8xi4>, %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x16xi8>, vector<16x8xi4> into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

// CHECK-LABEL: func.func @i8_i4_4x16x8_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<16x4xi8>, %[[B:.*]]: vector<16x8xi4>, %[[C:.*]]: vector<4x8xi32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<16x4xi8> to vector<64xi8>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<16x8xi4> to vector<128xi4>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[FA]] : vector<64xi8> to vector<16xi32>
// CHECK: %[[LTU:.*]] = "xllvm.intr.aie2.v16int32"()
// CHECK: %[[LTM:.*]] = llvm.mlir.constant(36 : i32) : i32
// CHECK: "xllvm.intr.aie2.vshuffle"(%[[LTI]], %[[LTU]], %[[LTM]])
// COUNT-LABEL: func.func @i8_i4_4x16x8_km_kn(
// COUNT-COUNT-1: "xllvm.intr.aie2.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2.vshuffle"
func.func @i8_i4_4x16x8_km_kn(%A : vector<16x4xi8>, %B : vector<16x8xi4>, %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C lhs_transposed : vector<16x4xi8>, vector<16x8xi4> into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

// CHECK-LABEL: func.func @i8_4x8x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<4x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<4x8xi32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<4x8xi8> to vector<32xi8>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xi8> to vector<64xi8>
// COUNT-LABEL: func.func @i8_4x8x8_mk_kn(
// COUNT-NOT: "xllvm.intr.aie2.vshuffle"
func.func @i8_4x8x8_mk_kn(%A : vector<4x8xi8>, %B : vector<8x8xi8>, %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xi8>, vector<8x8xi8> into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

// CHECK-LABEL: func.func @i8_4x8x8_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<4x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<4x8xi32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<4x8xi8> to vector<32xi8>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xi8> to vector<64xi8>
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[FB]] : vector<64xi8> to vector<16xi32>
// CHECK: %[[RTU:.*]] = "xllvm.intr.aie2.v16int32"()
// CHECK: %[[RTM:.*]] = llvm.mlir.constant(35 : i32) : i32
// CHECK: "xllvm.intr.aie2.vshuffle"(%[[RTI]], %[[RTU]], %[[RTM]])
// COUNT-LABEL: func.func @i8_4x8x8_mk_nk(
// COUNT-COUNT-1: "xllvm.intr.aie2.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2.vshuffle"
func.func @i8_4x8x8_mk_nk(%A : vector<4x8xi8>, %B : vector<8x8xi8>, %C : vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = aievec.matmul %A, %B, %C rhs_transposed : vector<4x8xi8>, vector<8x8xi8> into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_i8_2x8x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<2x8xi16>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<2x8xi64>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<2x8xi16> to vector<16xi16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xi8> to vector<64xi8>
// COUNT-LABEL: func.func @i16_i8_2x8x8_mk_kn(
// COUNT-NOT: "xllvm.intr.aie2.vshuffle"
func.func @i16_i8_2x8x8_mk_kn(%A : vector<2x8xi16>, %B : vector<8x8xi8>, %C : vector<2x8xi64>) -> vector<2x8xi64> {
  %0 = aievec.matmul %A, %B, %C : vector<2x8xi16>, vector<8x8xi8> into vector<2x8xi64>
  return %0 : vector<2x8xi64>
}

// -----

// CHECK-LABEL: func.func @i16_i8_2x8x8_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<2x8xi16>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<2x8xi64>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<2x8xi16> to vector<16xi16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xi8> to vector<64xi8>
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[FB]] : vector<64xi8> to vector<16xi32>
// CHECK: %[[RTU:.*]] = "xllvm.intr.aie2.v16int32"()
// CHECK: %[[RTM:.*]] = llvm.mlir.constant(35 : i32) : i32
// CHECK: "xllvm.intr.aie2.vshuffle"(%[[RTI]], %[[RTU]], %[[RTM]])
// COUNT-LABEL: func.func @i16_i8_2x8x8_mk_nk(
// COUNT-COUNT-1: "xllvm.intr.aie2.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2.vshuffle"
func.func @i16_i8_2x8x8_mk_nk(%A : vector<2x8xi16>, %B : vector<8x8xi8>, %C : vector<2x8xi64>) -> vector<2x8xi64> {
  %0 = aievec.matmul %A, %B, %C rhs_transposed : vector<2x8xi16>, vector<8x8xi8> into vector<2x8xi64>
  return %0 : vector<2x8xi64>
}

// -----

// CHECK-LABEL: func.func @i16_i8_4x8x4_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<4x8xi16>, %[[B:.*]]: vector<8x4xi8>, %[[C:.*]]: vector<4x4xi64>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<4x8xi16> to vector<32xi16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x4xi8> to vector<32xi8>
// COUNT-LABEL: func.func @i16_i8_4x8x4_mk_kn(
// COUNT-NOT: "xllvm.intr.aie2.vshuffle"
func.func @i16_i8_4x8x4_mk_kn(%A : vector<4x8xi16>, %B : vector<8x4xi8>, %C : vector<4x4xi64>) -> vector<4x4xi64> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xi16>, vector<8x4xi8> into vector<4x4xi64>
  return %0 : vector<4x4xi64>
}

// -----

// CHECK-LABEL: func.func @i16_i8_4x8x4_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x4xi16>, %[[B:.*]]: vector<8x4xi8>, %[[C:.*]]: vector<4x4xi64>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x4xi16> to vector<32xi16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x4xi8> to vector<32xi8>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[FA]] : vector<32xi16> to vector<16xi32>
// CHECK: %[[LTU:.*]] = "xllvm.intr.aie2.v16int32"()
// CHECK: %[[LTM:.*]] = llvm.mlir.constant(28 : i32) : i32
// CHECK: "xllvm.intr.aie2.vshuffle"(%[[LTI]], %[[LTU]], %[[LTM]])
// COUNT-LABEL: func.func @i16_i8_4x8x4_km_kn(
// COUNT-COUNT-1: "xllvm.intr.aie2.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2.vshuffle"
func.func @i16_i8_4x8x4_km_kn(%A : vector<8x4xi16>, %B : vector<8x4xi8>, %C : vector<4x4xi64>) -> vector<4x4xi64> {
  %0 = aievec.matmul %A, %B, %C lhs_transposed : vector<8x4xi16>, vector<8x4xi8> into vector<4x4xi64>
  return %0 : vector<4x4xi64>
}

// -----

// CHECK-LABEL: func.func @i16_2x4x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<2x4xi16>, %[[B:.*]]: vector<4x8xi16>, %[[C:.*]]: vector<2x8xi64>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<2x4xi16> to vector<8xi16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<4x8xi16> to vector<32xi16>
// COUNT-LABEL: func.func @i16_2x4x8_mk_kn(
// COUNT-NOT: "xllvm.intr.aie2.vshuffle"
func.func @i16_2x4x8_mk_kn(%A : vector<2x4xi16>, %B : vector<4x8xi16>, %C : vector<2x8xi64>) -> vector<2x8xi64> {
  %0 = aievec.matmul %A, %B, %C : vector<2x4xi16>, vector<4x8xi16> into vector<2x8xi64>
  return %0 : vector<2x8xi64>
}

// -----

// CHECK-LABEL: func.func @i16_2x4x8_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<2x4xi16>, %[[B:.*]]: vector<8x4xi16>, %[[C:.*]]: vector<2x8xi64>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<2x4xi16> to vector<8xi16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x4xi16> to vector<32xi16>
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[FB]] : vector<32xi16> to vector<16xi32>
// CHECK: %[[RTU:.*]] = "xllvm.intr.aie2.v16int32"()
// CHECK: %[[RTM:.*]] = llvm.mlir.constant(28 : i32) : i32
// CHECK: "xllvm.intr.aie2.vshuffle"(%[[RTI]], %[[RTU]], %[[RTM]])
// COUNT-LABEL: func.func @i16_2x4x8_mk_nk(
// COUNT-COUNT-1: "xllvm.intr.aie2.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2.vshuffle"
func.func @i16_2x4x8_mk_nk(%A : vector<2x4xi16>, %B : vector<8x4xi16>, %C : vector<2x8xi64>) -> vector<2x8xi64> {
  %0 = aievec.matmul %A, %B, %C rhs_transposed : vector<2x4xi16>, vector<8x4xi16> into vector<2x8xi64>
  return %0 : vector<2x8xi64>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<4x8xbf16> to vector<32xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x4xbf16> to vector<32xbf16>
// COUNT-LABEL: func.func @bf16_4x8x4_mk_kn(
// COUNT-NOT: "xllvm.intr.aie2.vshuffle"
func.func @bf16_4x8x4_mk_kn(%A : vector<4x8xbf16>, %B : vector<8x4xbf16>, %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = aievec.matmul %A, %B, %C : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<4x8xbf16> to vector<32xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<4x8xbf16> to vector<32xbf16>
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[FB]] : vector<32xbf16> to vector<16xi32>
// CHECK: %[[RTU:.*]] = "xllvm.intr.aie2.v16int32"()
// CHECK: %[[RTM:.*]] = llvm.mlir.constant(29 : i32) : i32
// CHECK: "xllvm.intr.aie2.vshuffle"(%[[RTI]], %[[RTU]], %[[RTM]])
// COUNT-LABEL: func.func @bf16_4x8x4_mk_nk(
// COUNT-COUNT-1: "xllvm.intr.aie2.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2.vshuffle"
func.func @bf16_4x8x4_mk_nk(%A : vector<4x8xbf16>, %B : vector<4x8xbf16>, %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = aievec.matmul %A, %B, %C rhs_transposed : vector<4x8xbf16>, vector<4x8xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x4xbf16> to vector<32xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x4xbf16> to vector<32xbf16>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[FA]] : vector<32xbf16> to vector<16xi32>
// CHECK: %[[LTU:.*]] = "xllvm.intr.aie2.v16int32"()
// CHECK: %[[LTM:.*]] = llvm.mlir.constant(28 : i32) : i32
// CHECK: "xllvm.intr.aie2.vshuffle"(%[[LTI]], %[[LTU]], %[[LTM]])
// COUNT-LABEL: func.func @bf16_4x8x4_km_kn(
// COUNT-COUNT-1: "xllvm.intr.aie2.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2.vshuffle"
func.func @bf16_4x8x4_km_kn(%A : vector<8x4xbf16>, %B : vector<8x4xbf16>, %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = aievec.matmul %A, %B, %C lhs_transposed : vector<8x4xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_km_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x4xbf16> to vector<32xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<4x8xbf16> to vector<32xbf16>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[FA]] : vector<32xbf16> to vector<16xi32>
// CHECK: %[[LTU:.*]] = "xllvm.intr.aie2.v16int32"()
// CHECK: %[[LTM:.*]] = llvm.mlir.constant(28 : i32) : i32
// CHECK: "xllvm.intr.aie2.vshuffle"(%[[LTI]], %[[LTU]], %[[LTM]])
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[FB]] : vector<32xbf16> to vector<16xi32>
// CHECK: %[[RTU:.*]] = "xllvm.intr.aie2.v16int32"()
// CHECK: %[[RTM:.*]] = llvm.mlir.constant(29 : i32) : i32
// CHECK: "xllvm.intr.aie2.vshuffle"(%[[RTI]], %[[RTU]], %[[RTM]])
// COUNT-LABEL: func.func @bf16_4x8x4_km_nk(
// COUNT-COUNT-2: "xllvm.intr.aie2.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2.vshuffle"
func.func @bf16_4x8x4_km_nk(%A : vector<8x4xbf16>, %B : vector<4x8xbf16>, %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = aievec.matmul %A, %B, %C lhs_transposed rhs_transposed : vector<8x4xbf16>, vector<4x8xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}
