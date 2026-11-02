//===- matmul-layouts-aie2p.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s
// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s --check-prefix=COUNT

// Each AIE2P matmul recipe with each stored operand layout. An operand stored
// the way the recipe reads it is not shuffled. Otherwise it is transposed by
// one vshuffle (512 bits, or 256 bits padded to 512), or by the
// t16_8x8_lo/hi pair (52/53) for an 8x8 bf16 tile. COUNT checks the total
// number of vshuffles, including those a recipe uses to broadcast lhs columns
// (8 for the elementwise bf16 recipes, plus the 52/53 pair that turns the
// padded [m, k] lhs of the 4x8x4 recipe into [k, m]).

// CHECK-LABEL: func.func @bfp16_8x8x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[FB]] : vector<64xbf16> to vector<32xi32>
// CHECK: %[[RTL:.*]] = vector.shuffle %[[RTI]], %[[RTI]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// CHECK: %[[RTH:.*]] = vector.shuffle %[[RTI]], %[[RTI]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
// CHECK: %[[RTM0:.*]] = llvm.mlir.constant(52 : i32) : i32
// CHECK: %[[RTM1:.*]] = llvm.mlir.constant(53 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTL]], %[[RTH]], %[[RTM0]])
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTL]], %[[RTH]], %[[RTM1]])
// COUNT-LABEL: func.func @bfp16_8x8x8_mk_kn(
// COUNT-COUNT-2: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @bfp16_8x8x8_mk_kn(%A : vector<8x8xbf16>, %B : vector<8x8xbf16>, %C : vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @bfp16_8x8x8_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xbf16> to vector<64xbf16>
// COUNT-LABEL: func.func @bfp16_8x8x8_mk_nk(
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @bfp16_8x8x8_mk_nk(%A : vector<8x8xbf16>, %B : vector<8x8xbf16>, %C : vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C rhs_transposed : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @bfp16_8x8x8_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[FA]] : vector<64xbf16> to vector<32xi32>
// CHECK: %[[LTL:.*]] = vector.shuffle %[[LTI]], %[[LTI]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// CHECK: %[[LTH:.*]] = vector.shuffle %[[LTI]], %[[LTI]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
// CHECK: %[[LTM0:.*]] = llvm.mlir.constant(52 : i32) : i32
// CHECK: %[[LTM1:.*]] = llvm.mlir.constant(53 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTL]], %[[LTH]], %[[LTM0]])
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTL]], %[[LTH]], %[[LTM1]])
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[FB]] : vector<64xbf16> to vector<32xi32>
// CHECK: %[[RTL:.*]] = vector.shuffle %[[RTI]], %[[RTI]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// CHECK: %[[RTH:.*]] = vector.shuffle %[[RTI]], %[[RTI]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
// CHECK: %[[RTM0:.*]] = llvm.mlir.constant(52 : i32) : i32
// CHECK: %[[RTM1:.*]] = llvm.mlir.constant(53 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTL]], %[[RTH]], %[[RTM0]])
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTL]], %[[RTH]], %[[RTM1]])
// COUNT-LABEL: func.func @bfp16_8x8x8_km_kn(
// COUNT-COUNT-4: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @bfp16_8x8x8_km_kn(%A : vector<8x8xbf16>, %B : vector<8x8xbf16>, %C : vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C lhs_transposed : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @bfp16_8x8x8_km_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[FA]] : vector<64xbf16> to vector<32xi32>
// CHECK: %[[LTL:.*]] = vector.shuffle %[[LTI]], %[[LTI]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// CHECK: %[[LTH:.*]] = vector.shuffle %[[LTI]], %[[LTI]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
// CHECK: %[[LTM0:.*]] = llvm.mlir.constant(52 : i32) : i32
// CHECK: %[[LTM1:.*]] = llvm.mlir.constant(53 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTL]], %[[LTH]], %[[LTM0]])
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTL]], %[[LTH]], %[[LTM1]])
// COUNT-LABEL: func.func @bfp16_8x8x8_km_nk(
// COUNT-COUNT-2: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @bfp16_8x8x8_km_nk(%A : vector<8x8xbf16>, %B : vector<8x8xbf16>, %C : vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C lhs_transposed rhs_transposed : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @bfp16_4x8x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<4x8xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<4x8xbf16> to vector<32xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[FB]] : vector<64xbf16> to vector<32xi32>
// CHECK: %[[RTL:.*]] = vector.shuffle %[[RTI]], %[[RTI]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// CHECK: %[[RTH:.*]] = vector.shuffle %[[RTI]], %[[RTI]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
// CHECK: %[[RTM0:.*]] = llvm.mlir.constant(52 : i32) : i32
// CHECK: %[[RTM1:.*]] = llvm.mlir.constant(53 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTL]], %[[RTH]], %[[RTM0]])
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTL]], %[[RTH]], %[[RTM1]])
// COUNT-LABEL: func.func @bfp16_4x8x8_mk_kn(
// COUNT-COUNT-2: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @bfp16_4x8x8_mk_kn(%A : vector<4x8xbf16>, %B : vector<8x8xbf16>, %C : vector<4x8xf32>) -> vector<4x8xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C : vector<4x8xbf16>, vector<8x8xbf16> into vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @bfp16_4x8x8_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<4x8xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<4x8xbf16> to vector<32xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xbf16> to vector<64xbf16>
// COUNT-LABEL: func.func @bfp16_4x8x8_mk_nk(
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @bfp16_4x8x8_mk_nk(%A : vector<4x8xbf16>, %B : vector<8x8xbf16>, %C : vector<4x8xf32>) -> vector<4x8xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C rhs_transposed : vector<4x8xbf16>, vector<8x8xbf16> into vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @bfp16_4x8x8_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<4x8xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x4xbf16> to vector<32xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[FA]] : vector<32xbf16> to vector<16xi32>
// CHECK: %[[LTM:.*]] = llvm.mlir.constant(28 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTI]], %[[LTI]], %[[LTM]])
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[FB]] : vector<64xbf16> to vector<32xi32>
// CHECK: %[[RTL:.*]] = vector.shuffle %[[RTI]], %[[RTI]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// CHECK: %[[RTH:.*]] = vector.shuffle %[[RTI]], %[[RTI]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
// CHECK: %[[RTM0:.*]] = llvm.mlir.constant(52 : i32) : i32
// CHECK: %[[RTM1:.*]] = llvm.mlir.constant(53 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTL]], %[[RTH]], %[[RTM0]])
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTL]], %[[RTH]], %[[RTM1]])
// COUNT-LABEL: func.func @bfp16_4x8x8_km_kn(
// COUNT-COUNT-3: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @bfp16_4x8x8_km_kn(%A : vector<8x4xbf16>, %B : vector<8x8xbf16>, %C : vector<4x8xf32>) -> vector<4x8xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C lhs_transposed : vector<8x4xbf16>, vector<8x8xbf16> into vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @bfp16_4x8x8_km_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<4x8xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x4xbf16> to vector<32xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[FA]] : vector<32xbf16> to vector<16xi32>
// CHECK: %[[LTM:.*]] = llvm.mlir.constant(28 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTI]], %[[LTI]], %[[LTM]])
// COUNT-LABEL: func.func @bfp16_4x8x8_km_nk(
// COUNT-COUNT-1: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @bfp16_4x8x8_km_nk(%A : vector<8x4xbf16>, %B : vector<8x8xbf16>, %C : vector<4x8xf32>) -> vector<4x8xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C lhs_transposed rhs_transposed : vector<8x4xbf16>, vector<8x8xbf16> into vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @elem_8x8x4_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<8x4xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x4xbf16> to vector<32xbf16>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[FA]] : vector<64xbf16> to vector<32xi32>
// CHECK: %[[LTL:.*]] = vector.shuffle %[[LTI]], %[[LTI]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// CHECK: %[[LTH:.*]] = vector.shuffle %[[LTI]], %[[LTI]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
// CHECK: %[[LTM0:.*]] = llvm.mlir.constant(52 : i32) : i32
// CHECK: %[[LTM1:.*]] = llvm.mlir.constant(53 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTL]], %[[LTH]], %[[LTM0]])
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTL]], %[[LTH]], %[[LTM1]])
// COUNT-LABEL: func.func @elem_8x8x4_mk_kn(
// COUNT-COUNT-10: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @elem_8x8x4_mk_kn(%A : vector<8x8xbf16>, %B : vector<8x4xbf16>, %C : vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C : vector<8x8xbf16>, vector<8x4xbf16> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

// CHECK-LABEL: func.func @elem_8x8x4_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<8x4xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<4x8xbf16> to vector<32xbf16>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[FA]] : vector<64xbf16> to vector<32xi32>
// CHECK: %[[LTL:.*]] = vector.shuffle %[[LTI]], %[[LTI]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// CHECK: %[[LTH:.*]] = vector.shuffle %[[LTI]], %[[LTI]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
// CHECK: %[[LTM0:.*]] = llvm.mlir.constant(52 : i32) : i32
// CHECK: %[[LTM1:.*]] = llvm.mlir.constant(53 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTL]], %[[LTH]], %[[LTM0]])
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTL]], %[[LTH]], %[[LTM1]])
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[FB]] : vector<32xbf16> to vector<16xi32>
// CHECK: %[[RTM:.*]] = llvm.mlir.constant(29 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTI]], %[[RTI]], %[[RTM]])
// COUNT-LABEL: func.func @elem_8x8x4_mk_nk(
// COUNT-COUNT-11: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @elem_8x8x4_mk_nk(%A : vector<8x8xbf16>, %B : vector<4x8xbf16>, %C : vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C rhs_transposed : vector<8x8xbf16>, vector<4x8xbf16> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

// CHECK-LABEL: func.func @elem_8x8x4_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<8x4xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x4xbf16> to vector<32xbf16>
// COUNT-LABEL: func.func @elem_8x8x4_km_kn(
// COUNT-COUNT-8: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @elem_8x8x4_km_kn(%A : vector<8x8xbf16>, %B : vector<8x4xbf16>, %C : vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C lhs_transposed : vector<8x8xbf16>, vector<8x4xbf16> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

// CHECK-LABEL: func.func @elem_8x8x4_km_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<8x4xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<4x8xbf16> to vector<32xbf16>
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[FB]] : vector<32xbf16> to vector<16xi32>
// CHECK: %[[RTM:.*]] = llvm.mlir.constant(29 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTI]], %[[RTI]], %[[RTM]])
// COUNT-LABEL: func.func @elem_8x8x4_km_nk(
// COUNT-COUNT-9: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @elem_8x8x4_km_nk(%A : vector<8x8xbf16>, %B : vector<4x8xbf16>, %C : vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C lhs_transposed rhs_transposed : vector<8x8xbf16>, vector<4x8xbf16> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

// CHECK-LABEL: func.func @elem_4x8x4_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<4x8xbf16> to vector<32xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x4xbf16> to vector<32xbf16>
// COUNT-LABEL: func.func @elem_4x8x4_mk_kn(
// COUNT-COUNT-10: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @elem_4x8x4_mk_kn(%A : vector<4x8xbf16>, %B : vector<8x4xbf16>, %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @elem_4x8x4_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<4x8xbf16> to vector<32xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<4x8xbf16> to vector<32xbf16>
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[FB]] : vector<32xbf16> to vector<16xi32>
// CHECK: %[[RTM:.*]] = llvm.mlir.constant(29 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTI]], %[[RTI]], %[[RTM]])
// COUNT-LABEL: func.func @elem_4x8x4_mk_nk(
// COUNT-COUNT-11: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @elem_4x8x4_mk_nk(%A : vector<4x8xbf16>, %B : vector<4x8xbf16>, %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C rhs_transposed : vector<4x8xbf16>, vector<4x8xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @elem_4x8x4_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x4xbf16> to vector<32xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x4xbf16> to vector<32xbf16>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[FA]] : vector<32xbf16> to vector<16xi32>
// CHECK: %[[LTM:.*]] = llvm.mlir.constant(28 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTI]], %[[LTI]], %[[LTM]])
// COUNT-LABEL: func.func @elem_4x8x4_km_kn(
// COUNT-COUNT-11: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @elem_4x8x4_km_kn(%A : vector<8x4xbf16>, %B : vector<8x4xbf16>, %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C lhs_transposed : vector<8x4xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @elem_4x8x4_km_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x4xbf16> to vector<32xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<4x8xbf16> to vector<32xbf16>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[FA]] : vector<32xbf16> to vector<16xi32>
// CHECK: %[[LTM:.*]] = llvm.mlir.constant(28 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTI]], %[[LTI]], %[[LTM]])
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[FB]] : vector<32xbf16> to vector<16xi32>
// CHECK: %[[RTM:.*]] = llvm.mlir.constant(29 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTI]], %[[RTI]], %[[RTM]])
// COUNT-LABEL: func.func @elem_4x8x4_km_nk(
// COUNT-COUNT-12: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @elem_4x8x4_km_nk(%A : vector<8x4xbf16>, %B : vector<4x8xbf16>, %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C lhs_transposed rhs_transposed : vector<8x4xbf16>, vector<4x8xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @i8_8x8x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<8x8xi32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x8xi8> to vector<64xi8>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xi8> to vector<64xi8>
// COUNT-LABEL: func.func @i8_8x8x8_mk_kn(
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @i8_8x8x8_mk_kn(%A : vector<8x8xi8>, %B : vector<8x8xi8>, %C : vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = aievec.matmul_aie2p %A, %B, %C : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i8_8x8x8_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<8x8xi32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x8xi8> to vector<64xi8>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xi8> to vector<64xi8>
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[FB]] : vector<64xi8> to vector<16xi32>
// CHECK: %[[RTM:.*]] = llvm.mlir.constant(35 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTI]], %[[RTI]], %[[RTM]])
// COUNT-LABEL: func.func @i8_8x8x8_mk_nk(
// COUNT-COUNT-1: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @i8_8x8x8_mk_nk(%A : vector<8x8xi8>, %B : vector<8x8xi8>, %C : vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = aievec.matmul_aie2p %A, %B, %C rhs_transposed : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i8_8x8x8_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<8x8xi32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x8xi8> to vector<64xi8>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xi8> to vector<64xi8>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[FA]] : vector<64xi8> to vector<16xi32>
// CHECK: %[[LTM:.*]] = llvm.mlir.constant(35 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTI]], %[[LTI]], %[[LTM]])
// COUNT-LABEL: func.func @i8_8x8x8_km_kn(
// COUNT-COUNT-1: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @i8_8x8x8_km_kn(%A : vector<8x8xi8>, %B : vector<8x8xi8>, %C : vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = aievec.matmul_aie2p %A, %B, %C lhs_transposed : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i8_8x8x8_km_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<8x8xi32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x8xi8> to vector<64xi8>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xi8> to vector<64xi8>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[FA]] : vector<64xi8> to vector<16xi32>
// CHECK: %[[LTM:.*]] = llvm.mlir.constant(35 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTI]], %[[LTI]], %[[LTM]])
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[FB]] : vector<64xi8> to vector<16xi32>
// CHECK: %[[RTM:.*]] = llvm.mlir.constant(35 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTI]], %[[RTI]], %[[RTM]])
// COUNT-LABEL: func.func @i8_8x8x8_km_nk(
// COUNT-COUNT-2: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @i8_8x8x8_km_nk(%A : vector<8x8xi8>, %B : vector<8x8xi8>, %C : vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = aievec.matmul_aie2p %A, %B, %C lhs_transposed rhs_transposed : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_8x2x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x2xi16>, %[[B:.*]]: vector<2x8xi16>, %[[C:.*]]: vector<8x8xi32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x2xi16> to vector<16xi16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<2x8xi16> to vector<16xi16>
// COUNT-LABEL: func.func @i16_8x2x8_mk_kn(
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @i16_8x2x8_mk_kn(%A : vector<8x2xi16>, %B : vector<2x8xi16>, %C : vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = aievec.matmul_aie2p %A, %B, %C : vector<8x2xi16>, vector<2x8xi16> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_8x2x8_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x2xi16>, %[[B:.*]]: vector<8x2xi16>, %[[C:.*]]: vector<8x8xi32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x2xi16> to vector<16xi16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x2xi16> to vector<16xi16>
// CHECK: %[[RTP:.*]] = vector.shuffle %[[FB]], %[[FB]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] : vector<16xi16>, vector<16xi16>
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[RTP]] : vector<32xi16> to vector<16xi32>
// CHECK: %[[RTM:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTI]], %[[RTI]], %[[RTM]])
// COUNT-LABEL: func.func @i16_8x2x8_mk_nk(
// COUNT-COUNT-1: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @i16_8x2x8_mk_nk(%A : vector<8x2xi16>, %B : vector<8x2xi16>, %C : vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = aievec.matmul_aie2p %A, %B, %C rhs_transposed : vector<8x2xi16>, vector<8x2xi16> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_8x2x8_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<2x8xi16>, %[[B:.*]]: vector<2x8xi16>, %[[C:.*]]: vector<8x8xi32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<2x8xi16> to vector<16xi16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<2x8xi16> to vector<16xi16>
// CHECK: %[[LTP:.*]] = vector.shuffle %[[FA]], %[[FA]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] : vector<16xi16>, vector<16xi16>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[LTP]] : vector<32xi16> to vector<16xi32>
// CHECK: %[[LTM:.*]] = llvm.mlir.constant(43 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTI]], %[[LTI]], %[[LTM]])
// COUNT-LABEL: func.func @i16_8x2x8_km_kn(
// COUNT-COUNT-1: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @i16_8x2x8_km_kn(%A : vector<2x8xi16>, %B : vector<2x8xi16>, %C : vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = aievec.matmul_aie2p %A, %B, %C lhs_transposed : vector<2x8xi16>, vector<2x8xi16> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_8x2x8_km_nk(
// CHECK-SAME: %[[A:.*]]: vector<2x8xi16>, %[[B:.*]]: vector<8x2xi16>, %[[C:.*]]: vector<8x8xi32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<2x8xi16> to vector<16xi16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x2xi16> to vector<16xi16>
// CHECK: %[[LTP:.*]] = vector.shuffle %[[FA]], %[[FA]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] : vector<16xi16>, vector<16xi16>
// CHECK: %[[LTI:.*]] = llvm.bitcast %[[LTP]] : vector<32xi16> to vector<16xi32>
// CHECK: %[[LTM:.*]] = llvm.mlir.constant(43 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[LTI]], %[[LTI]], %[[LTM]])
// CHECK: %[[RTP:.*]] = vector.shuffle %[[FB]], %[[FB]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] : vector<16xi16>, vector<16xi16>
// CHECK: %[[RTI:.*]] = llvm.bitcast %[[RTP]] : vector<32xi16> to vector<16xi32>
// CHECK: %[[RTM:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[RTI]], %[[RTI]], %[[RTM]])
// COUNT-LABEL: func.func @i16_8x2x8_km_nk(
// COUNT-COUNT-2: "xllvm.intr.aie2p.vshuffle"
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @i16_8x2x8_km_nk(%A : vector<2x8xi16>, %B : vector<8x2xi16>, %C : vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = aievec.matmul_aie2p %A, %B, %C lhs_transposed rhs_transposed : vector<2x8xi16>, vector<8x2xi16> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// With k = 1 a layout flag moves no data.
// CHECK-LABEL: func.func @outer_8x1x8(
// CHECK-SAME: %[[A:.*]]: vector<8x1xbf16>, %[[B:.*]]: vector<1x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x1xbf16> to vector<8xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<1x8xbf16> to vector<8xbf16>
// COUNT-LABEL: func.func @outer_8x1x8(
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @outer_8x1x8(%A : vector<8x1xbf16>, %B : vector<1x8xbf16>, %C : vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C : vector<8x1xbf16>, vector<1x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @outer_8x1x8_flags(
// CHECK-SAME: %[[A:.*]]: vector<1x8xbf16>, %[[B:.*]]: vector<8x1xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK-DAG: %[[FA:.*]] = vector.shape_cast %[[A]] : vector<1x8xbf16> to vector<8xbf16>
// CHECK-DAG: %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x1xbf16> to vector<8xbf16>
// COUNT-LABEL: func.func @outer_8x1x8_flags(
// COUNT-NOT: "xllvm.intr.aie2p.vshuffle"
func.func @outer_8x1x8_flags(%A : vector<1x8xbf16>, %B : vector<8x1xbf16>, %C : vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C lhs_transposed rhs_transposed : vector<1x8xbf16>, vector<8x1xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}
