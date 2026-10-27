//===- test-contract-layouts-to-llvm-aie2.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie2" | aie-opt -split-input-file --convert-aievec-to-llvm | FileCheck %s

// Square tiles in stored layouts other than [m, k] x [k, n], from
// vector.contract to the MAC: each operand is shuffled exactly when the
// multiply reads it the other way.

// acc stored [n, m] is computed as rhs^T * lhs^T: one t16_8x4 shuffle for
// rhs, which becomes a [k, m] lhs, and one t16_4x8 for lhs, which becomes an
// [n, k] rhs.
// CHECK-LABEL: func.func @aie2_bf16_acc_nm(
// CHECK:       llvm.mlir.constant(28 : i32)
// CHECK:       "xllvm.intr.aie2.vshuffle"
// CHECK:       llvm.mlir.constant(29 : i32)
// CHECK:       "xllvm.intr.aie2.vshuffle"
// CHECK-NOT:   vshuffle
// CHECK:       "xllvm.intr.aie2.bf.mac16.conf"
func.func @aie2_bf16_acc_nm(%A: vector<4x8xbf16>, %B: vector<8x4xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// rhs stored [n, k]: one t16_4x8 shuffle.
// CHECK-LABEL: func.func @aie2_bf16_rhs_nk(
// CHECK:       llvm.mlir.constant(29 : i32)
// CHECK-COUNT-1: "xllvm.intr.aie2.vshuffle"
// CHECK-NOT:   vshuffle
// CHECK:       "xllvm.intr.aie2.bf.mac16.conf"
func.func @aie2_bf16_rhs_nk(%A: vector<4x8xbf16>, %B: vector<4x8xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xbf16>, vector<4x8xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}
