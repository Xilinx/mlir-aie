//===- test-contract-layouts-to-llvm-aie2p.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie2p" | aie-opt -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

// Square tiles in stored layouts other than [m, k] x [k, n], from
// vector.contract to the MAC: each operand is shuffled exactly when the
// multiply reads it the other way.

// rhs stored [n, k] behind a unit k-block dim is what bfp16 reads.
// CHECK-LABEL: func.func @bf16_rhs_nk(
// CHECK-NOT:   vshuffle
// CHECK:       "xllvm.intr.aie2p.BFP576.BFP576.ACC2048.mac.conf"
#a = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#bt = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#c = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
func.func @bf16_rhs_nk(%A: vector<1x8x8xbf16>, %B: vector<1x8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [#a, #bt, #c], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<1x8x8xbf16>, vector<1x8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// rhs stored [k, n] behind a unit k-block dim is transposed by the 52/53
// pair.
// CHECK-LABEL: func.func @bf16_rhs_kn(
// CHECK-COUNT-2: "xllvm.intr.aie2p.vshuffle"
// CHECK-NOT:   vshuffle
// CHECK:       "xllvm.intr.aie2p.BFP576.BFP576.ACC2048.mac.conf"
#a = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#b = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#c = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
func.func @bf16_rhs_kn(%A: vector<1x8x8xbf16>, %B: vector<1x8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [#a, #b, #c], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<1x8x8xbf16>, vector<1x8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// lhs stored [k, m] and rhs [n, k]: only the lhs moves.
// CHECK-LABEL: func.func @bf16_lhs_km_rhs_nk(
// CHECK-SAME:  %[[A:[a-zA-Z0-9]+]]: vector<8x8xbf16>
// CHECK:       %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK:       llvm.bitcast %[[FA]] : vector<64xbf16> to vector<32xi32>
// CHECK-COUNT-2: "xllvm.intr.aie2p.vshuffle"
// CHECK-NOT:   vshuffle
// CHECK:       "xllvm.intr.aie2p.BFP576.BFP576.ACC2048.mac.conf"
func.func @bf16_lhs_km_rhs_nk(%A: vector<8x8xbf16>, %B: vector<8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// acc stored [n, m] with lhs [m, k] and rhs [k, n] is computed as
// rhs^T * lhs^T: lhs becomes an [n, k] rhs, as bfp16 reads it, and only rhs,
// which becomes a [k, m] lhs, moves.
// CHECK-LABEL: func.func @bf16_acc_nm(
// CHECK-SAME:  %[[A:[a-zA-Z0-9]+]]: vector<8x8xbf16>, %[[B:[a-zA-Z0-9]+]]: vector<8x8xbf16>
// CHECK:       %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK:       llvm.bitcast %[[FB]] : vector<64xbf16> to vector<32xi32>
// CHECK-COUNT-2: "xllvm.intr.aie2p.vshuffle"
// CHECK-NOT:   vshuffle
// CHECK:       "xllvm.intr.aie2p.BFP576.BFP576.ACC2048.mac.conf"
func.func @bf16_acc_nm(%A: vector<8x8xbf16>, %B: vector<8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// A square i8 tile with lhs stored [k, m] takes one t8_8x8 shuffle.
// CHECK-LABEL: func.func @i8_lhs_km(
// CHECK:       llvm.mlir.constant(35 : i32)
// CHECK-COUNT-1: "xllvm.intr.aie2p.vshuffle"
// CHECK-NOT:   vshuffle
// CHECK:       "xllvm.intr.aie2p.I512.I512.ACC2048.mac.conf"
func.func @i8_lhs_km(%A: vector<8x8xi8>, %B: vector<8x8xi8>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %a = arith.extsi %A : vector<8x8xi8> to vector<8x8xi32>
  %b = arith.extsi %B : vector<8x8xi8> to vector<8x8xi32>
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %a, %b, %C : vector<8x8xi32>, vector<8x8xi32> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}
