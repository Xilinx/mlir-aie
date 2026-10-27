//===- test-contract-layouts-aie2p.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s

// Each AIE2P matmul tile with each stored layout of its operands: lhs [m, k]
// or [k, m], rhs [k, n] or [n, k], acc [m, n] or [n, m]. The matmul op keeps
// the stored layouts in its lhs_transposed and rhs_transposed flags. An acc
// stored [n, m] is computed as rhs^T * lhs^T: the operands swap and both
// flags flip.

// CHECK-LABEL: func.func @bf16_8x8x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x8_mk_kn(%A: vector<8x8xbf16>, %B: vector<8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x8x8_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] rhs_transposed : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x8_mk_nk(%A: vector<8x8xbf16>, %B: vector<8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x8x8_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] lhs_transposed : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x8_km_kn(%A: vector<8x8xbf16>, %B: vector<8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x8x8_km_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] lhs_transposed rhs_transposed : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x8_km_nk(%A: vector<8x8xbf16>, %B: vector<8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x8x8_mk_kn_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] lhs_transposed rhs_transposed : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x8_mk_kn_nm(%A: vector<8x8xbf16>, %B: vector<8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x8x8_mk_nk_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] rhs_transposed : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x8_mk_nk_nm(%A: vector<8x8xbf16>, %B: vector<8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x8x8_km_kn_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] lhs_transposed : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x8_km_kn_nm(%A: vector<8x8xbf16>, %B: vector<8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x8x8_km_nk_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x8_km_nk_nm(%A: vector<8x8xbf16>, %B: vector<8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<4x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] : vector<4x8xbf16>, vector<8x8xbf16> into vector<4x8xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x8_mk_kn(%A: vector<4x8xbf16>, %B: vector<8x8xbf16>, %C: vector<4x8xf32>) -> vector<4x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xbf16>, vector<8x8xbf16> into vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x8_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<4x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] rhs_transposed : vector<4x8xbf16>, vector<8x8xbf16> into vector<4x8xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x8_mk_nk(%A: vector<4x8xbf16>, %B: vector<8x8xbf16>, %C: vector<4x8xf32>) -> vector<4x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xbf16>, vector<8x8xbf16> into vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x8_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<4x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] lhs_transposed : vector<8x4xbf16>, vector<8x8xbf16> into vector<4x8xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x8_km_kn(%A: vector<8x4xbf16>, %B: vector<8x8xbf16>, %C: vector<4x8xf32>) -> vector<4x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x4xbf16>, vector<8x8xbf16> into vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x8_km_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<4x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] lhs_transposed rhs_transposed : vector<8x4xbf16>, vector<8x8xbf16> into vector<4x8xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x8_km_nk(%A: vector<8x4xbf16>, %B: vector<8x8xbf16>, %C: vector<4x8xf32>) -> vector<4x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x4xbf16>, vector<8x8xbf16> into vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x8_mk_kn_nm(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] lhs_transposed rhs_transposed : vector<8x8xbf16>, vector<4x8xbf16> into vector<8x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x8_mk_kn_nm(%A: vector<4x8xbf16>, %B: vector<8x8xbf16>, %C: vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xbf16>, vector<8x8xbf16> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x8_mk_nk_nm(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] rhs_transposed : vector<8x8xbf16>, vector<4x8xbf16> into vector<8x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x8_mk_nk_nm(%A: vector<4x8xbf16>, %B: vector<8x8xbf16>, %C: vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xbf16>, vector<8x8xbf16> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x8_km_kn_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] lhs_transposed : vector<8x8xbf16>, vector<8x4xbf16> into vector<8x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x8_km_kn_nm(%A: vector<8x4xbf16>, %B: vector<8x8xbf16>, %C: vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x4xbf16>, vector<8x8xbf16> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x8_km_nk_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] : vector<8x8xbf16>, vector<8x4xbf16> into vector<8x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x8_km_nk_nm(%A: vector<8x4xbf16>, %B: vector<8x8xbf16>, %C: vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x4xbf16>, vector<8x8xbf16> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x8x4_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<8x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] : vector<8x8xbf16>, vector<8x4xbf16> into vector<8x4xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x4_mk_kn(%A: vector<8x8xbf16>, %B: vector<8x4xbf16>, %C: vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<8x4xbf16> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x8x4_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<8x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] rhs_transposed : vector<8x8xbf16>, vector<4x8xbf16> into vector<8x4xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x4_mk_nk(%A: vector<8x8xbf16>, %B: vector<4x8xbf16>, %C: vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<4x8xbf16> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x8x4_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<8x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] lhs_transposed : vector<8x8xbf16>, vector<8x4xbf16> into vector<8x4xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x4_km_kn(%A: vector<8x8xbf16>, %B: vector<8x4xbf16>, %C: vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<8x4xbf16> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x8x4_km_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<8x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] lhs_transposed rhs_transposed : vector<8x8xbf16>, vector<4x8xbf16> into vector<8x4xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x4_km_nk(%A: vector<8x8xbf16>, %B: vector<4x8xbf16>, %C: vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<4x8xbf16> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x8x4_mk_kn_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<4x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] lhs_transposed rhs_transposed : vector<8x4xbf16>, vector<8x8xbf16> into vector<4x8xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x4_mk_kn_nm(%A: vector<8x8xbf16>, %B: vector<8x4xbf16>, %C: vector<4x8xf32>) -> vector<4x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<8x4xbf16> into vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x8x4_mk_nk_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<4x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] rhs_transposed : vector<4x8xbf16>, vector<8x8xbf16> into vector<4x8xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x4_mk_nk_nm(%A: vector<8x8xbf16>, %B: vector<4x8xbf16>, %C: vector<4x8xf32>) -> vector<4x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<4x8xbf16> into vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x8x4_km_kn_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<4x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] lhs_transposed : vector<8x4xbf16>, vector<8x8xbf16> into vector<4x8xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x4_km_kn_nm(%A: vector<8x8xbf16>, %B: vector<8x4xbf16>, %C: vector<4x8xf32>) -> vector<4x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<8x4xbf16> into vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x8x4_km_nk_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<4x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] : vector<4x8xbf16>, vector<8x8xbf16> into vector<4x8xf32>
// CHECK: return %[[R]]
func.func @bf16_8x8x4_km_nk_nm(%A: vector<8x8xbf16>, %B: vector<4x8xbf16>, %C: vector<4x8xf32>) -> vector<4x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xbf16>, vector<4x8xbf16> into vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_mk_kn(%A: vector<4x8xbf16>, %B: vector<8x4xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] rhs_transposed : vector<4x8xbf16>, vector<4x8xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_mk_nk(%A: vector<4x8xbf16>, %B: vector<4x8xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xbf16>, vector<4x8xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] lhs_transposed : vector<8x4xbf16>, vector<8x4xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_km_kn(%A: vector<8x4xbf16>, %B: vector<8x4xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x4xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_km_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] lhs_transposed rhs_transposed : vector<8x4xbf16>, vector<4x8xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_km_nk(%A: vector<8x4xbf16>, %B: vector<4x8xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x4xbf16>, vector<4x8xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_mk_kn_nm(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] lhs_transposed rhs_transposed : vector<8x4xbf16>, vector<4x8xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_mk_kn_nm(%A: vector<4x8xbf16>, %B: vector<8x4xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_mk_nk_nm(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] rhs_transposed : vector<4x8xbf16>, vector<4x8xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_mk_nk_nm(%A: vector<4x8xbf16>, %B: vector<4x8xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xbf16>, vector<4x8xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_km_kn_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] lhs_transposed : vector<8x4xbf16>, vector<8x4xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_km_kn_nm(%A: vector<8x4xbf16>, %B: vector<8x4xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x4xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_km_nk_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_km_nk_nm(%A: vector<8x4xbf16>, %B: vector<4x8xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x4xbf16>, vector<4x8xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @i8_8x8x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i8_8x8x8_mk_kn(%A: vector<8x8xi8>, %B: vector<8x8xi8>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i8_8x8x8_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] rhs_transposed : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i8_8x8x8_mk_nk(%A: vector<8x8xi8>, %B: vector<8x8xi8>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i8_8x8x8_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] lhs_transposed : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i8_8x8x8_km_kn(%A: vector<8x8xi8>, %B: vector<8x8xi8>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i8_8x8x8_km_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] lhs_transposed rhs_transposed : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i8_8x8x8_km_nk(%A: vector<8x8xi8>, %B: vector<8x8xi8>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i8_8x8x8_mk_kn_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] lhs_transposed rhs_transposed : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i8_8x8x8_mk_kn_nm(%A: vector<8x8xi8>, %B: vector<8x8xi8>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i8_8x8x8_mk_nk_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] rhs_transposed : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i8_8x8x8_mk_nk_nm(%A: vector<8x8xi8>, %B: vector<8x8xi8>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i8_8x8x8_km_kn_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] lhs_transposed : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i8_8x8x8_km_kn_nm(%A: vector<8x8xi8>, %B: vector<8x8xi8>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i8_8x8x8_km_nk_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i8_8x8x8_km_nk_nm(%A: vector<8x8xi8>, %B: vector<8x8xi8>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_8x2x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x2xi16>, %[[B:.*]]: vector<2x8xi16>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] : vector<8x2xi16>, vector<2x8xi16> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i16_8x2x8_mk_kn(%A: vector<8x2xi16>, %B: vector<2x8xi16>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x2xi16>, vector<2x8xi16> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_8x2x8_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x2xi16>, %[[B:.*]]: vector<8x2xi16>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] rhs_transposed : vector<8x2xi16>, vector<8x2xi16> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i16_8x2x8_mk_nk(%A: vector<8x2xi16>, %B: vector<8x2xi16>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x2xi16>, vector<8x2xi16> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_8x2x8_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<2x8xi16>, %[[B:.*]]: vector<2x8xi16>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] lhs_transposed : vector<2x8xi16>, vector<2x8xi16> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i16_8x2x8_km_kn(%A: vector<2x8xi16>, %B: vector<2x8xi16>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<2x8xi16>, vector<2x8xi16> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_8x2x8_km_nk(
// CHECK-SAME: %[[A:.*]]: vector<2x8xi16>, %[[B:.*]]: vector<8x2xi16>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] lhs_transposed rhs_transposed : vector<2x8xi16>, vector<8x2xi16> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i16_8x2x8_km_nk(%A: vector<2x8xi16>, %B: vector<8x2xi16>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<2x8xi16>, vector<8x2xi16> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_8x2x8_mk_kn_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x2xi16>, %[[B:.*]]: vector<2x8xi16>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] lhs_transposed rhs_transposed : vector<2x8xi16>, vector<8x2xi16> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i16_8x2x8_mk_kn_nm(%A: vector<8x2xi16>, %B: vector<2x8xi16>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x2xi16>, vector<2x8xi16> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_8x2x8_mk_nk_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x2xi16>, %[[B:.*]]: vector<8x2xi16>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] rhs_transposed : vector<8x2xi16>, vector<8x2xi16> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i16_8x2x8_mk_nk_nm(%A: vector<8x2xi16>, %B: vector<8x2xi16>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x2xi16>, vector<8x2xi16> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_8x2x8_km_kn_nm(
// CHECK-SAME: %[[A:.*]]: vector<2x8xi16>, %[[B:.*]]: vector<2x8xi16>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] lhs_transposed : vector<2x8xi16>, vector<2x8xi16> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i16_8x2x8_km_kn_nm(%A: vector<2x8xi16>, %B: vector<2x8xi16>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<2x8xi16>, vector<2x8xi16> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_8x2x8_km_nk_nm(
// CHECK-SAME: %[[A:.*]]: vector<2x8xi16>, %[[B:.*]]: vector<8x2xi16>, %[[C:.*]]: vector<8x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[B]], %[[A]], %[[C]] : vector<8x2xi16>, vector<2x8xi16> into vector<8x8xi32>
// CHECK: return %[[R]]
func.func @i16_8x2x8_km_nk_nm(%A: vector<2x8xi16>, %B: vector<8x2xi16>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<2x8xi16>, vector<8x2xi16> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// -----

// With k = 1 the operands have one layout.
// CHECK-LABEL: func.func @bf16_8x1x8(
// CHECK-SAME: %[[A:.*]]: vector<8x1xbf16>, %[[B:.*]]: vector<1x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK: %[[R:.*]] = aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] : vector<8x1xbf16>, vector<1x8xbf16> into vector<8x8xf32>
// CHECK: return %[[R]]
func.func @bf16_8x1x8(%A: vector<8x1xbf16>, %B: vector<1x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x1xbf16>, vector<1x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_8x1x8_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x1xbf16>, %[[B:.*]]: vector<1x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK-DAG: %[[L:.*]] = vector.shape_cast %[[B]] : vector<1x8xbf16> to vector<8x1xbf16>
// CHECK-DAG: %[[R:.*]] = vector.shape_cast %[[A]] : vector<8x1xbf16> to vector<1x8xbf16>
// CHECK: aievec.matmul_aie2p %[[L]], %[[R]], %[[C]] : vector<8x1xbf16>, vector<1x8xbf16> into vector<8x8xf32>
func.func @bf16_8x1x8_nm(%A: vector<8x1xbf16>, %B: vector<1x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x1xbf16>, vector<1x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// A packed tiling's 8x8x8 tile with a unit k-block dim d0 leading every
// operand: rhs stored [n, k] is what bfp16 reads, so it is taken as it is.
// CHECK-LABEL: func.func @unit_kblock_rhs_nk(
// CHECK-SAME: %[[A:.*]]: vector<1x8x8xbf16>, %[[B:.*]]: vector<1x8x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK-DAG: %[[L:.*]] = vector.shape_cast %[[A]] : vector<1x8x8xbf16> to vector<8x8xbf16>
// CHECK-DAG: %[[R:.*]] = vector.shape_cast %[[B]] : vector<1x8x8xbf16> to vector<8x8xbf16>
// CHECK: aievec.matmul_aie2p %[[L]], %[[R]], %[[C]] rhs_transposed : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
#a = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#bt = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#c = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
func.func @unit_kblock_rhs_nk(%A: vector<1x8x8xbf16>, %B: vector<1x8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [#a, #bt, #c], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<1x8x8xbf16>, vector<1x8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// The same tile with rhs stored [k, n].
// CHECK-LABEL: func.func @unit_kblock_rhs_kn(
// CHECK: aievec.matmul_aie2p %{{.*}}, %{{.*}}, %{{.*}} : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
#a = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#b = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#c = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
func.func @unit_kblock_rhs_kn(%A: vector<1x8x8xbf16>, %B: vector<1x8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract {indexing_maps = [#a, #b, #c], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<1x8x8xbf16>, vector<1x8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// Unit dims between and after the m, n and k dims, of both iterator kinds:
// lhs [k, 1, m], rhs [n, k, 1], acc [m, 1, n].
// CHECK-LABEL: func.func @unit_dims_inside(
// CHECK-SAME: %[[A:.*]]: vector<8x1x8xbf16>, %[[B:.*]]: vector<8x8x1xbf16>, %[[C:.*]]: vector<8x1x8xf32>)
// CHECK-DAG: %[[ACC:.*]] = vector.shape_cast %[[C]] : vector<8x1x8xf32> to vector<8x8xf32>
// CHECK-DAG: %[[L:.*]] = vector.shape_cast %[[A]] : vector<8x1x8xbf16> to vector<8x8xbf16>
// CHECK-DAG: %[[R:.*]] = vector.shape_cast %[[B]] : vector<8x8x1xbf16> to vector<8x8xbf16>
// CHECK: %[[MM:.*]] = aievec.matmul_aie2p %[[L]], %[[R]], %[[ACC]] lhs_transposed rhs_transposed : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
// CHECK: %[[RES:.*]] = vector.shape_cast %[[MM]] : vector<8x8xf32> to vector<8x1x8xf32>
// CHECK: return %[[RES]]
#a = affine_map<(d0, d1, d2, d3, d4) -> (d3, d1, d0)>
#b = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#c = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
func.func @unit_dims_inside(%A: vector<8x1x8xbf16>, %B: vector<8x8x1xbf16>, %C: vector<8x1x8xf32>) -> vector<8x1x8xf32> {
  %0 = vector.contract {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x1x8xbf16>, vector<8x8x1xbf16> into vector<8x1x8xf32>
  return %0 : vector<8x1x8xf32>
}

// -----

// A unit batch dim in every operand and an acc stored [n, m]: the 4x8x8
// contraction is computed as the 8x8x4 rhs^T * lhs^T.
// CHECK-LABEL: func.func @unit_batch_acc_nm(
// CHECK-SAME: %[[A:.*]]: vector<1x4x8xbf16>, %[[B:.*]]: vector<1x8x8xbf16>, %[[C:.*]]: vector<1x8x4xf32>)
// CHECK-DAG: %[[ACC:.*]] = vector.shape_cast %[[C]] : vector<1x8x4xf32> to vector<8x4xf32>
// CHECK-DAG: %[[L:.*]] = vector.shape_cast %[[B]] : vector<1x8x8xbf16> to vector<8x8xbf16>
// CHECK-DAG: %[[R:.*]] = vector.shape_cast %[[A]] : vector<1x4x8xbf16> to vector<4x8xbf16>
// CHECK: %[[MM:.*]] = aievec.matmul_aie2p %[[L]], %[[R]], %[[ACC]] lhs_transposed rhs_transposed : vector<8x8xbf16>, vector<4x8xbf16> into vector<8x4xf32>
// CHECK: vector.shape_cast %[[MM]] : vector<8x4xf32> to vector<1x8x4xf32>
#a = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#b = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#c = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
func.func @unit_batch_acc_nm(%A: vector<1x4x8xbf16>, %B: vector<1x8x8xbf16>, %C: vector<1x8x4xf32>) -> vector<1x8x4xf32> {
  %0 = vector.contract {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<1x4x8xbf16>, vector<1x8x8xbf16> into vector<1x8x4xf32>
  return %0 : vector<1x8x4xf32>
}

// -----

// f32 operands widened from bf16 are narrowed back; the stored layouts are
// kept.
// CHECK-LABEL: func.func @extf_rhs_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>)
// CHECK: aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] rhs_transposed : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
func.func @extf_rhs_nk(%A: vector<8x8xbf16>, %B: vector<8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  %a = arith.extf %A : vector<8x8xbf16> to vector<8x8xf32>
  %b = arith.extf %B : vector<8x8xbf16> to vector<8x8xf32>
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %a, %b, %C : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// The operands of an acc stored [n, m] swap together with their signedness.
// CHECK-LABEL: func.func @i8_signedness_acc_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<8x8xi32>)
// CHECK-DAG: %[[L:.*]] = builtin.unrealized_conversion_cast %[[B]] : vector<8x8xi8> to vector<8x8xsi8>
// CHECK-DAG: %[[R:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<8x8xi8> to vector<8x8xui8>
// CHECK: aievec.matmul_aie2p %[[L]], %[[R]], %[[C]] lhs_transposed rhs_transposed : vector<8x8xsi8>, vector<8x8xui8> into vector<8x8xi32>
func.func @i8_signedness_acc_nm(%A: vector<8x8xi8>, %B: vector<8x8xi8>, %C: vector<8x8xi32>) -> vector<8x8xi32> {
  %a = arith.extui %A : vector<8x8xi8> to vector<8x8xi32>
  %b = arith.extsi %B : vector<8x8xi8> to vector<8x8xi32>
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %a, %b, %C : vector<8x8xi32>, vector<8x8xi32> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}
