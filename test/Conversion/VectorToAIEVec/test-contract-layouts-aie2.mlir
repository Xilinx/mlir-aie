//===- test-contract-layouts-aie2.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s

// AIE2 matmul tiles with each stored layout that the lowering can bring into
// the [m, k] x [k, n] layout the MAC reads, with one 512-bit shuffle. The
// matmul op keeps the stored layouts in its lhs_transposed and
// rhs_transposed flags. An acc stored [n, m] is computed as rhs^T * lhs^T.
// The layouts that need any other transpose are rejected, see
// test-contract-layouts-unsupported-aie2.mlir. i8 x i4 is not reached from
// vector.contract, as the arith.extsi from i4 is rewritten first; its layouts
// are tested on aievec.matmul in AIEVecToLLVM/matmul-layouts-aie2.mlir.

// CHECK-LABEL: func.func @i8_4x8x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<4x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<4x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] : vector<4x8xi8>, vector<8x8xi8> into vector<4x8xi32>
// CHECK: return %[[R]]
func.func @i8_4x8x8_mk_kn(%A: vector<4x8xi8>, %B: vector<8x8xi8>, %C: vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xi8>, vector<8x8xi8> into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

// CHECK-LABEL: func.func @i8_4x8x8_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<4x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<4x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] rhs_transposed : vector<4x8xi8>, vector<8x8xi8> into vector<4x8xi32>
// CHECK: return %[[R]]
func.func @i8_4x8x8_mk_nk(%A: vector<4x8xi8>, %B: vector<8x8xi8>, %C: vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xi8>, vector<8x8xi8> into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_i8_4x4x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<4x4xi16>, %[[B:.*]]: vector<4x8xi8>, %[[C:.*]]: vector<4x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %{{.*}}, %[[C]] : vector<4x4xi16>, vector<4x8xsi8> into vector<4x8xi32>
// CHECK: return %[[R]]
func.func @i16_i8_4x4x8_mk_kn(%A: vector<4x4xi16>, %B: vector<4x8xi8>, %C: vector<4x8xi32>) -> vector<4x8xi32> {
  %b = arith.extsi %B : vector<4x8xi8> to vector<4x8xi16>
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %b, %C : vector<4x4xi16>, vector<4x8xi16> into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_4x2x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<4x2xi16>, %[[B:.*]]: vector<2x8xi16>, %[[C:.*]]: vector<4x8xi32>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] : vector<4x2xi16>, vector<2x8xi16> into vector<4x8xi32>
// CHECK: return %[[R]]
func.func @i16_4x2x8_mk_kn(%A: vector<4x2xi16>, %B: vector<2x8xi16>, %C: vector<4x8xi32>) -> vector<4x8xi32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x2xi16>, vector<2x8xi16> into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

// CHECK-LABEL: func.func @i16_i8_2x8x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<2x8xi16>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<2x8xi64>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %{{.*}}, %[[C]] : vector<2x8xi16>, vector<8x8xsi8> into vector<2x8xi64>
// CHECK: return %[[R]]
func.func @i16_i8_2x8x8_mk_kn(%A: vector<2x8xi16>, %B: vector<8x8xi8>, %C: vector<2x8xi64>) -> vector<2x8xi64> {
  %b = arith.extsi %B : vector<8x8xi8> to vector<8x8xi16>
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %b, %C : vector<2x8xi16>, vector<8x8xi16> into vector<2x8xi64>
  return %0 : vector<2x8xi64>
}

// -----

// CHECK-LABEL: func.func @i16_i8_2x8x8_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<2x8xi16>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<2x8xi64>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %{{.*}}, %[[C]] rhs_transposed : vector<2x8xi16>, vector<8x8xsi8> into vector<2x8xi64>
// CHECK: return %[[R]]
func.func @i16_i8_2x8x8_mk_nk(%A: vector<2x8xi16>, %B: vector<8x8xi8>, %C: vector<2x8xi64>) -> vector<2x8xi64> {
  %b = arith.extsi %B : vector<8x8xi8> to vector<8x8xi16>
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %b, %C : vector<2x8xi16>, vector<8x8xi16> into vector<2x8xi64>
  return %0 : vector<2x8xi64>
}

// -----

// CHECK-LABEL: func.func @i16_i8_4x8x4_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<4x8xi16>, %[[B:.*]]: vector<8x4xi8>, %[[C:.*]]: vector<4x4xi64>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %{{.*}}, %[[C]] : vector<4x8xi16>, vector<8x4xsi8> into vector<4x4xi64>
// CHECK: return %[[R]]
func.func @i16_i8_4x8x4_mk_kn(%A: vector<4x8xi16>, %B: vector<8x4xi8>, %C: vector<4x4xi64>) -> vector<4x4xi64> {
  %b = arith.extsi %B : vector<8x4xi8> to vector<8x4xi16>
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %b, %C : vector<4x8xi16>, vector<8x4xi16> into vector<4x4xi64>
  return %0 : vector<4x4xi64>
}

// -----

// CHECK-LABEL: func.func @i16_i8_4x8x4_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x4xi16>, %[[B:.*]]: vector<8x4xi8>, %[[C:.*]]: vector<4x4xi64>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %{{.*}}, %[[C]] lhs_transposed : vector<8x4xi16>, vector<8x4xsi8> into vector<4x4xi64>
// CHECK: return %[[R]]
func.func @i16_i8_4x8x4_km_kn(%A: vector<8x4xi16>, %B: vector<8x4xi8>, %C: vector<4x4xi64>) -> vector<4x4xi64> {
  %b = arith.extsi %B : vector<8x4xi8> to vector<8x4xi16>
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %b, %C : vector<8x4xi16>, vector<8x4xi16> into vector<4x4xi64>
  return %0 : vector<4x4xi64>
}

// -----

// CHECK-LABEL: func.func @i16_2x4x8_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<2x4xi16>, %[[B:.*]]: vector<4x8xi16>, %[[C:.*]]: vector<2x8xi64>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] : vector<2x4xi16>, vector<4x8xi16> into vector<2x8xi64>
// CHECK: return %[[R]]
func.func @i16_2x4x8_mk_kn(%A: vector<2x4xi16>, %B: vector<4x8xi16>, %C: vector<2x8xi64>) -> vector<2x8xi64> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<2x4xi16>, vector<4x8xi16> into vector<2x8xi64>
  return %0 : vector<2x8xi64>
}

// -----

// CHECK-LABEL: func.func @i16_2x4x8_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<2x4xi16>, %[[B:.*]]: vector<8x4xi16>, %[[C:.*]]: vector<2x8xi64>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] rhs_transposed : vector<2x4xi16>, vector<8x4xi16> into vector<2x8xi64>
// CHECK: return %[[R]]
func.func @i16_2x4x8_mk_nk(%A: vector<2x4xi16>, %B: vector<8x4xi16>, %C: vector<2x8xi64>) -> vector<2x8xi64> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<2x4xi16>, vector<8x4xi16> into vector<2x8xi64>
  return %0 : vector<2x8xi64>
}

// -----

// CHECK-LABEL: func.func @i16_4x4x4_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<4x4xi16>, %[[B:.*]]: vector<4x4xi16>, %[[C:.*]]: vector<4x4xi64>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] : vector<4x4xi16>, vector<4x4xi16> into vector<4x4xi64>
// CHECK: return %[[R]]
func.func @i16_4x4x4_mk_kn(%A: vector<4x4xi16>, %B: vector<4x4xi16>, %C: vector<4x4xi64>) -> vector<4x4xi64> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x4xi16>, vector<4x4xi16> into vector<4x4xi64>
  return %0 : vector<4x4xi64>
}

// -----

// CHECK-LABEL: func.func @i32_i16_4x2x4_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<4x2xi32>, %[[B:.*]]: vector<2x4xi16>, %[[C:.*]]: vector<4x4xi64>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %{{.*}}, %[[C]] : vector<4x2xi32>, vector<2x4xsi16> into vector<4x4xi64>
// CHECK: return %[[R]]
func.func @i32_i16_4x2x4_mk_kn(%A: vector<4x2xi32>, %B: vector<2x4xi16>, %C: vector<4x4xi64>) -> vector<4x4xi64> {
  %b = arith.extsi %B : vector<2x4xi16> to vector<2x4xi32>
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %b, %C : vector<4x2xi32>, vector<2x4xi32> into vector<4x4xi64>
  return %0 : vector<4x4xi64>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_mk_kn(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_mk_kn(%A: vector<4x8xbf16>, %B: vector<8x4xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_mk_nk(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] rhs_transposed : vector<4x8xbf16>, vector<4x8xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_mk_nk(%A: vector<4x8xbf16>, %B: vector<4x8xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xbf16>, vector<4x8xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_km_kn(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] lhs_transposed : vector<8x4xbf16>, vector<8x4xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_km_kn(%A: vector<8x4xbf16>, %B: vector<8x4xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x4xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_km_nk(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul %[[A]], %[[B]], %[[C]] lhs_transposed rhs_transposed : vector<8x4xbf16>, vector<4x8xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_km_nk(%A: vector<8x4xbf16>, %B: vector<4x8xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x4xbf16>, vector<4x8xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_mk_kn_nm(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul %[[B]], %[[A]], %[[C]] lhs_transposed rhs_transposed : vector<8x4xbf16>, vector<4x8xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_mk_kn_nm(%A: vector<4x8xbf16>, %B: vector<8x4xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_mk_nk_nm(
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul %[[B]], %[[A]], %[[C]] rhs_transposed : vector<4x8xbf16>, vector<4x8xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_mk_nk_nm(%A: vector<4x8xbf16>, %B: vector<4x8xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xbf16>, vector<4x8xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_km_kn_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul %[[B]], %[[A]], %[[C]] lhs_transposed : vector<8x4xbf16>, vector<8x4xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_km_kn_nm(%A: vector<8x4xbf16>, %B: vector<8x4xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x4xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bf16_4x8x4_km_nk_nm(
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<4x4xf32>)
// CHECK: %[[R:.*]] = aievec.matmul %[[B]], %[[A]], %[[C]] : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
// CHECK: return %[[R]]
func.func @bf16_4x8x4_km_nk_nm(%A: vector<8x4xbf16>, %B: vector<4x8xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x4xbf16>, vector<4x8xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// An acc stored [n, m] of a recipe with no 256-bit transpose works when the
// swap lands on the layouts the MAC reads.
// CHECK-LABEL: func.func @i16_4x4x4_km_nk_nm(
// CHECK-SAME: %[[A:.*]]: vector<4x4xi16>, %[[B:.*]]: vector<4x4xi16>, %[[C:.*]]: vector<4x4xi64>)
// CHECK: %[[R:.*]] = aievec.matmul %[[B]], %[[A]], %[[C]] : vector<4x4xi16>, vector<4x4xi16> into vector<4x4xi64>
// CHECK: return %[[R]]
func.func @i16_4x4x4_km_nk_nm(%A: vector<4x4xi16>, %B: vector<4x4xi16>, %C: vector<4x4xi64>) -> vector<4x4xi64> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x4xi16>, vector<4x4xi16> into vector<4x4xi64>
  return %0 : vector<4x4xi64>
}
// -----

// A tile in a 6-D contraction with rhs stored [n, k] behind unit dims, as
// packed matmuls produce: the rhs is taken as it is stored.
// CHECK-LABEL: func.func @packed_rhs_nk(
// CHECK-SAME: %[[A:.*]]: vector<1x1x4x8xbf16>, %[[B:.*]]: vector<1x1x4x8xbf16>, %[[C:.*]]: vector<1x1x4x4xf32>)
// CHECK-DAG: %[[ACC:.*]] = vector.shape_cast %[[C]] : vector<1x1x4x4xf32> to vector<4x4xf32>
// CHECK-DAG: %[[L:.*]] = vector.shape_cast %[[A]] : vector<1x1x4x8xbf16> to vector<4x8xbf16>
// CHECK-DAG: %[[R:.*]] = vector.shape_cast %[[B]] : vector<1x1x4x8xbf16> to vector<4x8xbf16>
// CHECK: %[[MM:.*]] = aievec.matmul %[[L]], %[[R]], %[[ACC]] rhs_transposed : vector<4x8xbf16>, vector<4x8xbf16> into vector<4x4xf32>
// CHECK: vector.shape_cast %[[MM]] : vector<4x4xf32> to vector<1x1x4x4xf32>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
func.func @packed_rhs_nk(%A : vector<1x1x4x8xbf16>, %B : vector<1x1x4x8xbf16>,
                         %C : vector<1x1x4x4xf32>) -> vector<1x1x4x4xf32> {
  %res = vector.contract {indexing_maps = [#map1, #map2, #map3],
                          iterator_types = ["parallel", "parallel", "reduction",
                                            "parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>} %A, %B, %C :
              vector<1x1x4x8xbf16>, vector<1x1x4x8xbf16> into vector<1x1x4x4xf32>
  return %res : vector<1x1x4x4xf32>
}

// -----

// The same with operands widened to f32.
// CHECK-LABEL: func.func @packed_rhs_nk_extf(
// CHECK-SAME: %[[A:.*]]: vector<1x1x4x8xbf16>, %[[B:.*]]: vector<1x1x4x8xbf16>, %[[C:.*]]: vector<1x1x4x4xf32>)
// CHECK-DAG: %[[L:.*]] = vector.shape_cast %[[A]] : vector<1x1x4x8xbf16> to vector<4x8xbf16>
// CHECK-DAG: %[[R:.*]] = vector.shape_cast %[[B]] : vector<1x1x4x8xbf16> to vector<4x8xbf16>
// CHECK: aievec.matmul %[[L]], %[[R]], %{{.*}} rhs_transposed : vector<4x8xbf16>, vector<4x8xbf16> into vector<4x4xf32>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
func.func @packed_rhs_nk_extf(%A : vector<1x1x4x8xbf16>, %B : vector<1x1x4x8xbf16>,
                              %C : vector<1x1x4x4xf32>) -> vector<1x1x4x4xf32> {
  %lhs = arith.extf %A : vector<1x1x4x8xbf16> to vector<1x1x4x8xf32>
  %rhs = arith.extf %B : vector<1x1x4x8xbf16> to vector<1x1x4x8xf32>
  %res = vector.contract {indexing_maps = [#map1, #map2, #map3],
                          iterator_types = ["parallel", "parallel", "reduction",
                                            "parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>} %lhs, %rhs, %C :
              vector<1x1x4x8xf32>, vector<1x1x4x8xf32> into vector<1x1x4x4xf32>
  return %res : vector<1x1x4x4xf32>
}

// -----

// A unit k-block dim leading every operand, with rhs stored [n, k].
// CHECK-LABEL: func.func @unit_kblock_rhs_nk(
// CHECK: aievec.matmul %{{.*}}, %{{.*}}, %{{.*}} rhs_transposed : vector<4x8xbf16>, vector<4x8xbf16> into vector<4x4xf32>
#a = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#bt = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#c = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
func.func @unit_kblock_rhs_nk(%A: vector<1x4x8xbf16>, %B: vector<1x4x8xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [#a, #bt, #c], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<1x4x8xbf16>, vector<1x4x8xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// An extui operand keeps its signedness through a transposed layout.
// CHECK-LABEL: func.func @i8_extui_rhs_nk(
// CHECK-SAME: %[[A:.*]]: vector<4x8xi8>, %[[B:.*]]: vector<8x8xi8>, %[[C:.*]]: vector<4x8xi32>)
// CHECK-DAG: %[[L:.*]] = builtin.unrealized_conversion_cast %[[A]] : vector<4x8xi8> to vector<4x8xsi8>
// CHECK-DAG: %[[R:.*]] = builtin.unrealized_conversion_cast %[[B]] : vector<8x8xi8> to vector<8x8xui8>
// CHECK: aievec.matmul %[[L]], %[[R]], %[[C]] rhs_transposed : vector<4x8xsi8>, vector<8x8xui8> into vector<4x8xi32>
func.func @i8_extui_rhs_nk(%A: vector<4x8xi8>, %B: vector<8x8xi8>, %C: vector<4x8xi32>) -> vector<4x8xi32> {
  %a = arith.extsi %A : vector<4x8xi8> to vector<4x8xi32>
  %b = arith.extui %B : vector<8x8xi8> to vector<8x8xi32>
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %a, %b, %C : vector<4x8xi32>, vector<8x8xi32> into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}
