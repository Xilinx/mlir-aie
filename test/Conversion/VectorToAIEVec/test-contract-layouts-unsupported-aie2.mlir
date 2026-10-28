//===- test-contract-layouts-unsupported-aie2.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie2" -verify-diagnostics

// Contractions the AIE2 matmul can not take in their stored layout are
// rejected rather than multiplied as if they were [m, k] x [k, n].

// A square tile with lhs stored [k, m]: 4x4 i16 has no 256-bit transpose.
func.func @square_lhs_km(%A: vector<4x4xi16>, %B: vector<4x4xi16>, %C: vector<4x4xi64>) -> vector<4x4xi64> {
  // expected-error @+2 {{cannot lower to an AIE2 matrix multiply: the AIE2 4x4x4 multiply reads lhs as [m, k], and no shuffle transposes vector<4x4xi16>}}
  // expected-error @+1 {{failed to legalize operation 'vector.contract'}}
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x4xi16>, vector<4x4xi16> into vector<4x4xi64>
  return %0 : vector<4x4xi64>
}

// -----

// The same square tile with rhs stored [n, k].
func.func @square_rhs_nk(%A: vector<4x4xi16>, %B: vector<4x4xi16>, %C: vector<4x4xi64>) -> vector<4x4xi64> {
  // expected-error @+2 {{cannot lower to an AIE2 matrix multiply: the AIE2 4x4x4 multiply reads rhs as [k, n], and no shuffle transposes vector<4x4xi16>}}
  // expected-error @+1 {{failed to legalize operation 'vector.contract'}}
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x4xi16>, vector<4x4xi16> into vector<4x4xi64>
  return %0 : vector<4x4xi64>
}

// -----

// The same square tile with acc stored [n, m]: as rhs^T * lhs^T, both
// operands need a 256-bit transpose.
func.func @square_acc_nm(%A: vector<4x4xi16>, %B: vector<4x4xi16>, %C: vector<4x4xi64>) -> vector<4x4xi64> {
  // expected-error @+2 {{cannot lower to an AIE2 matrix multiply: the AIE2 4x4x4 multiply reads lhs as [m, k], and no shuffle transposes vector<4x4xi16>}}
  // expected-error @+1 {{failed to legalize operation 'vector.contract'}}
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x4xi16>, vector<4x4xi16> into vector<4x4xi64>
  return %0 : vector<4x4xi64>
}

// -----

// lhs stored [k, m] of 4x8 i8 is 256 bits.
func.func @i8_lhs_km(%A: vector<8x4xi8>, %B: vector<8x8xi8>, %C: vector<4x8xi32>) -> vector<4x8xi32> {
  // expected-error @+2 {{cannot lower to an AIE2 matrix multiply: the AIE2 4x8x8 multiply reads lhs as [m, k], and no shuffle transposes vector<8x4xi8>}}
  // expected-error @+1 {{failed to legalize operation 'vector.contract'}}
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x4xi8>, vector<8x8xi8> into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

// An acc stored [n, m] whose swapped tile, 8x8x4 i8, no MAC takes.
func.func @i8_acc_nm(%A: vector<4x8xi8>, %B: vector<8x8xi8>, %C: vector<8x4xi32>) -> vector<8x4xi32> {
  // expected-error @+2 {{cannot lower to an AIE2 matrix multiply: no AIE2 matrix multiply takes 8x8 i8 times 8x4 i8 into i32}}
  // expected-error @+1 {{failed to legalize operation 'vector.contract'}}
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d0)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<4x8xi8>, vector<8x8xi8> into vector<8x4xi32>
  return %0 : vector<8x4xi32>
}

// -----

// A batch dim larger than 1.
func.func @batch(%A: vector<2x4x8xbf16>, %B: vector<2x8x4xbf16>, %C: vector<2x4x4xf32>) -> vector<2x4x4xf32> {
  // expected-error @+2 {{cannot lower to an AIE2 matrix multiply: d0 of size 2 is not the only m, n or k dim larger than 1}}
  // expected-error @+1 {{failed to legalize operation 'vector.contract'}}
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<2x4x8xbf16>, vector<2x8x4xbf16> into vector<2x4x4xf32>
  return %0 : vector<2x4x4xf32>
}

// -----

func.func @kind_mul(%A: vector<4x8xbf16>, %B: vector<8x4xbf16>, %C: vector<4x4xf32>) -> vector<4x4xf32> {
  // expected-error @+2 {{cannot lower to an AIE2 matrix multiply: the combining kind is not add}}
  // expected-error @+1 {{failed to legalize operation 'vector.contract'}}
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<mul>} %A, %B, %C : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}
