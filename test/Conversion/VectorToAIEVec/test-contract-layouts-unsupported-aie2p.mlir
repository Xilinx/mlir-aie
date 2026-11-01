//===- test-contract-layouts-unsupported-aie2p.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie2p" -verify-diagnostics

// Contractions that are not one AIE2P matmul tile are rejected. Every stored
// layout of a tile the AIE2P matmul takes can be brought into the layout its
// multiply reads, see test-contract-layouts-aie2p.mlir.

// A batch dim larger than 1.
func.func @batch(%A: vector<2x8x8xbf16>, %B: vector<2x8x8xbf16>, %C: vector<2x8x8xf32>) -> vector<2x8x8xf32> {
  // expected-error @+2 {{cannot lower to an AIE2P matrix multiply: d0 of size 2 is not the only m, n or k dim larger than 1}}
  // expected-error @+1 {{failed to legalize operation 'vector.contract'}}
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<2x8x8xbf16>, vector<2x8x8xbf16> into vector<2x8x8xf32>
  return %0 : vector<2x8x8xf32>
}

// -----

// Two reduction dims larger than 1.
func.func @two_k_dims(%A: vector<2x8x8xbf16>, %B: vector<2x8x8xbf16>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  // expected-error @+2 {{cannot lower to an AIE2P matrix multiply: d3 of size 8 is not the only m, n or k dim larger than 1}}
  // expected-error @+1 {{failed to legalize operation 'vector.contract'}}
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<2x8x8xbf16>, vector<2x8x8xbf16> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// A tile no multiply takes.
func.func @tile_16x8x8(%A: vector<16x8xbf16>, %B: vector<8x8xbf16>, %C: vector<16x8xf32>) -> vector<16x8xf32> {
  // expected-error @+2 {{cannot lower to an AIE2P matrix multiply: no AIE2P matrix multiply takes 16x8 bf16 times 8x8 bf16 into f32}}
  // expected-error @+1 {{failed to legalize operation 'vector.contract'}}
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<16x8xbf16>, vector<8x8xbf16> into vector<16x8xf32>
  return %0 : vector<16x8xf32>
}

// -----

// f32 operands that are not widened from bf16.
func.func @f32_operands(%A: vector<8x8xf32>, %B: vector<8x8xf32>, %C: vector<8x8xf32>) -> vector<8x8xf32> {
  // expected-error @+2 {{cannot lower to an AIE2P matrix multiply: no AIE2P matrix multiply takes 8x8 f32 times 8x8 f32 into f32}}
  // expected-error @+1 {{failed to legalize operation 'vector.contract'}}
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}
