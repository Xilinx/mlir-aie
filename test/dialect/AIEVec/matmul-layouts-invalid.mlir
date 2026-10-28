//===- matmul-layouts-invalid.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// A transposed operand must contract with the others in its stored shape.
func.func @lhs_transposed_shape(%A : vector<4x8xbf16>, %B : vector<8x4xbf16>,
                                %C : vector<4x4xf32>) -> vector<4x4xf32> {
  // expected-error @+1 {{'aievec.matmul' op lhs vector<4x8xbf16> [k, m] times rhs vector<8x4xbf16> [k, n] is not a contraction into vector<4x4xf32>}}
  %0 = aievec.matmul %A, %B, %C lhs_transposed : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// AIE2 has no shuffle for a 256-bit tile: 8x4 i8.
func.func @aie2_lhs_256_bits(%A : vector<8x4xi8>, %B : vector<8x8xi8>,
                             %C : vector<4x8xi32>) -> vector<4x8xi32> {
  // expected-error @+1 {{'aievec.matmul' op the AIE2 4x8x8 multiply reads lhs as [m, k], and no shuffle transposes vector<8x4xi8>}}
  %0 = aievec.matmul %A, %B, %C lhs_transposed : vector<8x4xi8>, vector<8x8xi8> into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

// No shuffle moves 4-bit values.
func.func @aie2_rhs_i4(%A : vector<4x16xi8>, %B : vector<8x16xi4>,
                       %C : vector<4x8xi32>) -> vector<4x8xi32> {
  // expected-error @+1 {{'aievec.matmul' op the AIE2 4x16x8 multiply reads rhs as [k, n], and no shuffle transposes vector<8x16xi4>}}
  %0 = aievec.matmul %A, %B, %C rhs_transposed : vector<4x16xi8>, vector<8x16xi4> into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

func.func @aie2p_tile(%A : vector<8x8xbf16>, %B : vector<8x16xbf16>,
                      %C : vector<8x16xf32>) -> vector<8x16xf32> {
  // expected-error @+1 {{'aievec.matmul_aie2p' op no AIE2P matrix multiply takes 8x8 bf16 times 8x16 bf16 into f32}}
  %0 = aievec.matmul_aie2p %A, %B, %C : vector<8x8xbf16>, vector<8x16xbf16> into vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// -----

func.func @aie2p_rhs_transposed_shape(%A : vector<8x8xbf16>, %B : vector<8x4xbf16>,
                                      %C : vector<8x4xf32>) -> vector<8x4xf32> {
  // expected-error @+1 {{'aievec.matmul_aie2p' op lhs vector<8x8xbf16> [m, k] times rhs vector<8x4xbf16> [n, k] is not a contraction into vector<8x4xf32>}}
  %0 = aievec.matmul_aie2p %A, %B, %C rhs_transposed : vector<8x8xbf16>, vector<8x4xbf16> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

func.func @aie2p_rank(%A : vector<1x8x8xbf16>, %B : vector<1x8x8xbf16>,
                      %C : vector<1x8x8xf32>) -> vector<1x8x8xf32> {
  // expected-error @+1 {{'aievec.matmul_aie2p' op operand #0 must be a vector compatible with a lhs operand of AIE2P matrix-multiply and accumulate, but got 'vector<1x8x8xbf16>'}}
  %0 = aievec.matmul_aie2p %A, %B, %C : vector<1x8x8xbf16>, vector<1x8x8xbf16> into vector<1x8x8xf32>
  return %0 : vector<1x8x8xf32>
}
