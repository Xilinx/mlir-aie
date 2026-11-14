//===- bfp16-ops-invalid.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -verify-diagnostics

func.func @rows_not_8_values(%a: vector<8x4xbf16>) {
  // expected-error @+1 {{expects a tile of 8 or 4 rows of 8 values, got 8x4}}
  %m, %e = aievec.to_bfp16 %a : vector<8x4xbf16> -> vector<64xi8>, vector<8xi8>
  return
}

// -----

func.func @two_rows(%a: vector<8x2xbf16>) {
  // expected-error @+1 {{expects a tile of 8 or 4 rows of 8 values, got 2x8}}
  %m, %e = aievec.to_bfp16 %a transpose : vector<8x2xbf16> -> vector<64xi8>, vector<8xi8>
  return
}

// -----

func.func @matmul_tensor_acc(%m: vector<64xi8>, %e: vector<8xi8>, %acc: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{must be vector of shape <8x8> of 32-bit float or vector of shape <4x8> of 32-bit float}}
  %r = aievec.matmul_aie2p_bfp16 %m, %e, %m, %e, %acc : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}
