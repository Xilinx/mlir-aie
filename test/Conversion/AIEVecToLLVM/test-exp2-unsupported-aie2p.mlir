//===- test-exp2-unsupported-aie2p.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" --mlir-print-op-on-diagnostic=false -verify-diagnostics

// The exp2 intrinsic takes 16 lanes; 8 lanes have no lowering.
func.func @v8f32_exp2(%arg0 : vector<8xf32>) -> vector<8xf32> {
  // expected-warning @+2 {{aievec.exp2 conversion only supports v16 and v32 of bfloat16 or float.}}
  // expected-error @+1 {{failed to legalize operation 'aievec.exp2'}}
  %0 = aievec.exp2 %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

// -----

// Neither bf16 nor f32.
func.func @v16f16_exp2(%arg0 : vector<16xf16>) -> vector<16xf16> {
  // expected-warning @+2 {{aievec.exp2 conversion only supports v16 and v32 of bfloat16 or float.}}
  // expected-error @+1 {{failed to legalize operation 'aievec.exp2'}}
  %0 = aievec.exp2 %arg0 : vector<16xf16>
  return %0 : vector<16xf16>
}
