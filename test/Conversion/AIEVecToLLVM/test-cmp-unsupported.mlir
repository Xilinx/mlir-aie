//===- test-cmp-unsupported.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2" --mlir-print-op-on-diagnostic=false -verify-diagnostics
// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" --mlir-print-op-on-diagnostic=false -verify-diagnostics

// i8 compares have no lowering.
func.func @i8_ult(%a: vector<64xi8>, %b: vector<64xi8>) -> ui64 {
  // expected-error @+1 {{failed to legalize operation 'aievec.cmp'}}
  %0 = aievec.cmp %a, %b {pred = "ult"} : vector<64xi8>, vector<64xi8>, ui64
  return %0 : ui64
}

// -----

// eq and ne have no signed or unsigned form.
func.func @i32_ueq(%a: vector<16xi32>, %b: vector<16xi32>) -> ui32 {
  // expected-error @+1 {{failed to legalize operation 'aievec.cmp'}}
  %0 = aievec.cmp %a, %b {pred = "ueq"} : vector<16xi32>, vector<16xi32>, ui32
  return %0 : ui32
}

// -----

func.func @i16_unknown(%a: vector<32xi16>, %b: vector<32xi16>) -> ui32 {
  // expected-error @+1 {{failed to legalize operation 'aievec.cmp'}}
  %0 = aievec.cmp %a, %b {pred = "lt"} : vector<32xi16>, vector<32xi16>, ui32
  return %0 : ui32
}
