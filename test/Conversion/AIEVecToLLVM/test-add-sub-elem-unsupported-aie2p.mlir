//===- test-add-sub-elem-unsupported-aie2p.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie2p" -split-input-file -verify-diagnostics

// f32 add_elem and sub_elem lanes that do not divide the 64-lane accumulator,
// or do not fit in it, have no lowering.

func.func @add_elem_v24f32(%lhs : vector<24xf32>, %rhs : vector<24xf32>) -> vector<24xf32> {
  // expected-warning @+2 {{aievec.add_elem conversion is not supported.}}
  // expected-error @+1 {{failed to legalize operation 'aievec.add_elem'}}
  %0 = aievec.add_elem %lhs, %rhs : vector<24xf32>
  return %0 : vector<24xf32>
}

// -----

func.func @add_elem_v128f32(%lhs : vector<128xf32>, %rhs : vector<128xf32>) -> vector<128xf32> {
  // expected-warning @+2 {{aievec.add_elem conversion is not supported.}}
  // expected-error @+1 {{failed to legalize operation 'aievec.add_elem'}}
  %0 = aievec.add_elem %lhs, %rhs : vector<128xf32>
  return %0 : vector<128xf32>
}

// -----

func.func @sub_elem_v24f32(%lhs : vector<24xf32>, %rhs : vector<24xf32>) -> vector<24xf32> {
  // expected-warning @+2 {{aievec.sub_elem conversion is not supported.}}
  // expected-error @+1 {{failed to legalize operation 'aievec.sub_elem'}}
  %0 = aievec.sub_elem %lhs, %rhs : vector<24xf32>
  return %0 : vector<24xf32>
}

// -----

func.func @sub_elem_v128f32(%lhs : vector<128xf32>, %rhs : vector<128xf32>) -> vector<128xf32> {
  // expected-warning @+2 {{aievec.sub_elem conversion is not supported.}}
  // expected-error @+1 {{failed to legalize operation 'aievec.sub_elem'}}
  %0 = aievec.sub_elem %lhs, %rhs : vector<128xf32>
  return %0 : vector<128xf32>
}
