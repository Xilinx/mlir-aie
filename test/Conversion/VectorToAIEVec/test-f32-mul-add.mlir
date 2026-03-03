//===- test-f32-mul-add.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie2p target-backend=llvmir" | FileCheck %s

// Test: f32 mulf→addf should NOT be deferred to FMA (FMA only handles bf16).
// The addf should be independently converted to aievec.add_elem instead of
// failing to legalize.

// CHECK-LABEL: func @test_f32_mul_add
// CHECK: arith.mulf
// CHECK: aievec.add_elem
func.func @test_f32_mul_add(%a: vector<16xf32>, %b: vector<16xf32>, %c: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.mulf %a, %b : vector<16xf32>
  %1 = arith.addf %0, %c : vector<16xf32>
  return %1 : vector<16xf32>
}

// -----

// bf16 mulf→addf should still be deferred to FMA (existing behavior)
// CHECK-LABEL: func @test_bf16_mul_add_fma
// CHECK: aievec.mac_elem
func.func @test_bf16_mul_add_fma(%a: vector<16xbf16>, %b: vector<16xbf16>, %c: vector<16xbf16>) -> vector<16xbf16> {
  %0 = arith.mulf %a, %b : vector<16xbf16>
  %1 = arith.addf %0, %c : vector<16xbf16>
  return %1 : vector<16xbf16>
}
