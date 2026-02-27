//===- test-bf16-divf.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie2p target-backend=llvmir" | FileCheck %s

// Test: bf16 vector arith.divf converted to a * inv(b) via f32 promotion.
// a/b → ups(a) → ups(b) → inv(b_f32) → mul_elem(a_f32, inv_b) → srs → bf16
// Note: extf/truncf lowered to ups/srs, 1.0/b lowered to aievec.inv by
// existing patterns in the same pass.

// CHECK-LABEL: func @test_bf16_divf_v16
// CHECK: aievec.ups {{.*}} : vector<16xbf16>, vector<16xf32>
// CHECK: aievec.ups {{.*}} : vector<16xbf16>, vector<16xf32>
// CHECK: aievec.inv
// CHECK: aievec.mul_elem
// CHECK: aievec.srs {{.*}} : vector<16xf32>, i32, vector<16xbf16>
func.func @test_bf16_divf_v16(%a: vector<16xbf16>, %b: vector<16xbf16>) -> vector<16xbf16> {
  %0 = arith.divf %a, %b : vector<16xbf16>
  return %0 : vector<16xbf16>
}
