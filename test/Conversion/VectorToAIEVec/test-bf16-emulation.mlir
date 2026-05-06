//===- test-bf16-emulation.mlir - bf16 emulation of f32 ops --------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Test the bf16-emulation option which demotes f32 vector arithmetic to bf16.

// RUN: aie-opt %s -split-input-file --canonicalize-vector-for-aievec="aie-target=aie2 target-backend=llvmir bf16-emulation=true" | FileCheck %s

// Test: basic addf demotion
// CHECK-LABEL: func @test_addf
// CHECK-SAME: (%[[A:.*]]: vector<16xf32>, %[[B:.*]]: vector<16xf32>)
// CHECK: %[[A_BF16:.*]] = arith.truncf %[[A]] : vector<16xf32> to vector<16xbf16>
// CHECK: %[[B_BF16:.*]] = arith.truncf %[[B]] : vector<16xf32> to vector<16xbf16>
// CHECK: %[[RES_BF16:.*]] = arith.addf %[[A_BF16]], %[[B_BF16]] : vector<16xbf16>
// CHECK: %[[RES:.*]] = arith.extf %[[RES_BF16]] : vector<16xbf16> to vector<16xf32>
// CHECK: return %[[RES]]
func.func @test_addf(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.addf %a, %b : vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// Test: basic mulf demotion
// CHECK-LABEL: func @test_mulf
// CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: arith.mulf {{.*}} : vector<16xbf16>
// CHECK: arith.extf {{.*}} : vector<16xbf16> to vector<16xf32>
func.func @test_mulf(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.mulf %a, %b : vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// Test: basic subf demotion
// CHECK-LABEL: func @test_subf
// CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: arith.subf {{.*}} : vector<16xbf16>
// CHECK: arith.extf {{.*}} : vector<16xbf16> to vector<16xf32>
func.func @test_subf(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.subf %a, %b : vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// Test: chain optimization - intermediate extf->truncf should be eliminated
// CHECK-LABEL: func @test_chain_optimization
// CHECK-SAME: (%[[A:.*]]: vector<16xf32>, %[[B:.*]]: vector<16xf32>, %[[C:.*]]: vector<16xf32>)
// CHECK: %[[A_BF16:.*]] = arith.truncf %[[A]] : vector<16xf32> to vector<16xbf16>
// CHECK: %[[B_BF16:.*]] = arith.truncf %[[B]] : vector<16xf32> to vector<16xbf16>
// CHECK: %[[ADD:.*]] = arith.addf %[[A_BF16]], %[[B_BF16]] : vector<16xbf16>
// No intermediate extf->truncf between add and mul:
// CHECK: %[[C_BF16:.*]] = arith.truncf %[[C]] : vector<16xf32> to vector<16xbf16>
// CHECK: %[[MUL:.*]] = arith.mulf %[[ADD]], %[[C_BF16]] : vector<16xbf16>
// CHECK: %[[RES:.*]] = arith.extf %[[MUL]] : vector<16xbf16> to vector<16xf32>
// CHECK: return %[[RES]]
func.func @test_chain_optimization(%a: vector<16xf32>, %b: vector<16xf32>, %c: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.addf %a, %b : vector<16xf32>
  %1 = arith.mulf %0, %c : vector<16xf32>
  return %1 : vector<16xf32>
}

// -----

// Test: cmpf + select demotion
// CHECK-LABEL: func @test_cmpf_select
// CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: arith.cmpf ogt, {{.*}} : vector<16xbf16>
// CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: arith.select {{.*}} : vector<16xi1>, vector<16xbf16>
// CHECK: arith.extf {{.*}} : vector<16xbf16> to vector<16xf32>
func.func @test_cmpf_select(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %cmp = arith.cmpf ogt, %a, %b : vector<16xf32>
  %sel = arith.select %cmp, %a, %b : vector<16xi1>, vector<16xf32>
  return %sel : vector<16xf32>
}

// -----

// Test: vector.fma demotion
// CHECK-LABEL: func @test_fma
// CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: vector.fma {{.*}} : vector<16xbf16>
// CHECK: arith.extf {{.*}} : vector<16xbf16> to vector<16xf32>
func.func @test_fma(%a: vector<16xf32>, %b: vector<16xf32>, %c: vector<16xf32>) -> vector<16xf32> {
  %0 = vector.fma %a, %b, %c : vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// Test: maximumf demotion
// CHECK-LABEL: func @test_maximumf
// CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: arith.maximumf {{.*}} : vector<16xbf16>
// CHECK: arith.extf {{.*}} : vector<16xbf16> to vector<16xf32>
func.func @test_maximumf(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.maximumf %a, %b : vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// Test: divf is NOT demoted (bf16 vector divf unsupported on all AIE targets)
// CHECK-LABEL: func @test_divf_not_demoted
// CHECK-NOT: arith.truncf
// CHECK: arith.divf {{.*}} : vector<16xf32>
// CHECK-NOT: arith.extf
func.func @test_divf_not_demoted(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.divf %a, %b : vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// Test: chain with divf - addf/mulf are bf16, divf stays f32
// CHECK-LABEL: func @test_chain_with_divf
// CHECK-SAME: (%[[A:.*]]: vector<16xf32>, %[[B:.*]]: vector<16xf32>, %[[C:.*]]: vector<16xf32>)
// addf is demoted to bf16:
// CHECK: arith.truncf %[[A]] : vector<16xf32> to vector<16xbf16>
// CHECK: arith.truncf %[[B]] : vector<16xf32> to vector<16xbf16>
// CHECK: arith.addf {{.*}} : vector<16xbf16>
// divf stays in f32 (with extf from the addf result):
// CHECK: arith.extf {{.*}} : vector<16xbf16> to vector<16xf32>
// CHECK: arith.divf {{.*}} : vector<16xf32>
// mulf is demoted to bf16 (with truncf from the divf result):
// CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: arith.truncf %[[C]] : vector<16xf32> to vector<16xbf16>
// CHECK: arith.mulf {{.*}} : vector<16xbf16>
// CHECK: arith.extf {{.*}} : vector<16xbf16> to vector<16xf32>
func.func @test_chain_with_divf(%a: vector<16xf32>, %b: vector<16xf32>, %c: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.addf %a, %b : vector<16xf32>
  %1 = arith.divf %0, %b : vector<16xf32>
  %2 = arith.mulf %1, %c : vector<16xf32>
  return %2 : vector<16xf32>
}

// -----

// Test: bf16 ops are NOT affected (only f32 ops are demoted)
// CHECK-LABEL: func @test_bf16_unchanged
// CHECK-NOT: arith.truncf
// CHECK-NOT: arith.extf
// CHECK: arith.addf {{.*}} : vector<16xbf16>
func.func @test_bf16_unchanged(%a: vector<16xbf16>, %b: vector<16xbf16>) -> vector<16xbf16> {
  %0 = arith.addf %a, %b : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// -----

// Test: scalar f32 ops are NOT demoted (only vector ops)
// CHECK-LABEL: func @test_scalar_unchanged
// CHECK-NOT: arith.truncf
// CHECK-NOT: arith.extf
// CHECK: arith.addf {{.*}} : f32
func.func @test_scalar_unchanged(%a: f32, %b: f32) -> f32 {
  %0 = arith.addf %a, %b : f32
  return %0 : f32
}

// -----

// Test: vector.reduction is NOT demoted (scalar bf16_to_fp unsupported on
// older Peano; keeping reductions in f32 is safe since vector inputs are
// already demoted by binary patterns)
// CHECK-LABEL: func @test_reduction_not_demoted
// CHECK-NOT: arith.truncf
// CHECK: vector.reduction <add>, %{{.*}} : vector<16xf32> into f32
// CHECK-NOT: arith.extf
func.func @test_reduction_not_demoted(%a: vector<16xf32>) -> f32 {
  %0 = vector.reduction <add>, %a : vector<16xf32> into f32
  return %0 : f32
}

// -----

// Test: vector.multi_reduction is NOT demoted
// CHECK-LABEL: func @test_multi_reduction_not_demoted
// CHECK-NOT: arith.truncf {{.*}} : vector<4x16xf32> to vector<4x16xbf16>
// CHECK: vector.multi_reduction <add>, %{{.*}}, %{{.*}} [1] : vector<4x16xf32> to vector<4xf32>
func.func @test_multi_reduction_not_demoted(%a: vector<4x16xf32>, %acc: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.multi_reduction <add>, %a, %acc [1] : vector<4x16xf32> to vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// Test: negf demotion
// CHECK-LABEL: func @test_negf
// CHECK: arith.truncf {{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: arith.negf {{.*}} : vector<16xbf16>
// CHECK: arith.extf {{.*}} : vector<16xbf16> to vector<16xf32>
func.func @test_negf(%a: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.negf %a : vector<16xf32>
  return %0 : vector<16xf32>
}
