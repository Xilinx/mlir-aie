//===- test-bf16-emulation.mlir - bf16 emulation of f32 ops --------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test the bf16-emulation option, which runs f32 vector arithmetic on the bf16
// datapath: multiplies and transcendentals take bf16 inputs, sums stay f32.

// RUN: aie-opt %s -split-input-file --canonicalize-vector-for-aievec="aie-target=aie2 bf16-emulation" | FileCheck %s

// Test: addf stays f32
// CHECK-LABEL: func @test_addf
// CHECK-NOT: arith.truncf
// CHECK: arith.addf {{.*}} : vector<16xf32>
func.func @test_addf(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.addf %a, %b : vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// Test: mulf rounds its inputs to bf16; the product stays f32
// CHECK-LABEL: func @test_mulf
// CHECK-SAME: (%[[A:.*]]: vector<16xf32>, %[[B:.*]]: vector<16xf32>)
// CHECK: %[[A_BF16:.*]] = arith.truncf %[[A]] : vector<16xf32> to vector<16xbf16>
// CHECK: %[[A_R:.*]] = arith.extf %[[A_BF16]] : vector<16xbf16> to vector<16xf32>
// CHECK: %[[B_BF16:.*]] = arith.truncf %[[B]] : vector<16xf32> to vector<16xbf16>
// CHECK: %[[B_R:.*]] = arith.extf %[[B_BF16]] : vector<16xbf16> to vector<16xf32>
// CHECK: %[[RES:.*]] = arith.mulf %[[A_R]], %[[B_R]] : vector<16xf32>
// CHECK: return %[[RES]]
func.func @test_mulf(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.mulf %a, %b : vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// Test: subf stays f32
// CHECK-LABEL: func @test_subf
// CHECK-NOT: arith.truncf
// CHECK: arith.subf {{.*}} : vector<16xf32>
func.func @test_subf(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.subf %a, %b : vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// Test: a sum feeding a multiply is rounded once, as a multiply input
// CHECK-LABEL: func @test_chain_optimization
// CHECK-SAME: (%[[A:.*]]: vector<16xf32>, %[[B:.*]]: vector<16xf32>, %[[C:.*]]: vector<16xf32>)
// CHECK: %[[ADD:.*]] = arith.addf %[[A]], %[[B]] : vector<16xf32>
// CHECK: %[[ADD_BF16:.*]] = arith.truncf %[[ADD]] : vector<16xf32> to vector<16xbf16>
// CHECK: %[[ADD_R:.*]] = arith.extf %[[ADD_BF16]]
// CHECK: %[[C_BF16:.*]] = arith.truncf %[[C]] : vector<16xf32> to vector<16xbf16>
// CHECK: %[[C_R:.*]] = arith.extf %[[C_BF16]]
// CHECK: %[[MUL:.*]] = arith.mulf %[[ADD_R]], %[[C_R]] : vector<16xf32>
// CHECK: return %[[MUL]]
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

// Test: vector.fma rounds its multiplicands; the accumulator stays f32
// CHECK-LABEL: func @test_fma
// CHECK-SAME: (%[[A:.*]]: vector<16xf32>, %[[B:.*]]: vector<16xf32>, %[[C:.*]]: vector<16xf32>)
// CHECK: %[[A_BF16:.*]] = arith.truncf %[[A]] : vector<16xf32> to vector<16xbf16>
// CHECK: %[[A_R:.*]] = arith.extf %[[A_BF16]]
// CHECK: %[[B_BF16:.*]] = arith.truncf %[[B]] : vector<16xf32> to vector<16xbf16>
// CHECK: %[[B_R:.*]] = arith.extf %[[B_BF16]]
// CHECK: %[[RES:.*]] = vector.fma %[[A_R]], %[[B_R]], %[[C]] : vector<16xf32>
// CHECK: return %[[RES]]
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

// Test: chain with divf - addf and divf stay f32, mulf takes bf16 inputs
// CHECK-LABEL: func @test_chain_with_divf
// CHECK-SAME: (%[[A:.*]]: vector<16xf32>, %[[B:.*]]: vector<16xf32>, %[[C:.*]]: vector<16xf32>)
// CHECK: %[[ADD:.*]] = arith.addf %[[A]], %[[B]] : vector<16xf32>
// CHECK: %[[DIV:.*]] = arith.divf %[[ADD]], %[[B]] : vector<16xf32>
// CHECK: arith.truncf %[[DIV]] : vector<16xf32> to vector<16xbf16>
// CHECK: arith.truncf %[[C]] : vector<16xf32> to vector<16xbf16>
// CHECK: arith.mulf {{.*}} : vector<16xf32>
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
// older Peano)
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

// -----

// Test: a round trip through some other pair of float types is left alone
// CHECK-LABEL: func @test_unrelated_round_trip_unchanged
// CHECK: %[[EXT:.*]] = arith.extf %{{.*}} : vector<16xf16> to vector<16xf64>
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[EXT]] : vector<16xf64> to vector<16xf16>
// CHECK: return %[[TRUNC]]
func.func @test_unrelated_round_trip_unchanged(%a: vector<16xf16>) -> vector<16xf16> {
  %0 = arith.extf %a : vector<16xf16> to vector<16xf64>
  %1 = arith.truncf %0 : vector<16xf64> to vector<16xf16>
  return %1 : vector<16xf16>
}

// -----

// Test: a multiply whose only user is an add becomes an fma when both allow
// contraction
// CHECK-LABEL: func @test_mul_add_to_fma
// CHECK-SAME: (%[[A:.*]]: vector<32xf32>, %[[B:.*]]: vector<32xf32>, %[[C:.*]]: vector<32xf32>)
// CHECK: %[[A_BF16:.*]] = arith.truncf %[[A]] : vector<32xf32> to vector<32xbf16>
// CHECK: %[[A_R:.*]] = arith.extf %[[A_BF16]]
// CHECK: %[[B_BF16:.*]] = arith.truncf %[[B]] : vector<32xf32> to vector<32xbf16>
// CHECK: %[[B_R:.*]] = arith.extf %[[B_BF16]]
// CHECK: %[[RES:.*]] = vector.fma %[[A_R]], %[[B_R]], %[[C]] : vector<32xf32>
// CHECK-NOT: arith.mulf
// CHECK: return %[[RES]]
func.func @test_mul_add_to_fma(%a: vector<32xf32>, %b: vector<32xf32>, %c: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %1 = arith.addf %c, %0 fastmath<contract> : vector<32xf32>
  return %1 : vector<32xf32>
}

// -----

// Test: a multiply with another user is not folded into the add
// CHECK-LABEL: func @test_mul_add_multi_use
// CHECK: %[[MUL:.*]] = arith.mulf {{.*}} : vector<16xf32>
// CHECK: arith.addf %[[MUL]], {{.*}} : vector<16xf32>
// CHECK-NOT: vector.fma
func.func @test_mul_add_multi_use(%a: vector<16xf32>, %b: vector<16xf32>, %c: vector<16xf32>) -> (vector<16xf32>, vector<16xf32>) {
  %0 = arith.mulf %a, %b fastmath<contract> : vector<16xf32>
  %1 = arith.addf %0, %c fastmath<contract> : vector<16xf32>
  return %0, %1 : vector<16xf32>, vector<16xf32>
}

// -----

// Test: a multiply whose inputs are already bf16 is left alone
// CHECK-LABEL: func @test_mulf_bf16_inputs
// CHECK-SAME: (%[[A:.*]]: vector<16xbf16>, %[[B:.*]]: vector<16xbf16>)
// CHECK: %[[A_F32:.*]] = arith.extf %[[A]]
// CHECK: %[[B_F32:.*]] = arith.extf %[[B]]
// CHECK: arith.mulf %[[A_F32]], %[[B_F32]] : vector<16xf32>
// CHECK-NOT: arith.truncf
func.func @test_mulf_bf16_inputs(%a: vector<16xbf16>, %b: vector<16xbf16>) -> vector<16xf32> {
  %0 = arith.extf %a : vector<16xbf16> to vector<16xf32>
  %1 = arith.extf %b : vector<16xbf16> to vector<16xf32>
  %2 = arith.mulf %0, %1 : vector<16xf32>
  return %2 : vector<16xf32>
}

// -----

// Test: constants are rounded to bf16 at compile time; one that is already
// exact in bf16 is used as it is
// CHECK-LABEL: func @test_mulf_constants
// CHECK-SAME: (%[[A:.*]]: vector<16xbf16>)
// CHECK-DAG: %[[ROUNDED:.*]] = arith.constant dense<7.968750e-01> : vector<16xf32>
// CHECK-DAG: %[[HALF:.*]] = arith.constant dense<5.000000e-01> : vector<16xf32>
// CHECK: %[[A_F32:.*]] = arith.extf %[[A]]
// CHECK: %[[M:.*]] = arith.mulf %[[A_F32]], %[[ROUNDED]] : vector<16xf32>
// CHECK: arith.mulf %[[A_F32]], %[[HALF]] : vector<16xf32>
// CHECK-NOT: arith.truncf
func.func @test_mulf_constants(%a: vector<16xbf16>) -> (vector<16xf32>, vector<16xf32>) {
  %c = arith.constant dense<0.797884> : vector<16xf32>
  %h = arith.constant dense<0.5> : vector<16xf32>
  %0 = arith.extf %a : vector<16xbf16> to vector<16xf32>
  %1 = arith.mulf %0, %c : vector<16xf32>
  %2 = arith.mulf %0, %h : vector<16xf32>
  return %1, %2 : vector<16xf32>, vector<16xf32>
}

// -----

// Test: math.tanh runs in bf16
// CHECK-LABEL: func @test_tanh
// CHECK-SAME: (%[[A:.*]]: vector<32xf32>)
// CHECK: %[[A_BF16:.*]] = arith.truncf %[[A]] : vector<32xf32> to vector<32xbf16>
// CHECK: %[[T:.*]] = math.tanh %[[A_BF16]] : vector<32xbf16>
// CHECK: %[[RES:.*]] = arith.extf %[[T]] : vector<32xbf16> to vector<32xf32>
// CHECK: return %[[RES]]
func.func @test_tanh(%a: vector<32xf32>) -> vector<32xf32> {
  %0 = math.tanh %a : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

// Test: only the operand that is not already bf16 is rounded
// CHECK-LABEL: func @test_mulf_mixed
// CHECK-SAME: (%[[A:.*]]: vector<16xbf16>, %[[B:.*]]: vector<16xf32>)
// CHECK: %[[A_F32:.*]] = arith.extf %[[A]]
// CHECK-NOT: arith.truncf %[[A_F32]]
// CHECK: %[[B_BF16:.*]] = arith.truncf %[[B]] : vector<16xf32> to vector<16xbf16>
// CHECK: %[[B_R:.*]] = arith.extf %[[B_BF16]]
// CHECK: arith.mulf %[[A_F32]], %[[B_R]] : vector<16xf32>
func.func @test_mulf_mixed(%a: vector<16xbf16>, %b: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.extf %a : vector<16xbf16> to vector<16xf32>
  %1 = arith.mulf %0, %b : vector<16xf32>
  return %1 : vector<16xf32>
}

// -----

// Test: an fma whose multiplicands are already bf16 is left alone
// CHECK-LABEL: func @test_fma_bf16_inputs
// CHECK-NOT: arith.truncf
// CHECK: vector.fma {{.*}} : vector<16xf32>
func.func @test_fma_bf16_inputs(%a: vector<16xbf16>, %b: vector<16xbf16>, %c: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.extf %a : vector<16xbf16> to vector<16xf32>
  %1 = arith.extf %b : vector<16xbf16> to vector<16xf32>
  %2 = vector.fma %0, %1, %c : vector<16xf32>
  return %2 : vector<16xf32>
}

// -----

// Test: the multiply may be either operand of the add
// CHECK-LABEL: func @test_mul_add_to_fma_lhs
// CHECK-SAME: (%[[A:.*]]: vector<16xf32>, %[[B:.*]]: vector<16xf32>, %[[C:.*]]: vector<16xf32>)
// CHECK: vector.fma %{{.*}}, %{{.*}}, %[[C]] : vector<16xf32>
// CHECK-NOT: arith.addf
func.func @test_mul_add_to_fma_lhs(%a: vector<16xf32>, %b: vector<16xf32>, %c: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.mulf %a, %b fastmath<contract> : vector<16xf32>
  %1 = arith.addf %0, %c fastmath<contract> : vector<16xf32>
  return %1 : vector<16xf32>
}

// -----

// Test: without contraction on both ops the multiply and add stay separate;
// the multiply keeps its flags
// CHECK-LABEL: func @test_mul_add_no_contract
// CHECK: %[[M1:.*]] = arith.mulf %{{.*}}, %{{.*}} fastmath<contract> : vector<16xf32>
// CHECK: arith.addf %[[M1]], %{{.*}} : vector<16xf32>
// CHECK: %[[M2:.*]] = arith.mulf %{{.*}}, %{{.*}} : vector<16xf32>
// CHECK: arith.addf %[[M2]], %{{.*}} fastmath<contract> : vector<16xf32>
// CHECK-NOT: vector.fma
func.func @test_mul_add_no_contract(%a: vector<16xf32>, %b: vector<16xf32>, %c: vector<16xf32>) -> (vector<16xf32>, vector<16xf32>) {
  %0 = arith.mulf %a, %b fastmath<contract> : vector<16xf32>
  %1 = arith.addf %0, %c : vector<16xf32>
  %2 = arith.mulf %a, %c : vector<16xf32>
  %3 = arith.addf %2, %b fastmath<contract> : vector<16xf32>
  return %1, %3 : vector<16xf32>, vector<16xf32>
}
