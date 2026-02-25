//===- test-inv-aie2p.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p target-backend=llvmir" | FileCheck %s

// Test: arith.divf 1.0/x → aievec.inv for scalar f32 input (AIE2P LLVMIR backend)

// CHECK-LABEL: func @test_inv_f32_aie2p
// CHECK-SAME: %[[ARG:.*]]: f32
func.func @test_inv_f32_aie2p(%arg0: f32) -> f32 {
    // CHECK: %[[INV:.*]] = aievec.inv %[[ARG]] : f32
    %cst = arith.constant 1.0 : f32
    %0 = arith.divf %cst, %arg0 : f32
    // CHECK: return %[[INV]] : f32
    return %0 : f32
}

// CHECK-LABEL: func @test_inv_with_truncf_aie2p
// CHECK-SAME: %[[ARG:.*]]: f32
func.func @test_inv_with_truncf_aie2p(%arg0: f32) -> bf16 {
    // CHECK: %[[INV:.*]] = aievec.inv %[[ARG]] : f32
    %cst = arith.constant 1.0 : f32
    %0 = arith.divf %cst, %arg0 : f32
    // CHECK: %[[TRUNC:.*]] = arith.truncf %[[INV]] : f32 to bf16
    %1 = arith.truncf %0 : f32 to bf16
    // CHECK: return %[[TRUNC]] : bf16
    return %1 : bf16
}

// CHECK-LABEL: func @test_no_inv_non_one_constant
// CHECK-SAME: %[[ARG:.*]]: f32
func.func @test_no_inv_non_one_constant(%arg0: f32) -> f32 {
    // This should NOT be converted to aievec.inv since constant is not 1.0
    // CHECK: %[[CST:.*]] = arith.constant 2.000000e+00 : f32
    %cst = arith.constant 2.0 : f32
    // CHECK: %[[DIV:.*]] = arith.divf %[[CST]], %[[ARG]] : f32
    %0 = arith.divf %cst, %arg0 : f32
    // CHECK: return %[[DIV]] : f32
    return %0 : f32
}

// CHECK-LABEL: func @test_no_inv_general_divf
// CHECK-SAME: %[[A:.*]]: f32, %[[B:.*]]: f32
func.func @test_no_inv_general_divf(%a: f32, %b: f32) -> f32 {
    // This should NOT be converted to aievec.inv since LHS is not a constant
    // CHECK: %[[DIV:.*]] = arith.divf %[[A]], %[[B]] : f32
    %0 = arith.divf %a, %b : f32
    // CHECK: return %[[DIV]] : f32
    return %0 : f32
}

// -----
// Vector tests for arith.divf dense<1.0>/x → aievec.inv

// CHECK-LABEL: func @test_inv_v16f32_aie2p
// CHECK-SAME: %[[ARG:.*]]: vector<16xf32>
func.func @test_inv_v16f32_aie2p(%arg0: vector<16xf32>) -> vector<16xf32> {
    // CHECK: %[[INV:.*]] = aievec.inv %[[ARG]] : vector<16xf32>
    %cst = arith.constant dense<1.0> : vector<16xf32>
    %0 = arith.divf %cst, %arg0 : vector<16xf32>
    // CHECK: return %[[INV]] : vector<16xf32>
    return %0 : vector<16xf32>
}

// CHECK-LABEL: func @test_inv_v32f32_aie2p
// CHECK-SAME: %[[ARG:.*]]: vector<32xf32>
func.func @test_inv_v32f32_aie2p(%arg0: vector<32xf32>) -> vector<32xf32> {
    // CHECK: %[[INV:.*]] = aievec.inv %[[ARG]] : vector<32xf32>
    %cst = arith.constant dense<1.0> : vector<32xf32>
    %0 = arith.divf %cst, %arg0 : vector<32xf32>
    // CHECK: return %[[INV]] : vector<32xf32>
    return %0 : vector<32xf32>
}

// CHECK-LABEL: func @test_no_inv_vector_non_one_constant
// CHECK-SAME: %[[ARG:.*]]: vector<16xf32>
func.func @test_no_inv_vector_non_one_constant(%arg0: vector<16xf32>) -> vector<16xf32> {
    // This should NOT be converted to aievec.inv since constant is not 1.0
    // CHECK: %[[CST:.*]] = arith.constant dense<2.000000e+00> : vector<16xf32>
    %cst = arith.constant dense<2.0> : vector<16xf32>
    // CHECK: %[[DIV:.*]] = arith.divf %[[CST]], %[[ARG]] : vector<16xf32>
    %0 = arith.divf %cst, %arg0 : vector<16xf32>
    // CHECK: return %[[DIV]] : vector<16xf32>
    return %0 : vector<16xf32>
}

// CHECK-LABEL: func @test_no_inv_vector_general_divf
// CHECK-SAME: %[[A:.*]]: vector<16xf32>, %[[B:.*]]: vector<16xf32>
func.func @test_no_inv_vector_general_divf(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
    // This should NOT be converted to aievec.inv since LHS is not a constant
    // CHECK: %[[DIV:.*]] = arith.divf %[[A]], %[[B]] : vector<16xf32>
    %0 = arith.divf %a, %b : vector<16xf32>
    // CHECK: return %[[DIV]] : vector<16xf32>
    return %0 : vector<16xf32>
}
