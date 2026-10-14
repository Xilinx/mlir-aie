//===- test-exp-aie2p.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s

// CHECK-LABEL: func @test_exp_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xbf16>
func.func @test_exp_aie2p(%a: vector<16xbf16>) -> vector<16xbf16> {
    // CHECK: %[[EXP:.*]] = aievec.exp %[[A]] : vector<16xbf16>
    %0 = math.exp %a : vector<16xbf16>
    // CHECK: return %[[EXP]] : vector<16xbf16>
    return %0 : vector<16xbf16>
}

// CHECK-LABEL: func @test_exp_v32bf16_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<32xbf16>
func.func @test_exp_v32bf16_aie2p(%a: vector<32xbf16>) -> vector<32xbf16> {
    // CHECK: %[[EXP:.*]] = aievec.exp %[[A]] : vector<32xbf16>
    %0 = math.exp %a : vector<32xbf16>
    // CHECK: return %[[EXP]] : vector<32xbf16>
    return %0 : vector<32xbf16>
}

// CHECK-LABEL: func @test_exp_v16f32_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xf32>
func.func @test_exp_v16f32_aie2p(%a: vector<16xf32>) -> vector<16xf32> {
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[BF16:.*]] = aievec.srs %[[A]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
    // CHECK: %[[EXP:.*]] = aievec.exp %[[BF16]] : vector<16xbf16>
    // CHECK: %[[RES:.*]] = aievec.ups %[[EXP]] {shift = 0 : i8} : vector<16xbf16>, vector<16xf32>
    %0 = math.exp %a : vector<16xf32>
    // CHECK: return %[[RES]] : vector<16xf32>
    return %0 : vector<16xf32>
}

// CHECK-LABEL: func @test_exp_v32f32_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<32xf32>
func.func @test_exp_v32f32_aie2p(%a: vector<32xf32>) -> vector<32xf32> {
    // CHECK: %[[BF16:.*]] = aievec.srs %[[A]], %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
    // CHECK: %[[EXP:.*]] = aievec.exp %[[BF16]] : vector<32xbf16>
    // CHECK: %[[RES:.*]] = aievec.ups %[[EXP]] {shift = 0 : i8} : vector<32xbf16>, vector<32xf32>
    %0 = math.exp %a : vector<32xf32>
    // CHECK: return %[[RES]] : vector<32xf32>
    return %0 : vector<32xf32>
}

// CHECK-LABEL: func @test_exp_v8f32_aie2p
func.func @test_exp_v8f32_aie2p(%a: vector<8xf32>) -> vector<8xf32> {
    // CHECK: math.exp %{{.*}} : vector<8xf32>
    %0 = math.exp %a : vector<8xf32>
    return %0 : vector<8xf32>
}
