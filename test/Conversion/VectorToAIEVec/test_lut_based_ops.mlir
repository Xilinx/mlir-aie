//===- test_lut_based_ops.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2 target-backend=llvmir" | FileCheck %s

// CHECK-DAG: func.func private @getRsqrtBf16_wrapper(vector<16xbf16>) -> vector<16xbf16>
// CHECK-DAG: func.func private @getExpBf16_wrapper(vector<16xbf16>) -> vector<8xi64>
// CHECK-LABEL: func.func @test_exp_lut
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xbf16>
func.func @test_exp_lut(%a: vector<16xbf16>) -> vector<16xbf16> {
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[CALL:.*]] = call @getExpBf16_wrapper(%[[A]]) : (vector<16xbf16>) -> vector<8xi64>
    // CHECK: %[[CAST:.*]] = vector.bitcast %[[CALL]] : vector<8xi64> to vector<16xf32>
    // CHECK: %[[SRS:.*]] = aievec.srs %[[CAST]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
    %0 = math.exp %a : vector<16xbf16>
    // CHECK: return %[[SRS]] : vector<16xbf16>
    return %0 : vector<16xbf16>
}

// CHECK-LABEL: func.func @test_rsqrt
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xbf16>
func.func @test_rsqrt(%a: vector<16xbf16>) -> vector<16xbf16> {
    // CHECK: %[[CALL:.*]] = call @getRsqrtBf16_wrapper(%[[A]]) : (vector<16xbf16>) -> vector<16xbf16>
    %0 = math.rsqrt %a : vector<16xbf16>
    // CHECK: return %[[CALL]] : vector<16xbf16>
    return %0 : vector<16xbf16>
}
