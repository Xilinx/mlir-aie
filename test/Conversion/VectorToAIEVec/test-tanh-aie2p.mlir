//===- test-tanh-aie2p.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p target-backend=llvmir" | FileCheck %s

// CHECK-LABEL: func @test_tanh_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xbf16>
func.func @test_tanh_aie2p(%a: vector<16xbf16>) -> vector<16xbf16> {
    // CHECK: %[[TANH:.*]] = aievec.tanh %[[A]] : vector<16xbf16>
    %0 = math.tanh %a : vector<16xbf16>
    // CHECK: return %[[TANH]] : vector<16xbf16>
    return %0 : vector<16xbf16>
}

// CHECK-LABEL: func @test_tanh_v32bf16_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<32xbf16>
func.func @test_tanh_v32bf16_aie2p(%a: vector<32xbf16>) -> vector<32xbf16> {
    // CHECK: %[[TANH:.*]] = aievec.tanh %[[A]] : vector<32xbf16>
    %0 = math.tanh %a : vector<32xbf16>
    // CHECK: return %[[TANH]] : vector<32xbf16>
    return %0 : vector<32xbf16>
}
