//===- test-trunc-ext.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec | FileCheck %s
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s --check-prefix=AIE2P

// CHECK-LABEL: func.func @test_exti(
// CHECK-SAME: %[[V1:[a-zA-Z0-9]+]]: vector<32xi8>
// CHECK-SAME: %[[V2:[a-zA-Z0-9]+]]: vector<32xi8>
// CHECK-SAME: %[[V3:[a-zA-Z0-9]+]]: vector<32xi16>
// AIE2P-LABEL: func.func @test_exti(
// AIE2P-SAME: %[[V1:[a-zA-Z0-9]+]]: vector<32xi8>
// AIE2P-SAME: %[[V2:[a-zA-Z0-9]+]]: vector<32xi8>
// AIE2P-SAME: %[[V3:[a-zA-Z0-9]+]]: vector<32xi16>
func.func @test_exti(%vi8_4_i16 : vector<32xi8>, %vi8_4_i32 : vector<32xi8>,
                     %vi16_4_i32 : vector<32xi16>) ->
                        (vector<32xi16>, vector<32xi32>, vector<32xi32>) {
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // AIE2P: %[[C0:.*]] = arith.constant 0 : i32
    %0 = arith.extsi %vi8_4_i16 : vector<32xi8> to vector<32xi16>
    // CHECK: %[[V1A:.*]] = aievec.ups %[[V1]] {shift = 0 : i8} :
    // CHECK-SAME:                           vector<32xi8>, vector<32xi32>
    // CHECK: %[[V1E:.*]] = aievec.srs %[[V1A]], %[[C0]] :
    // CHECK-SAME:                           vector<32xi32>, i32, vector<32xi16>
    // AIE2P: %[[V1A:.*]] = aievec.ups %[[V1]] {shift = 0 : i8} :
    // AIE2P-SAME:                           vector<32xi8>, vector<32xi32>
    // AIE2P: %[[V1E:.*]] = aievec.srs %[[V1A]], %[[C0]] :
    // AIE2P-SAME:                           vector<32xi32>, i32, vector<32xi16>
    %1 = arith.extsi %vi8_4_i32 : vector<32xi8> to vector<32xi32>
    // CHECK: %[[V2A:.*]] = aievec.ups %[[V2]] {shift = 0 : i8} :
    // CHECK-SAME:                           vector<32xi8>, vector<32xi32>
    // CHECK: %[[V2E:.*]] = aievec.cast %[[V2A]] {isResAcc = false} :
    // CHECK-SAME:                           vector<32xi32>, vector<32xi32>
    // AIE2P: %[[V2A:.*]] = aievec.ups %[[V2]] {shift = 0 : i8} :
    // AIE2P-SAME:                           vector<32xi8>, vector<32xi32>
    // AIE2P: %[[V2E:.*]] = aievec.cast %[[V2A]] {isResAcc = false} :
    // AIE2P-SAME:                           vector<32xi32>, vector<32xi32>
    %2 = arith.extsi %vi16_4_i32 : vector<32xi16> to vector<32xi32>
    // CHECK: %[[V3A:.*]] = aievec.ups %[[V3]] {shift = 0 : i8} :
    // CHECK-SAME:                           vector<32xi16>, vector<32xi32>
    // CHECK: %[[V3E:.*]] = aievec.cast %[[V3A]] {isResAcc = false} :
    // CHECK-SAME:                           vector<32xi32>, vector<32xi32>
    // AIE2P: %[[V3A:.*]] = aievec.ups %[[V3]] {shift = 0 : i8} :
    // AIE2P-SAME:                           vector<32xi16>, vector<32xi32>
    // AIE2P: %[[V3E:.*]] = aievec.cast %[[V3A]] {isResAcc = false} :
    // AIE2P-SAME:                           vector<32xi32>, vector<32xi32>
    return %0, %1, %2 : vector<32xi16>, vector<32xi32>, vector<32xi32>
    // CHECK: return %[[V1E]], %[[V2E]], %[[V3E]]
    // AIE2P: return %[[V1E]], %[[V2E]], %[[V3E]]
}

// CHECK-LABEL: func.func @test_extf(
// CHECK-SAME: %[[V:[a-zA-Z0-9]+]]: vector<16xbf16>
// AIE2P-LABEL: func.func @test_extf(
// AIE2P-SAME: %[[V:[a-zA-Z0-9]+]]: vector<16xbf16>
func.func @test_extf(%vbf16_4_f32 : vector<16xbf16>) -> vector<16xf32> {
    %0 = arith.extf %vbf16_4_f32 : vector<16xbf16> to vector<16xf32>
    // CHECK: %[[VA:.*]] = aievec.ups %[[V]] {shift = 0 : i8} :
    // CHECK-SAME:                           vector<16xbf16>, vector<16xf32>
    // CHECK: %[[VE:.*]] = aievec.cast %[[VA]] {isResAcc = false} :
    // CHECK-SAME:                           vector<16xf32>, vector<16xf32>
    // AIE2P: %[[VA:.*]] = aievec.ups %[[V]] {shift = 0 : i8} :
    // AIE2P-SAME:                           vector<16xbf16>, vector<16xf32>
    // AIE2P: %[[VE:.*]] = aievec.cast %[[VA]] {isResAcc = false} :
    // AIE2P-SAME:                           vector<16xf32>, vector<16xf32>
    return %0 : vector<16xf32>
    // CHECK: return %[[VE]]
    // AIE2P: return %[[VE]]
}

// CHECK-LABEL: func.func @test_trunci(
// CHECK-SAME: %[[V1:[a-zA-Z0-9]+]]: vector<32xi32>
// CHECK-SAME: %[[V2:[a-zA-Z0-9]+]]: vector<32xi32>
// CHECK-SAME: %[[V3:[a-zA-Z0-9]+]]: vector<32xi16>
// AIE2P-LABEL: func.func @test_trunci(
// AIE2P-SAME: %[[V1:[a-zA-Z0-9]+]]: vector<32xi32>
// AIE2P-SAME: %[[V2:[a-zA-Z0-9]+]]: vector<32xi32>
// AIE2P-SAME: %[[V3:[a-zA-Z0-9]+]]: vector<32xi16>
func.func @test_trunci(%vi32_4_i8 : vector<32xi32>, %vi32_4_i16 : vector<32xi32>,
                       %vi16_4_i8 : vector<32xi16>) ->
                        (vector<32xi8>, vector<32xi16>, vector<32xi8>) {
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // AIE2P: %[[C0:.*]] = arith.constant 0 : i32
    %0 = arith.trunci %vi32_4_i8 : vector<32xi32> to vector<32xi8>
    // CHECK: %[[V1A:.*]] = aievec.cast %[[V1]] {isResAcc = true} :
    // CHECK-SAME:                           vector<32xi32>, vector<32xi32>
    // CHECK: %[[V1T:.*]] = aievec.srs %[[V1A]], %[[C0]] :
    // CHECK-SAME:                           vector<32xi32>, i32, vector<32xi8>
    // AIE2P: %[[V1A:.*]] = aievec.cast %[[V1]] {isResAcc = true} :
    // AIE2P-SAME:                           vector<32xi32>, vector<32xi32>
    // AIE2P: %[[V1T:.*]] = aievec.srs %[[V1A]], %[[C0]] :
    // AIE2P-SAME:                           vector<32xi32>, i32, vector<32xi8>
    %1 = arith.trunci %vi32_4_i16 : vector<32xi32> to vector<32xi16>
    // CHECK: %[[V2A:.*]] = aievec.cast %[[V2]] {isResAcc = true} :
    // CHECK-SAME:                           vector<32xi32>, vector<32xi32>
    // CHECK: %[[V2T:.*]] = aievec.srs %[[V2A]], %[[C0]] :
    // CHECK-SAME:                           vector<32xi32>, i32, vector<32xi16>
    // AIE2P: %[[V2A:.*]] = aievec.cast %[[V2]] {isResAcc = true} :
    // AIE2P-SAME:                           vector<32xi32>, vector<32xi32>
    // AIE2P: %[[V2T:.*]] = aievec.srs %[[V2A]], %[[C0]] :
    // AIE2P-SAME:                           vector<32xi32>, i32, vector<32xi16>
    %2 = arith.trunci %vi16_4_i8 : vector<32xi16> to vector<32xi8>
    // CHECK: %[[V3A:.*]] = aievec.ups %[[V3]] {shift = 0 : i8} :
    // CHECK-SAME:                           vector<32xi16>, vector<32xi32>
    // CHECK: %[[V3T:.*]] = aievec.srs %[[V3A]], %[[C0]] :
    // CHECK-SAME:                           vector<32xi32>, i32, vector<32xi8>
    // AIE2P: %[[V3A:.*]] = aievec.ups %[[V3]] {shift = 0 : i8} :
    // AIE2P-SAME:                           vector<32xi16>, vector<32xi32>
    // AIE2P: %[[V3T:.*]] = aievec.srs %[[V3A]], %[[C0]] :
    // AIE2P-SAME:                           vector<32xi32>, i32, vector<32xi8>
    return %0, %1, %2 : vector<32xi8>, vector<32xi16>, vector<32xi8>
    // CHECK: return %[[V1T]], %[[V2T]], %[[V3T]]
    // AIE2P: return %[[V1T]], %[[V2T]], %[[V3T]]
}

// CHECK-LABEL: func.func @test_truncf(
// CHECK-SAME: %[[V:[a-zA-Z0-9]+]]: vector<16xf32>
// AIE2P-LABEL: func.func @test_truncf(
// AIE2P-SAME: %[[V:[a-zA-Z0-9]+]]: vector<16xf32>
func.func @test_truncf(%vf32_4_bf16 : vector<16xf32>) -> vector<16xbf16> {
    %0 = arith.truncf %vf32_4_bf16 : vector<16xf32> to vector<16xbf16>
    // CHECK: %[[VA:.*]] = aievec.cast %[[V]] {isResAcc = true} :
    // CHECK-SAME:                           vector<16xf32>, vector<16xf32>
    // CHECK: %[[VT:.*]] = aievec.srs %[[VA]], %[[C0]] :
    // CHECK-SAME:                           vector<16xf32>, i32, vector<16xbf16>
    // AIE2P: %[[VA:.*]] = aievec.cast %[[V]] {isResAcc = true} :
    // AIE2P-SAME:                           vector<16xf32>, vector<16xf32>
    // AIE2P: %[[VT:.*]] = aievec.srs %[[VA]], %[[C0]] :
    // AIE2P-SAME:                           vector<16xf32>, i32, vector<16xbf16>
    return %0 : vector<16xbf16>
    // CHECK: return %[[VE]]
    // AIE2P: return %[[VE]]
}
