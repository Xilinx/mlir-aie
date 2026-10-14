//===- test-tanh-aie2p.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s

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

// CHECK-LABEL: func @test_tanh_v16f32_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xf32>
func.func @test_tanh_v16f32_aie2p(%a: vector<16xf32>) -> vector<16xf32> {
    // CHECK: %[[TANH:.*]] = aievec.tanh %[[A]] : vector<16xf32>
    %0 = math.tanh %a : vector<16xf32>
    // CHECK: return %[[TANH]] : vector<16xf32>
    return %0 : vector<16xf32>
}

// CHECK-LABEL: func @test_tanh_v32f32_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<32xf32>
func.func @test_tanh_v32f32_aie2p(%a: vector<32xf32>) -> vector<32xf32> {
    // CHECK: %[[TANH:.*]] = aievec.tanh %[[A]] : vector<32xf32>
    %0 = math.tanh %a : vector<32xf32>
    // CHECK: return %[[TANH]] : vector<32xf32>
    return %0 : vector<32xf32>
}

// CHECK-LABEL: func @test_tanh_f32_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: f32
func.func @test_tanh_f32_aie2p(%a: f32) -> f32 {
    // CHECK: %[[SPLAT:.*]] = aievec.broadcast_scalar %[[A]] : f32, vector<16xf32>
    // CHECK: %[[TANH:.*]] = aievec.tanh %[[SPLAT]] : vector<16xf32>
    // CHECK: %[[RES:.*]] = vector.extract %[[TANH]][0] : f32 from vector<16xf32>
    %0 = math.tanh %a : f32
    // CHECK: return %[[RES]] : f32
    return %0 : f32
}

// CHECK-LABEL: func @test_tanh_bf16_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: bf16
func.func @test_tanh_bf16_aie2p(%a: bf16) -> bf16 {
    // CHECK: %[[SPLAT:.*]] = aievec.broadcast_scalar %[[A]] : bf16, vector<16xbf16>
    // CHECK: %[[TANH:.*]] = aievec.tanh %[[SPLAT]] : vector<16xbf16>
    // CHECK: %[[RES:.*]] = vector.extract %[[TANH]][0] : bf16 from vector<16xbf16>
    %0 = math.tanh %a : bf16
    // CHECK: return %[[RES]] : bf16
    return %0 : bf16
}

// CHECK-LABEL: func @test_tanh_v1f32_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<1xf32>
func.func @test_tanh_v1f32_aie2p(%a: vector<1xf32>) -> vector<1xf32> {
    // CHECK: %[[ELT:.*]] = vector.extract %[[A]][0] : f32 from vector<1xf32>
    // CHECK: %[[SPLAT:.*]] = aievec.broadcast_scalar %[[ELT]] : f32, vector<16xf32>
    // CHECK: %[[TANH:.*]] = aievec.tanh %[[SPLAT]] : vector<16xf32>
    // CHECK: %[[LANE0:.*]] = vector.extract %[[TANH]][0] : f32 from vector<16xf32>
    // CHECK: %[[RES:.*]] = vector.broadcast %[[LANE0]] : f32 to vector<1xf32>
    %0 = math.tanh %a : vector<1xf32>
    // CHECK: return %[[RES]] : vector<1xf32>
    return %0 : vector<1xf32>
}

// Lane counts and element types without a lowering are left as they are.
// CHECK-LABEL: func @test_tanh_unsupported_aie2p
func.func @test_tanh_unsupported_aie2p(%a: vector<8xf32>, %b: f16,
                                       %c: tensor<16xf32>, %d: vector<[16]xf32>,
                                       %e: vector<[1]xf32>)
    -> (vector<8xf32>, f16, tensor<16xf32>, vector<[16]xf32>, vector<[1]xf32>) {
    // CHECK: math.tanh %{{.*}} : vector<8xf32>
    %0 = math.tanh %a : vector<8xf32>
    // CHECK: math.tanh %{{.*}} : f16
    %1 = math.tanh %b : f16
    // CHECK: math.tanh %{{.*}} : tensor<16xf32>
    %2 = math.tanh %c : tensor<16xf32>
    // CHECK: math.tanh %{{.*}} : vector<[16]xf32>
    %3 = math.tanh %d : vector<[16]xf32>
    // CHECK: math.tanh %{{.*}} : vector<[1]xf32>
    %4 = math.tanh %e : vector<[1]xf32>
    return %0, %1, %2, %3, %4
        : vector<8xf32>, f16, tensor<16xf32>, vector<[16]xf32>, vector<[1]xf32>
}
