//===- test-exp2-aie2p.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s --check-prefix=AIE2

// CHECK-LABEL: func @test_exp2_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xbf16>
func.func @test_exp2_aie2p(%a: vector<16xbf16>) -> vector<16xbf16> {
    // CHECK: %[[EXP2:.*]] = aievec.exp2 %[[A]] : vector<16xbf16>
    %0 = math.exp2 %a : vector<16xbf16>
    // CHECK: return %[[EXP2]] : vector<16xbf16>
    return %0 : vector<16xbf16>
}

// CHECK-LABEL: func @test_exp2_v32bf16_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<32xbf16>
func.func @test_exp2_v32bf16_aie2p(%a: vector<32xbf16>) -> vector<32xbf16> {
    // CHECK: %[[EXP2:.*]] = aievec.exp2 %[[A]] : vector<32xbf16>
    %0 = math.exp2 %a : vector<32xbf16>
    // CHECK: return %[[EXP2]] : vector<32xbf16>
    return %0 : vector<32xbf16>
}

// CHECK-LABEL: func @test_exp2_v16f32_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xf32>
func.func @test_exp2_v16f32_aie2p(%a: vector<16xf32>) -> vector<16xf32> {
    // CHECK: %[[EXP2:.*]] = aievec.exp2 %[[A]] : vector<16xf32>
    %0 = math.exp2 %a : vector<16xf32>
    // CHECK: return %[[EXP2]] : vector<16xf32>
    return %0 : vector<16xf32>
}

// CHECK-LABEL: func @test_exp2_v32f32_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<32xf32>
func.func @test_exp2_v32f32_aie2p(%a: vector<32xf32>) -> vector<32xf32> {
    // CHECK: %[[EXP2:.*]] = aievec.exp2 %[[A]] : vector<32xf32>
    %0 = math.exp2 %a : vector<32xf32>
    // CHECK: return %[[EXP2]] : vector<32xf32>
    return %0 : vector<32xf32>
}

// CHECK-LABEL: func @test_exp2_f32_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: f32
func.func @test_exp2_f32_aie2p(%a: f32) -> f32 {
    // CHECK: %[[SPLAT:.*]] = aievec.broadcast_scalar %[[A]] : f32, vector<16xf32>
    // CHECK: %[[EXP2:.*]] = aievec.exp2 %[[SPLAT]] : vector<16xf32>
    // CHECK: %[[RES:.*]] = vector.extract %[[EXP2]][0] : f32 from vector<16xf32>
    %0 = math.exp2 %a : f32
    // CHECK: return %[[RES]] : f32
    return %0 : f32
}

// CHECK-LABEL: func @test_exp2_bf16_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: bf16
func.func @test_exp2_bf16_aie2p(%a: bf16) -> bf16 {
    // CHECK: %[[SPLAT:.*]] = aievec.broadcast_scalar %[[A]] : bf16, vector<16xbf16>
    // CHECK: %[[EXP2:.*]] = aievec.exp2 %[[SPLAT]] : vector<16xbf16>
    // CHECK: %[[RES:.*]] = vector.extract %[[EXP2]][0] : bf16 from vector<16xbf16>
    %0 = math.exp2 %a : bf16
    // CHECK: return %[[RES]] : bf16
    return %0 : bf16
}

// CHECK-LABEL: func @test_exp2_v1f32_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<1xf32>
func.func @test_exp2_v1f32_aie2p(%a: vector<1xf32>) -> vector<1xf32> {
    // CHECK: %[[ELT:.*]] = vector.extract %[[A]][0] : f32 from vector<1xf32>
    // CHECK: %[[SPLAT:.*]] = aievec.broadcast_scalar %[[ELT]] : f32, vector<16xf32>
    // CHECK: %[[EXP2:.*]] = aievec.exp2 %[[SPLAT]] : vector<16xf32>
    // CHECK: %[[LANE0:.*]] = vector.extract %[[EXP2]][0] : f32 from vector<16xf32>
    // CHECK: %[[RES:.*]] = vector.broadcast %[[LANE0]] : f32 to vector<1xf32>
    %0 = math.exp2 %a : vector<1xf32>
    // CHECK: return %[[RES]] : vector<1xf32>
    return %0 : vector<1xf32>
}

// Lane counts the intrinsic does not take are brought to one it does: a
// narrower vector is repeated and its low lanes kept, a wider one is split.
// CHECK-LABEL: func @test_exp2_v8f32_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<8xf32>
func.func @test_exp2_v8f32_aie2p(%a: vector<8xf32>) -> vector<8xf32> {
    // CHECK: %[[REP:.*]] = aievec.concat %[[A]], %[[A]] : vector<8xf32>, vector<16xf32>
    // CHECK: %[[EXP2:.*]] = aievec.exp2 %[[REP]] : vector<16xf32>
    // CHECK: %[[RES:.*]] = aievec.ext %[[EXP2]] {index = 0 : i8} : vector<16xf32>, vector<8xf32>
    %0 = math.exp2 %a : vector<8xf32>
    // CHECK: return %[[RES]]
    return %0 : vector<8xf32>
}

// CHECK-LABEL: func @test_exp2_v64bf16_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<64xbf16>
func.func @test_exp2_v64bf16_aie2p(%a: vector<64xbf16>) -> vector<64xbf16> {
    // CHECK: %[[LO:.*]] = aievec.ext %[[A]] {index = 0 : i8} : vector<64xbf16>, vector<32xbf16>
    // CHECK: %[[ELO:.*]] = aievec.exp2 %[[LO]] : vector<32xbf16>
    // CHECK: %[[HI:.*]] = aievec.ext %[[A]] {index = 1 : i8} : vector<64xbf16>, vector<32xbf16>
    // CHECK: %[[EHI:.*]] = aievec.exp2 %[[HI]] : vector<32xbf16>
    // CHECK: %[[RES:.*]] = aievec.concat %[[ELO]], %[[EHI]] : vector<32xbf16>, vector<64xbf16>
    %0 = math.exp2 %a : vector<64xbf16>
    // CHECK: return %[[RES]]
    return %0 : vector<64xbf16>
}

// Element types and vector kinds without a lowering are left as they are.
// CHECK-LABEL: func @test_exp2_unsupported_aie2p
func.func @test_exp2_unsupported_aie2p(%b: f16, %c: tensor<16xf32>,
                                       %d: vector<[16]xf32>,
                                       %e: vector<[1]xf32>,
                                       %g: vector<16xf16>)
    -> (f16, tensor<16xf32>, vector<[16]xf32>, vector<[1]xf32>,
        vector<16xf16>) {
    // CHECK: math.exp2 %{{.*}} : f16
    %1 = math.exp2 %b : f16
    // CHECK: math.exp2 %{{.*}} : tensor<16xf32>
    %2 = math.exp2 %c : tensor<16xf32>
    // CHECK: math.exp2 %{{.*}} : vector<[16]xf32>
    %3 = math.exp2 %d : vector<[16]xf32>
    // CHECK: math.exp2 %{{.*}} : vector<[1]xf32>
    %4 = math.exp2 %e : vector<[1]xf32>
    // CHECK: math.exp2 %{{.*}} : vector<16xf16>
    %6 = math.exp2 %g : vector<16xf16>
    // CHECK-NOT: aievec.exp2
    return %1, %2, %3, %4, %6
        : f16, tensor<16xf32>, vector<[16]xf32>, vector<[1]xf32>,
          vector<16xf16>
}

// A rank-2 operand is flattened to the lane count the intrinsic takes.
// CHECK-LABEL: func @test_exp2_v2x16bf16_aie2p
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<2x16xbf16>
// CHECK: %[[FLAT:.*]] = vector.shape_cast %[[A]] : vector<2x16xbf16> to vector<32xbf16>
// CHECK: %[[EXP2:.*]] = aievec.exp2 %[[FLAT]] : vector<32xbf16>
// CHECK: %[[RES:.*]] = vector.shape_cast %[[EXP2]] : vector<32xbf16> to vector<2x16xbf16>
// CHECK: return %[[RES]] : vector<2x16xbf16>
func.func @test_exp2_v2x16bf16_aie2p(%a: vector<2x16xbf16>) -> vector<2x16xbf16> {
    %0 = math.exp2 %a : vector<2x16xbf16>
    return %0 : vector<2x16xbf16>
}

// AIE2 has no exp2 lowering: math.exp2 is left as it is.
// AIE2-LABEL: func @test_exp2_aie2
// AIE2-NOT: aievec.exp2
// AIE2: math.exp2 %{{.*}} : vector<16xbf16>
// AIE2-NOT: aievec.exp2
func.func @test_exp2_aie2(%a: vector<16xbf16>) -> vector<16xbf16> {
    %0 = math.exp2 %a : vector<16xbf16>
    return %0 : vector<16xbf16>
}
