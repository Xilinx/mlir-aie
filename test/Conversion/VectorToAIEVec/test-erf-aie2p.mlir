//===- test-erf-aie2p.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" \
// RUN:   --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s --check-prefix=LLVM

// erf(x) ~= tanh(a * x + b * x^3), with a = 2/sqrt(pi) split into the bf16
// values 1.125 and 3.372190e-03.

// CHECK-LABEL: func @test_erf_v16bf16
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xbf16>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[AHI:.*]] = arith.constant dense<1.125000e+00> : vector<16xbf16>
// CHECK-DAG: %[[ALO:.*]] = arith.constant dense<3.372190e-03> : vector<16xbf16>
// CHECK-DAG: %[[B:.*]] = arith.constant dense<1.010740e-01> : vector<16xbf16>
// CHECK: %[[X2F:.*]] = aievec.mul_elem %[[A]], %[[A]] : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
// CHECK: %[[X2:.*]] = aievec.srs %[[X2F]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
// CHECK: %[[X3F:.*]] = aievec.mul_elem %[[X2]], %[[A]] : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
// CHECK: %[[X3:.*]] = aievec.srs %[[X3F]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
// CHECK: %[[T0:.*]] = aievec.mul_elem %[[A]], %[[AHI]] : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
// CHECK: %[[T1:.*]] = aievec.mac_elem %[[A]], %[[ALO]], %[[T0]] : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
// CHECK: %[[T2:.*]] = aievec.mac_elem %[[X3]], %[[B]], %[[T1]] : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
// CHECK: %[[TANH:.*]] = aievec.tanh %[[T2]] : vector<16xf32>
// CHECK: %[[RES:.*]] = aievec.srs %[[TANH]], %[[C0]] : vector<16xf32>, i32, vector<16xbf16>
// CHECK: return %[[RES]] : vector<16xbf16>
// LLVM-LABEL: func @test_erf_v16bf16
// LLVM: "xllvm.intr.aie2p.tanh"
// LLVM-NOT: math.erf
// LLVM-NOT: aievec.
// LLVM: return
func.func @test_erf_v16bf16(%a: vector<16xbf16>) -> vector<16xbf16> {
  %0 = math.erf %a : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// CHECK-LABEL: func @test_erf_v16f32
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xf32>
// CHECK: %[[X:.*]] = aievec.srs %[[A]], %{{.*}} : vector<16xf32>, i32, vector<16xbf16>
// CHECK: aievec.mul_elem %[[X]], %[[X]] : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
// CHECK: aievec.mac_elem
// CHECK: %[[ARG:.*]] = aievec.mac_elem
// CHECK: %[[TANH:.*]] = aievec.tanh %[[ARG]] : vector<16xf32>
// CHECK-NOT: aievec.srs
// CHECK: return %[[TANH]] : vector<16xf32>
// LLVM-LABEL: func @test_erf_v16f32
// LLVM: "xllvm.intr.aie2p.tanh"
// LLVM-NOT: math.erf
// LLVM-NOT: aievec.
// LLVM: return
func.func @test_erf_v16f32(%a: vector<16xf32>) -> vector<16xf32> {
  %0 = math.erf %a : vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @test_erf_v32bf16
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<32xbf16>
// CHECK: aievec.mul_elem %[[A]], %[[A]] : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// CHECK: aievec.mac_elem {{.*}} : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// CHECK: %[[ARG:.*]] = aievec.mac_elem {{.*}} : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// CHECK: %[[TANH:.*]] = aievec.tanh %[[ARG]] : vector<32xf32>
// CHECK: %[[RES:.*]] = aievec.srs %[[TANH]], %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
// CHECK: return %[[RES]] : vector<32xbf16>
// LLVM-LABEL: func @test_erf_v32bf16
// LLVM-COUNT-2: "xllvm.intr.aie2p.tanh"
// LLVM-NOT: math.erf
// LLVM-NOT: aievec.
// LLVM: return
func.func @test_erf_v32bf16(%a: vector<32xbf16>) -> vector<32xbf16> {
  %0 = math.erf %a : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL: func @test_erf_v32f32
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<32xf32>
// CHECK: %[[X:.*]] = aievec.srs %[[A]], %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
// CHECK: aievec.mul_elem %[[X]], %[[X]] : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// CHECK: %[[TANH:.*]] = aievec.tanh %{{.*}} : vector<32xf32>
// CHECK: return %[[TANH]] : vector<32xf32>
func.func @test_erf_v32f32(%a: vector<32xf32>) -> vector<32xf32> {
  %0 = math.erf %a : vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: func @test_erf_v2x16bf16
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<2x16xbf16>
// CHECK: %[[FLAT:.*]] = vector.shape_cast %[[A]] : vector<2x16xbf16> to vector<32xbf16>
// CHECK: aievec.mul_elem %[[FLAT]], %[[FLAT]] : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// CHECK: aievec.tanh %{{.*}} : vector<32xf32>
// CHECK: %[[RES:.*]] = aievec.srs
// CHECK: vector.shape_cast %[[RES]] : vector<32xbf16> to vector<2x16xbf16>
func.func @test_erf_v2x16bf16(%a: vector<2x16xbf16>) -> vector<2x16xbf16> {
  %0 = math.erf %a : vector<2x16xbf16>
  return %0 : vector<2x16xbf16>
}

// CHECK-LABEL: func @test_erf_f32
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: f32
// CHECK: %[[SPLAT:.*]] = aievec.broadcast_scalar %[[A]] : f32, vector<16xf32>
// CHECK: aievec.srs %[[SPLAT]], %{{.*}} : vector<16xf32>, i32, vector<16xbf16>
// CHECK: %[[TANH:.*]] = aievec.tanh %{{.*}} : vector<16xf32>
// CHECK: %[[RES:.*]] = vector.extract %[[TANH]][0] : f32 from vector<16xf32>
// CHECK: return %[[RES]] : f32
// LLVM-LABEL: func @test_erf_f32
// LLVM: "xllvm.intr.aie2p.tanh"
// LLVM-NOT: math.erf
// LLVM-NOT: aievec.
// LLVM: return
func.func @test_erf_f32(%a: f32) -> f32 {
  %0 = math.erf %a : f32
  return %0 : f32
}

// CHECK-LABEL: func @test_erf_bf16
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: bf16
// CHECK: %[[SPLAT:.*]] = aievec.broadcast_scalar %[[A]] : bf16, vector<16xbf16>
// CHECK: aievec.mul_elem %[[SPLAT]], %[[SPLAT]] : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
// CHECK: %[[TANH:.*]] = aievec.tanh %{{.*}} : vector<16xf32>
// CHECK: %[[TANHBF16:.*]] = aievec.srs %[[TANH]], %{{.*}} : vector<16xf32>, i32, vector<16xbf16>
// CHECK: %[[RES:.*]] = vector.extract %[[TANHBF16]][0] : bf16 from vector<16xbf16>
// CHECK: return %[[RES]] : bf16
func.func @test_erf_bf16(%a: bf16) -> bf16 {
  %0 = math.erf %a : bf16
  return %0 : bf16
}

// CHECK-LABEL: func @test_erf_v1f32
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<1xf32>
// CHECK: %[[ELT:.*]] = vector.extract %[[A]][0] : f32 from vector<1xf32>
// CHECK: %[[SPLAT:.*]] = aievec.broadcast_scalar %[[ELT]] : f32, vector<16xf32>
// CHECK: aievec.srs %[[SPLAT]], %{{.*}} : vector<16xf32>, i32, vector<16xbf16>
// CHECK: %[[TANH:.*]] = aievec.tanh %{{.*}} : vector<16xf32>
// CHECK: %[[LANE0:.*]] = vector.extract %[[TANH]][0] : f32 from vector<16xf32>
// CHECK: %[[RES:.*]] = vector.broadcast %[[LANE0]] : f32 to vector<1xf32>
// CHECK: return %[[RES]] : vector<1xf32>
func.func @test_erf_v1f32(%a: vector<1xf32>) -> vector<1xf32> {
  %0 = math.erf %a : vector<1xf32>
  return %0 : vector<1xf32>
}

// Lane counts and element types without a lowering are left as they are.
// CHECK-LABEL: func @test_erf_unsupported
// CHECK: math.erf %{{.*}} : vector<8xf32>
// CHECK: math.erf %{{.*}} : vector<16xf16>
// CHECK: math.erf %{{.*}} : f16
// CHECK: math.erf %{{.*}} : tensor<16xf32>
// CHECK: math.erf %{{.*}} : vector<[16]xf32>
// CHECK: math.erf %{{.*}} : vector<[1]xf32>
func.func @test_erf_unsupported(%a: vector<8xf32>, %b: vector<16xf16>, %c: f16,
                                %d: tensor<16xf32>, %e: vector<[16]xf32>,
                                %f: vector<[1]xf32>)
    -> (vector<8xf32>, vector<16xf16>, f16, tensor<16xf32>, vector<[16]xf32>,
        vector<[1]xf32>) {
  %0 = math.erf %a : vector<8xf32>
  %1 = math.erf %b : vector<16xf16>
  %2 = math.erf %c : f16
  %3 = math.erf %d : tensor<16xf32>
  %4 = math.erf %e : vector<[16]xf32>
  %5 = math.erf %f : vector<[1]xf32>
  return %0, %1, %2, %3, %4, %5 : vector<8xf32>, vector<16xf16>, f16,
      tensor<16xf32>, vector<[16]xf32>, vector<[1]xf32>
}
