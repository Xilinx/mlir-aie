//===- test-nd-elementwise-aie2p.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Elementwise ops on n-D vectors with a native lane count, flattened to rank 1
// so the patterns that lower them can match.

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s

// CHECK-LABEL: func @mul_1x1x2x8_bf16
// CHECK: vector.shape_cast %{{.*}} : vector<1x1x2x8xbf16> to vector<16xbf16>
// CHECK: vector.shape_cast %{{.*}} : vector<1x1x2x8xbf16> to vector<16xbf16>
// CHECK: aievec.mul_elem
// CHECK: vector.shape_cast %{{.*}} : vector<16xbf16> to vector<1x1x2x8xbf16>
func.func @mul_1x1x2x8_bf16(%a: vector<1x1x2x8xbf16>,
                            %b: vector<1x1x2x8xbf16>) -> vector<1x1x2x8xbf16> {
  %0 = arith.mulf %a, %b : vector<1x1x2x8xbf16>
  return %0 : vector<1x1x2x8xbf16>
}

// CHECK-LABEL: func @mul_2x8_f32
// CHECK: aievec.mul_elem
func.func @mul_2x8_f32(%a: vector<2x8xf32>,
                       %b: vector<2x8xf32>) -> vector<2x8xf32> {
  %0 = arith.mulf %a, %b : vector<2x8xf32>
  return %0 : vector<2x8xf32>
}

// The reciprocal lowering matches a dividend of 1.0, so the splat is
// rematerialised at the flat type rather than shape_cast. It consumes the
// constant, so what proves it matched is that no constant and no divide
// are left.
// CHECK-LABEL: func @inv_1x1x2x8_f32
// CHECK-NOT: arith.divf
// CHECK: vector.shape_cast %{{.*}} : vector<1x1x2x8xf32> to vector<16xf32>
// CHECK: aievec.inv
// CHECK: vector.shape_cast %{{.*}} : vector<16xf32> to vector<1x1x2x8xf32>
func.func @inv_1x1x2x8_f32(%a: vector<1x1x2x8xf32>) -> vector<1x1x2x8xf32> {
  %cst = arith.constant dense<1.000000e+00> : vector<1x1x2x8xf32>
  %0 = arith.divf %cst, %a : vector<1x1x2x8xf32>
  return %0 : vector<1x1x2x8xf32>
}

// arith.negf: its pattern does not check the rank, so flattening is what
// keeps the rank out of aievec.neg.
// CHECK-LABEL: func @neg_1x1x2x8_f32
// CHECK: vector.shape_cast %{{.*}} : vector<1x1x2x8xf32> to vector<16xf32>
// CHECK: aievec.neg {{.*}} : vector<16xf32>
// CHECK: vector.shape_cast %{{.*}} : vector<16xf32> to vector<1x1x2x8xf32>
func.func @neg_1x1x2x8_f32(%a: vector<1x1x2x8xf32>) -> vector<1x1x2x8xf32> {
  %0 = arith.negf %a : vector<1x1x2x8xf32>
  return %0 : vector<1x1x2x8xf32>
}

// A multiply feeding an add is exempted as part of an FMA, but only at
// rank 1: above that it has to be flattened along with the add.
// CHECK-LABEL: func @fma_1x1x2x8_bf16
// CHECK-NOT: arith.mulf
// CHECK: aievec
func.func @fma_1x1x2x8_bf16(%a: vector<1x1x2x8xbf16>, %b: vector<1x1x2x8xbf16>,
                            %c: vector<1x1x2x8xbf16>) -> vector<1x1x2x8xbf16> {
  %0 = arith.mulf %a, %b : vector<1x1x2x8xbf16>
  %1 = arith.addf %c, %0 : vector<1x1x2x8xbf16>
  return %1 : vector<1x1x2x8xbf16>
}

// Rank-1 at 32 lanes: unchanged by the flattening.
// CHECK-LABEL: func @neg_32_f32
// CHECK-NOT: arith.negf
// CHECK: aievec.neg {{.*}} : vector<32xf32>
func.func @neg_32_f32(%a: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.negf %a : vector<32xf32>
  return %0 : vector<32xf32>
}

// Already rank-1: untouched, and not re-matched into a chain of shape casts.
// CHECK-LABEL: func @mul_16_bf16
// CHECK-NOT: vector.shape_cast
// CHECK: aievec.mul_elem
func.func @mul_16_bf16(%a: vector<16xbf16>,
                       %b: vector<16xbf16>) -> vector<16xbf16> {
  %0 = arith.mulf %a, %b : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// A lane count that is not native stays legal, so it is never handed to a
// pattern and keeps its original shape.
// CHECK-LABEL: func @mul_1x1x8x8_f32
// CHECK-NOT: vector.shape_cast
// CHECK: arith.mulf {{.*}} : vector<1x1x8x8xf32>
func.func @mul_1x1x8x8_f32(%a: vector<1x1x8x8xf32>,
                           %b: vector<1x1x8x8xf32>) -> vector<1x1x8x8xf32> {
  %0 = arith.mulf %a, %b : vector<1x1x8x8xf32>
  return %0 : vector<1x1x8x8xf32>
}
