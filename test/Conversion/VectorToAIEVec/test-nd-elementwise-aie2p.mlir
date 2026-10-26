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

// An extf feeding a contraction is not flattened: the contraction lowering
// folds it into a bf16 matmul.

#mapA = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#mapB = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#mapC = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map2A = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2B = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2C = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @extf_into_contract_1x1x8x8
// CHECK-NOT: arith.extf
// CHECK: aievec.matmul_aie2p {{.*}} : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
// CHECK-NOT: vector.contract
func.func @extf_into_contract_1x1x8x8(%a: vector<1x1x8x8xbf16>,
                                      %b: vector<1x1x8x8xbf16>,
                                      %c: vector<1x1x8x8xf32>)
    -> vector<1x1x8x8xf32> {
  %0 = arith.extf %a : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
  %1 = arith.extf %b : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
  %2 = vector.contract {indexing_maps = [#mapA, #mapB, #mapC],
                        iterator_types = ["parallel", "parallel", "reduction",
                                          "parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %c
       : vector<1x1x8x8xf32>, vector<1x1x8x8xf32> into vector<1x1x8x8xf32>
  return %2 : vector<1x1x8x8xf32>
}

// CHECK-LABEL: func @extf_into_contract_8x8
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>
// CHECK-NOT: arith.extf
// CHECK: aievec.matmul_aie2p %[[A]], %[[B]], %[[C]] : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
// CHECK-NOT: vector.contract
func.func @extf_into_contract_8x8(%a: vector<8x8xbf16>, %b: vector<8x8xbf16>,
                                  %c: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = arith.extf %a : vector<8x8xbf16> to vector<8x8xf32>
  %1 = arith.extf %b : vector<8x8xbf16> to vector<8x8xf32>
  %2 = vector.contract {indexing_maps = [#map2A, #map2B, #map2C],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %c
       : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
  return %2 : vector<8x8xf32>
}

// An extf with another user besides the contraction is kept whole as well;
// that other use is lowered at the original shape.
// CHECK-LABEL: func @extf_into_contract_and_return
// CHECK-SAME: %[[A:[a-z0-9]+]]: vector<1x1x8x8xbf16>
// CHECK: %[[UPS:.*]] = aievec.ups %[[A]] {{.*}} : vector<1x1x8x8xbf16>, vector<1x1x8x8xf32>
// CHECK: %[[EXT:.*]] = aievec.cast %[[UPS]]
// CHECK: aievec.matmul_aie2p {{.*}} : vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
// CHECK: return %{{.*}}, %[[EXT]]
// CHECK-NOT: vector.contract
func.func @extf_into_contract_and_return(%a: vector<1x1x8x8xbf16>,
                                         %b: vector<1x1x8x8xbf16>,
                                         %c: vector<1x1x8x8xf32>)
    -> (vector<1x1x8x8xf32>, vector<1x1x8x8xf32>) {
  %0 = arith.extf %a : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
  %1 = arith.extf %b : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
  %2 = vector.contract {indexing_maps = [#mapA, #mapB, #mapC],
                        iterator_types = ["parallel", "parallel", "reduction",
                                          "parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %c
       : vector<1x1x8x8xf32>, vector<1x1x8x8xf32> into vector<1x1x8x8xf32>
  return %2, %0 : vector<1x1x8x8xf32>, vector<1x1x8x8xf32>
}

// Without a contraction user, an n-D extf is flattened like any other
// elementwise op.
// CHECK-LABEL: func @extf_2x16
// CHECK: %[[FLAT:.*]] = vector.shape_cast %{{.*}} : vector<2x16xbf16> to vector<32xbf16>
// CHECK: %[[UPS:.*]] = aievec.ups %[[FLAT]] {{.*}} : vector<32xbf16>, vector<32xf32>
// CHECK: %[[EXT:.*]] = aievec.cast %[[UPS]]
// CHECK: vector.shape_cast %[[EXT]] : vector<32xf32> to vector<2x16xf32>
func.func @extf_2x16(%a: vector<2x16xbf16>) -> vector<2x16xf32> {
  %0 = arith.extf %a : vector<2x16xbf16> to vector<2x16xf32>
  return %0 : vector<2x16xf32>
}
