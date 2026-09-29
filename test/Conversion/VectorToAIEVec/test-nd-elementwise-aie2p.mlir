//===- test-nd-elementwise-aie2p.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The AIE2P legality predicates select on lane COUNT -- `getVectorLaneSize` is
// the product of every dimension -- while the patterns that implement them
// match rank-1 operands. Each of these has a native lane count and a rank
// above one, so before this they passed the legality check, found no pattern,
// and failed the conversion outright.

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

// The dividend must still reach `ConvertDivFToAIEVecInvOpPattern` as an
// `arith.constant` of 1.0, so the splat is rematerialised at the flat type
// rather than shape_cast -- a cast in front of it would hide the constant and
// take the reciprocal out of the only pattern that lowers it. That pattern
// consumes the constant, so what proves it matched is that no constant and no
// divide are left.
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

// arith.negf is the one case whose rank-1 pattern does not decline an n-D
// operand -- ComputeNegOpPattern never checks the rank, so without this the
// rank reaches aievec.neg and only fails in AIEVecToLLVM, where the shuffle
// that widens the accumulator indexes the leading dimension.
// CHECK-LABEL: func @neg_1x1x2x8_f32
// CHECK: vector.shape_cast %{{.*}} : vector<1x1x2x8xf32> to vector<16xf32>
// CHECK: aievec.neg {{.*}} : vector<16xf32>
// CHECK: vector.shape_cast %{{.*}} : vector<16xf32> to vector<1x1x2x8xf32>
func.func @neg_1x1x2x8_f32(%a: vector<1x1x2x8xf32>) -> vector<1x1x2x8xf32> {
  %0 = arith.negf %a : vector<1x1x2x8xf32>
  return %0 : vector<1x1x2x8xf32>
}

// A multiply feeding an add is exempted as "part of an FMA", but the pattern
// that would fuse it matches the add's operand directly and only at rank 1.
// At rank 4 the multiply has to be flattened along with the add, or it stays
// legal and nothing ever lowers it.
// CHECK-LABEL: func @fma_1x1x2x8_bf16
// CHECK-NOT: arith.mulf
// CHECK: aievec
func.func @fma_1x1x2x8_bf16(%a: vector<1x1x2x8xbf16>, %b: vector<1x1x2x8xbf16>,
                            %c: vector<1x1x2x8xbf16>) -> vector<1x1x2x8xbf16> {
  %0 = arith.mulf %a, %b : vector<1x1x2x8xbf16>
  %1 = arith.addf %c, %0 : vector<1x1x2x8xbf16>
  return %1 : vector<1x1x2x8xbf16>
}

// 32 lanes as well as 16. NegOpAIE2pConversion widens to ACC2048 either way,
// so stopping the predicate at 16 left this legal, unconverted, and impossible
// for Peano.
// CHECK-LABEL: func @neg_32_f32
// CHECK-NOT: arith.negf
// CHECK: aievec.neg {{.*}} : vector<32xf32>
func.func @neg_32_f32(%a: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.negf %a : vector<32xf32>
  return %0 : vector<32xf32>
}

// Already rank-1: untouched, and in particular not re-matched into an endless
// chain of shape casts.
// CHECK-LABEL: func @mul_16_bf16
// CHECK-NOT: vector.shape_cast
// CHECK: aievec.mul_elem
func.func @mul_16_bf16(%a: vector<16xbf16>,
                       %b: vector<16xbf16>) -> vector<16xbf16> {
  %0 = arith.mulf %a, %b : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// A lane count that is not native is still declined, by the same predicates as
// before -- flattening changes the shape and never the width. The op stays
// legal, so it is never handed to a pattern and keeps its original shape.
// CHECK-LABEL: func @mul_1x1x8x8_f32
// CHECK-NOT: vector.shape_cast
// CHECK: arith.mulf {{.*}} : vector<1x1x8x8xf32>
func.func @mul_1x1x8x8_f32(%a: vector<1x1x8x8xf32>,
                           %b: vector<1x1x8x8xf32>) -> vector<1x1x8x8xf32> {
  %0 = arith.mulf %a, %b : vector<1x1x8x8xf32>
  return %0 : vector<1x1x8x8xf32>
}
