//===- test-shuffle-aie2p.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm='aie-target=aie2p' -canonicalize | FileCheck %s
// RUN: aie-opt %s --convert-aievec-to-llvm='aie-target=aie2' -canonicalize | FileCheck %s --check-prefix=AIE2

// The mode number is AIE2's; the intrinsic is AIE2P's.
// CHECK-LABEL: @shuffle_t8_64x2_hi
// CHECK-SAME: %[[A:.*]]: vector<64xi8>, %[[B:.*]]: vector<64xi8>
func.func @shuffle_t8_64x2_hi(%a : vector<64xi8>, %b : vector<64xi8>) -> vector<64xi8> {
  // CHECK: %[[M:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: "xllvm.intr.aie2p.vshuffle"(%{{.*}}, %{{.*}}, %[[M]]) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
  // CHECK-NOT: xllvm.intr.aie2.vshuffle
  // AIE2: "xllvm.intr.aie2.vshuffle"
  %0 = aievec.shuffle %a, %b [t8_64x2_hi] : vector<64xi8>
  return %0 : vector<64xi8>
}

// One operand: the lhs fills both inputs.
// CHECK-LABEL: @shuffle_single_operand
// CHECK-SAME: %[[A:.*]]: vector<16xi32>
// CHECK: %[[M:.*]] = llvm.mlir.constant(34 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%[[A]], %[[A]], %[[M]])
func.func @shuffle_single_operand(%a : vector<16xi32>) -> vector<16xi32> {
  %0 = aievec.shuffle %a [t32_4x4] : vector<16xi32>
  return %0 : vector<16xi32>
}

// The 256-bit lane mode the repeated-vector lowering uses, on 2 x i256.
// CHECK-LABEL: @shuffle_t256_2x2_lo
// CHECK: %[[M:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK: "xllvm.intr.aie2p.vshuffle"(%{{.*}}, %{{.*}}, %[[M]])
func.func @shuffle_t256_2x2_lo(%a : vector<16xi32>) -> vector<16xi32> {
  %w = vector.bitcast %a : vector<16xi32> to vector<2xi256>
  %r = aievec.shuffle %w, %w [t256_2x2_lo] : vector<2xi256>
  %o = vector.bitcast %r : vector<2xi256> to vector<16xi32>
  return %o : vector<16xi32>
}
