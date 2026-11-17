//===- test-bf16-emulation-fma-aie2p.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A contracting f32 multiply and add under bf16 emulation, lowered to AIE2P:
// fused into a MAC where the vector is one the MAC takes, and lowered as a
// multiply and an add otherwise.

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie2p bf16-emulation=true" | FileCheck %s

// CHECK-LABEL: func @mul_add_32
// CHECK: %[[MAC:.*]] = aievec.mac_elem {{.*}} : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// CHECK-NOT: aievec.add_elem
// CHECK: return %[[MAC]]
func.func @mul_add_32(%a: vector<32xf32>, %b: vector<32xf32>, %c: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %1 = arith.addf %0, %c fastmath<contract> : vector<32xf32>
  return %1 : vector<32xf32>
}

// -----

// CHECK-LABEL: func @mul_add_8x8
// CHECK-NOT: vector.fma
// CHECK: %[[MUL:.*]] = aievec.mul_elem {{.*}} : vector<64xbf16>, vector<64xbf16>, vector<64xf32>
// CHECK: aievec.add_elem %[[MUL]], {{.*}} : vector<64xf32>
// CHECK-NOT: vector.fma
func.func @mul_add_8x8(%a: vector<8x8xf32>, %b: vector<8x8xf32>, %c: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = arith.mulf %a, %b fastmath<contract> : vector<8x8xf32>
  %1 = arith.addf %0, %c fastmath<contract> : vector<8x8xf32>
  return %1 : vector<8x8xf32>
}
