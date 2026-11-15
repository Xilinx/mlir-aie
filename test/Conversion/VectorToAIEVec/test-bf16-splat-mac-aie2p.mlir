//===- test-bf16-splat-mac-aie2p.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s

// A splat of a scalar widened from bf16 holds bf16 values: the multiply is a
// bf16 one, the splat a bf16 broadcast.
// CHECK-LABEL: func @mulf_32_splat
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[S:.*]]: bf16
// CHECK: %[[B:.*]] = aievec.broadcast_scalar %[[S]] : bf16, vector<32xbf16>
// CHECK: aievec.mul_elem %[[A]], %[[B]] : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// CHECK-NOT: arith.mulf
func.func @mulf_32_splat(%a: vector<32xbf16>, %s: bf16) -> vector<32xf32> {
  %0 = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %1 = arith.extf %s : bf16 to f32
  %2 = vector.broadcast %1 : f32 to vector<32xf32>
  %3 = arith.mulf %0, %2 : vector<32xf32>
  return %3 : vector<32xf32>
}

// -----

// The same feeding an add, both allowing contraction: one MAC.
// CHECK-LABEL: func @mulf_32_splat_into_add
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[S:.*]]: bf16, %[[C:.*]]: vector<32xf32>
// CHECK: %[[B:.*]] = aievec.broadcast_scalar %[[S]] : bf16, vector<32xbf16>
// CHECK: aievec.mac_elem %[[A]], %[[B]], %[[C]] : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// CHECK-NOT: arith.addf
func.func @mulf_32_splat_into_add(%a: vector<32xbf16>, %s: bf16, %c: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %1 = arith.extf %s : bf16 to f32
  %2 = vector.broadcast %1 : f32 to vector<32xf32>
  %3 = arith.mulf %0, %2 fastmath<contract> : vector<32xf32>
  %4 = arith.addf %3, %c fastmath<contract> : vector<32xf32>
  return %4 : vector<32xf32>
}
