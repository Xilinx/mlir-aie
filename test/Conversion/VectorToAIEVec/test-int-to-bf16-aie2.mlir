//===- test-int-to-bf16-aie2.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec='aie-target=aie2' -cse | FileCheck %s

// On AIE2, arith.sitofp v16i16 -> v16bf16 adds the magic constant as an
// arith.addi (AIE2 has no integer add on the accumulator), and arith.uitofp
// is scalarized.

// CHECK-LABEL: func.func @sitofp_v16
// CHECK: aievec.ups {{.*}} : vector<16xi16>, vector<16xi32>
// CHECK: arith.addi {{.*}} : vector<16xi32>
// CHECK-NOT: aievec.add_elem
// CHECK: aievec.srs
func.func @sitofp_v16(%a: vector<16xi16>) -> vector<16xbf16> {
  %0 = arith.sitofp %a : vector<16xi16> to vector<16xbf16>
  return %0 : vector<16xbf16>
}

// CHECK-LABEL: func.func @uitofp_v16
// CHECK-NOT: aievec.add_elem
// CHECK-NOT: aievec.ups
// CHECK: arith.uitofp {{.*}} : i16 to bf16
func.func @uitofp_v16(%a: vector<16xi16>) -> vector<16xbf16> {
  %0 = arith.uitofp %a : vector<16xi16> to vector<16xbf16>
  return %0 : vector<16xbf16>
}
