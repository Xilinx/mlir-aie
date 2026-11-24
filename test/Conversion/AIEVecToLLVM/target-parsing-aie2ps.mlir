//===- target-parsing-aie2ps.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// AIE2PS has no AIEVec lowering. convert-aievec-to-llvm diagnoses AIEVec ops,
// and leaves a module without them unchanged.

// RUN: aie-opt %s --split-input-file --verify-diagnostics --convert-aievec-to-llvm="aie-target=aie2ps" | FileCheck %s

func.func @ups(%arg0 : vector<16xi16>) -> vector<16xi32> {
  // expected-error @+1 {{'aievec.ups' op has no LLVM lowering for target 'aie2ps'}}
  %0 = aievec.ups %arg0 {shift = 0 : i8} : vector<16xi16>, vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

func.func @aie1_mac(%a : vector<32xi16>, %b : vector<16xi16>, %acc : vector<16xi48>) -> vector<16xi48> {
  // expected-error @+1 {{'aievec_aie1.mac' op has no LLVM lowering for target 'aie2ps'}}
  %0 = aievec_aie1.mac %a, %b, %acc : vector<32xi16>, vector<16xi16>, vector<16xi48>
  return %0 : vector<16xi48>
}

// -----

// The vector divf, which the AIE2 patterns unroll into helper calls, is kept.
// CHECK-LABEL: func.func @no_aievec
// CHECK-NEXT: arith.divf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-NEXT: return
// CHECK-NOT: llvm.
func.func @no_aievec(%a : vector<4xf32>, %b : vector<4xf32>) -> vector<4xf32> {
  %0 = arith.divf %a, %b : vector<4xf32>
  return %0 : vector<4xf32>
}
