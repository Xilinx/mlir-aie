//===- target-parsing.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The aie-target option of convert-aievec-to-llvm.

// "aie", "aie2", "aieml" and an empty value all select the AIE2 patterns.
// RUN: aie-opt %s --split-input-file --convert-aievec-to-llvm | FileCheck %s --check-prefix=AIE2
// RUN: aie-opt %s --split-input-file --convert-aievec-to-llvm="aie-target=aie2" | FileCheck %s --check-prefix=AIE2
// RUN: aie-opt %s --split-input-file --convert-aievec-to-llvm="aie-target=aieml" | FileCheck %s --check-prefix=AIE2
// RUN: aie-opt %s --split-input-file --convert-aievec-to-llvm="aie-target=aie" | FileCheck %s --check-prefix=AIE2
// RUN: aie-opt %s --split-input-file --convert-aievec-to-llvm="aie-target=" | FileCheck %s --check-prefix=AIE2
// RUN: aie-opt %s --split-input-file --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s --check-prefix=AIE2P

// An unknown target is an error.
// RUN: not aie-opt %s --split-input-file --convert-aievec-to-llvm="aie-target=aie3" 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not aie-opt %s --split-input-file --convert-aievec-to-llvm="aie-target=AIE2" 2>&1 | FileCheck %s --check-prefix=UNKNOWN-UPPER

// UNKNOWN: error: unknown AIE target 'aie3'
// UNKNOWN-UPPER: error: unknown AIE target 'AIE2'

// AIE2PS: see target-parsing-aie2ps.mlir.

// AIE2-LABEL: func.func @ups
// AIE2: xllvm.intr.aie2.acc32.v16.I256.ups
// AIE2P-LABEL: func.func @ups
// AIE2P: xllvm.intr.aie2p.acc32.v16.I256.ups
func.func @ups(%arg0 : vector<16xi16>) -> vector<16xi32> {
  %0 = aievec.ups %arg0 {shift = 0 : i8} : vector<16xi16>, vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// AIE2-LABEL: func.func @aie1_mac
// AIE2: llvm.aie.mac16.v32int16
func.func @aie1_mac(%a : vector<32xi16>, %b : vector<16xi16>, %acc : vector<16xi48>) -> vector<16xi48> {
  %0 = aievec_aie1.mac %a, %b, %acc : vector<32xi16>, vector<16xi16>, vector<16xi48>
  return %0 : vector<16xi48>
}

// -----

// No AIEVec ops. On AIE2 the vector divf is unrolled into helper calls.

// AIE2-LABEL: func.func @no_aievec
// AIE2: llvm.call @__aie2_scalar_fdiv
func.func @no_aievec(%a : vector<4xf32>, %b : vector<4xf32>) -> vector<4xf32> {
  %0 = arith.divf %a, %b : vector<4xf32>
  return %0 : vector<4xf32>
}
