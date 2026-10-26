//===- test-shuffle-aie2p-only-modes-on-aie2.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm='aie-target=aie2' -verify-diagnostics

// The AIE2 lowering rejects the modes AIE2 does not have.

func.func @aie2p_mode_on_aie2(%v : vector<16xi32>) -> vector<16xi32> {
  // expected-error @+2 {{shuffle mode 't32_1x16_flip' is not available on AIE2}}
  // expected-error @+1 {{failed to legalize operation 'aievec.shuffle'}}
  %0 = aievec.shuffle %v [t32_1x16_flip] : vector<16xi32>
  return %0 : vector<16xi32>
}
