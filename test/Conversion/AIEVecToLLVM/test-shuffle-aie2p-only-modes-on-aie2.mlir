//===- test-shuffle-aie2p-only-modes-on-aie2.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm='aie-target=aie2' -split-input-file -verify-diagnostics

// The AIE2 lowering rejects the modes AIE2 does not have.

func.func @t16_8x8_lo_on_aie2(%v : vector<32xbf16>) -> vector<32xbf16> {
  // expected-error @+2 {{shuffle mode 't16_8x8_lo' is not available on AIE2}}
  // expected-error @+1 {{failed to legalize operation 'aievec.shuffle'}}
  %0 = aievec.shuffle %v, %v [t16_8x8_lo] : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

func.func @t16_8x8_hi_on_aie2(%v : vector<32xi16>) -> vector<32xi16> {
  // expected-error @+2 {{shuffle mode 't16_8x8_hi' is not available on AIE2}}
  // expected-error @+1 {{failed to legalize operation 'aievec.shuffle'}}
  %0 = aievec.shuffle %v, %v [t16_8x8_hi] : vector<32xi16>
  return %0 : vector<32xi16>
}
