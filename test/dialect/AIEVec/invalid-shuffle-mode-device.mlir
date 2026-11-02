//===- invalid-shuffle-mode-device.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Shuffle modes 48 to 55 exist on AIE2P only.

aie.device(npu1) {
  %t = aie.tile(0, 2)
  %c = aie.core(%t) {
    %v = arith.constant dense<1> : vector<16xi32>
    // expected-error @+1 {{shuffle mode 't32_1x16_flip' is only available on AIE2P}}
    %r = aievec.shuffle %v [t32_1x16_flip] : vector<16xi32>
    aie.end
  }
}

// -----

aie.device(npu1) {
  %t = aie.tile(0, 2)
  %c = aie.core(%t) {
    %v = arith.constant dense<1.0> : vector<32xbf16>
    // expected-error @+1 {{shuffle mode 't16_8x8_lo' is only available on AIE2P}}
    %r = aievec.shuffle %v, %v [t16_8x8_lo] : vector<32xbf16>
    aie.end
  }
}

// -----

aie.device(xcve2302) {
  %t = aie.tile(0, 2)
  %c = aie.core(%t) {
    %v = arith.constant dense<1> : vector<8xi64>
    // expected-error @+1 {{shuffle mode 't64_4x2' is only available on AIE2P}}
    %r = aievec.shuffle %v [t64_4x2] : vector<8xi64>
    aie.end
  }
}
