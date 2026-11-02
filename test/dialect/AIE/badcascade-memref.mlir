//===- badcascade-memref.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu2) {
  %t = aie.tile(0, 2)
  %buf = aie.buffer(%t) : memref<32xbf16>
  %c = aie.core(%t) {
    // expected-error @+1 {{'aie.put_cascade' op operand #0 must be integer or floating-point or vector of any non-token type values, but got 'memref<32xbf16>'}}
    aie.put_cascade(%buf : memref<32xbf16>)
    aie.end
  }
}

// -----

aie.device(npu2) {
  %t = aie.tile(0, 2)
  %c = aie.core(%t) {
    // expected-error @+1 {{'aie.get_cascade' op result #0 must be integer or floating-point or vector of any non-token type values, but got 'memref<16xi32>'}}
    %v = aie.get_cascade() : memref<16xi32>
    aie.end
  }
}
