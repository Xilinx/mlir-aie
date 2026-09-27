//===- bad_require.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A constraint that is already false at compile time is a hard error, the
// same outcome as the ValueError the static Python path raises.
module {
  aie.device(npu1) {
    aie.runtime_sequence(%buf : memref<32xi32>) {
      %false = arith.constant false
      // expected-error@+1 {{shape constraint is violated at compile time: A must tile into (m*rows, k) blocks}}
      aiex.npu.require(%false) {message = "A must tile into (m*rows, k) blocks"} : i1
    }
  }
}
