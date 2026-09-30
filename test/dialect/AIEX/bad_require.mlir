//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-translate --aie-npu-to-binary -aie-output-binary=false %s 2>&1 | FileCheck %s
// RUN: aie-opt --canonicalize %s | FileCheck %s --check-prefix=DEAD

// A constraint that is already false where the static lowering reaches it is
// a hard error, the same outcome as the ValueError the static Python path
// raises. It is not a verifier error: a specialized sequence may carry one in
// a branch that canonicalization folds away (the ragged tail of a tiling
// whose row count divides evenly), so the guard only bites when live.

// CHECK: error: 'aiex.npu.require' op shape constraint is violated at compile time: A must tile into (m*rows, k) blocks

// DEAD-LABEL: aie.runtime_sequence @dead_branch
// DEAD-NOT: scf.if
// DEAD-NOT: aiex.npu.require
module {
  aie.device(npu1) {
    aie.runtime_sequence @live(%buf : memref<32xi32>) {
      %false = arith.constant false
      aiex.npu.require(%false) {message = "A must tile into (m*rows, k) blocks"} : i1
    }
    aie.runtime_sequence @dead_branch(%buf : memref<32xi32>) {
      %false = arith.constant false
      scf.if %false {
        aiex.npu.require(%false) {message = "ragged tail index exceeds the grid"} : i1
      }
    }
  }
}
