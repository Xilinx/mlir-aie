//===- bad_masterset_non_amsel.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}} 'aie.masterset' op amsel operand must be produced by an 'aie.amsel' op

// The amsel operands are plain index values, so nothing in the op's type
// signature stops an unrelated index from being passed here. Every consumer
// unconditionally casts them to aie.amsel, so the verifier has to establish it.
module {
  aie.device(npu2) {
    %c = arith.constant 0 : index
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      aie.masterset(North : 0, %c)
      aie.end
    }
  }
}
