//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aiecc.py %s |& FileCheck %s
// CHECK: error: 'AIE.lock' op in Column 4 and Row 4 is accessed from an unreachable tile in Column 1 and Row 1
module @test {
  %t1 = AIE.tile(1, 1)
  %t2 = AIE.tile(4, 4)
  %lock = AIE.lock(%t2, 3) { sym_name = "lock1" }

  AIE.core(%t1) {
    AIE.useLock(%lock, "Acquire", 1)
    AIE.end
  }
}
