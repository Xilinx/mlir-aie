//===- test_lock0.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL: module @test_lock0 {
// CHECK:  %0 = AIE.tile(3, 3)
// CHECK:  %1 = AIE.tile(2, 3)
// CHECK:  %2 = AIE.lock(%1, 0)
// CHECK:  %5 = AIE.core(%0) {
// CHECK:    AIE.useLock(%2, Acquire, 0)
// CHECK:    AIE.useLock(%2, Release, 1)
// CHECK:  }
// CHECK:  %6 = AIE.core(%1) {
// CHECK:    AIE.useLock(%2, Acquire, 1)
// CHECK:    AIE.useLock(%2, Release, 0)
// CHECK:  }

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// Tile-Tile
module @test_lock0 {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)
  %t23 = AIE.tile(2, 3)

  AIEX.token(0) {sym_name = "token0"}

  %m33 = AIE.mem(%t33) {
      AIE.end
  }

  %m23 = AIE.mem(%t23) {
      AIE.end
  }

  %c33 = AIE.core(%t33) {
    AIEX.useToken @token0(Acquire, 0)
    AIEX.useToken @token0(Release, 1)
    AIE.end
  }

  %c23 = AIE.core(%t23) {
    AIEX.useToken @token0(Acquire, 1)
    AIEX.useToken @token0(Release, 2)
    AIE.end
  }
 }
}
