//===- test_lock0_reuse.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL: module @test_lock0_reuse {
// CHECK:  %0 = AIE.tile(3, 3)
// CHECK:  %1 = AIE.tile(2, 3)
// CHECK:  %2 = AIE.lock(%1, 1)
// CHECK:  AIE.useLock(%2, Release, 0)
// CHECK:  %6 = AIE.core(%0) {
// CHECK:    AIE.useLock(%2, Acquire, 0)
// CHECK:    AIE.useLock(%2, Release, 1)
// CHECK:  }
// CHECK:  %7 = AIE.core(%1) {
// CHECK:    AIE.useLock(%2, Acquire, 1)
// CHECK:    AIE.useLock(%2, Release, 0)
// CHECK:  }

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// Tile-Tile
module @test_lock0_reuse {
  %t33 = AIE.tile(3, 3)
  %t23 = AIE.tile(2, 3)

  %t = AIE.lock(%t23, 0)
  AIE.token(0) {sym_name = "token0"}

  %m33 = AIE.mem(%t33) {
      AIE.end
  }

  %m23 = AIE.mem(%t23) {
      AIE.end
  }

  %c33 = AIE.core(%t33) {
    AIE.useToken @token0(Acquire, 0)
    AIE.useToken @token0(Release, 1)
    AIE.end
  }

  %c23 = AIE.core(%t23) {
    AIE.useToken @token0(Acquire, 1)
    // It releases back to token 0, which should be implemented into one lock
    // instead of two locks.  In this implemented lock, Token Acquire 0 is
    // implemented as Lock Acquire 0, Token Acquire 1 is implemented as Lock
    // Acquire 1, and Token Release 0/1 should be Lock Release 0/1 as well.
    AIE.useToken @token0(Release, 0)
    AIE.end
  }
}
