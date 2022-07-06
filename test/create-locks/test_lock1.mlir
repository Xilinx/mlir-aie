//===- test_lock1.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL: module @test_lock1 {
// CHECK:  %0 = AIE.tile(3, 2)
// CHECK:  %1 = AIE.lock(%0, 0)
// CHECK:  %2 = AIE.tile(3, 3)
// CHECK:  %3 = AIE.lock(%2, 0)
// CHECK:  %4 = AIE.tile(3, 4)
// CHECK:  %5 = AIE.lock(%4, 0)
// CHECK:  %6 = AIE.core(%0) {
// CHECK:    AIE.useLock(%1, Acquire, 0)
// CHECK:    AIE.useLock(%1, Release, 1)
// CHECK:  }
// CHECK:  %7 = AIE.core(%2) {
// CHECK:    AIE.useLock(%1, Acquire, 1)
// CHECK:    AIE.useLock(%3, Release, 0)
// CHECK:  }
// CHECK:  %8 = AIE.core(%4) {
// CHECK:    AIE.useLock(%3, Acquire, 0)
// CHECK:    AIE.useLock(%5, Release, 0)
// CHECK:  }
// CHECK:}

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// Tile-Tile-Tile
module @test_lock1 {
  %t32 = AIE.tile(3, 2)
  %t33 = AIE.tile(3, 3)
  %t34 = AIE.tile(3, 4)

  AIE.token(0) {sym_name = "token0"}

  %c32 = AIE.core(%t32) {
    AIE.useToken @token0(Acquire, 0)
    AIE.useToken @token0(Release, 1)
    AIE.end
  }

  %c33 = AIE.core(%t33) {
    AIE.useToken @token0(Acquire, 1)
    AIE.useToken @token0(Release, 2)
    AIE.end
  }

  %c34 = AIE.core(%t34) {
    AIE.useToken @token0(Acquire, 2)
    AIE.useToken @token0(Release, 3)
    AIE.end
  }
}
