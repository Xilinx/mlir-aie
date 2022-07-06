//===- test_lock2.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL: module @test_lock2 {
// CHECK:  %0 = AIE.tile(3, 3)
// CHECK:  %1 = AIE.lock(%0, 3)
// CHECK:  %2 = AIE.lock(%0, 2)
// CHECK:  %3 = AIE.lock(%0, 1)
// CHECK:  %4 = AIE.lock(%0, 0)
// CHECK:  %5 = AIE.tile(2, 3)
// CHECK:  %6 = AIE.lock(%5, 0)
// CHECK:  %7 = AIE.tile(3, 4)
// CHECK:  %8 = AIE.lock(%7, 0)
// CHECK:  %9 = AIE.tile(4, 3)
// CHECK:  %10 = AIE.lock(%9, 0)
// CHECK:  %11 = AIE.tile(3, 2)
// CHECK:  %12 = AIE.lock(%11, 0)
// CHECK:  %13 = AIE.core(%5) {
// CHECK:    AIE.useLock(%6, Acquire, 0)
// CHECK:    AIE.useLock(%6, Release, 1)
// CHECK:  }
// CHECK:  %14 = AIE.core(%0) {
// CHECK:    AIE.useLock(%2, Acquire, 0)
// CHECK:    AIE.useLock(%3, Acquire, 0)
// CHECK:    AIE.useLock(%4, Acquire, 0)
// CHECK:    AIE.useLock(%1, Acquire, 0)
// CHECK:    AIE.useLock(%6, Release, 0)
// CHECK:    AIE.useLock(%4, Release, 1)
// CHECK:    AIE.useLock(%3, Release, 1)
// CHECK:    AIE.useLock(%2, Release, 1)
// CHECK:  }
// CHECK:  %15 = AIE.core(%7) {
// CHECK:    AIE.useLock(%4, Acquire, 1)
// CHECK:    AIE.useLock(%8, Release, 0)
// CHECK:  }
// CHECK:  %16 = AIE.core(%9) {
// CHECK:    AIE.useLock(%3, Acquire, 1)
// CHECK:    AIE.useLock(%10, Release, 0)
// CHECK:  }
// CHECK:  %17 = AIE.core(%11) {
// CHECK:    AIE.useLock(%2, Acquire, 1)
// CHECK:    AIE.useLock(%12, Release, 0)
// CHECK:  }
// CHECK:}

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
//      Tile
//       |
// Tile-Tile-Tile
//       |
//      Tile
// single producer (tile(3, 3)), multiple consumers
module @test_lock2 {
  %t33 = AIE.tile(3, 3)
  %t23 = AIE.tile(2, 3)
  %t34 = AIE.tile(3, 4)
  %t43 = AIE.tile(4, 3)
  %t32 = AIE.tile(3, 2)

  AIE.token(0) {sym_name = "token0"}
  AIE.token(0) {sym_name = "token1"}
  AIE.token(0) {sym_name = "token2"}
  AIE.token(0) {sym_name = "token3"}

  %c23 = AIE.core(%t23) {
    AIE.useToken @token0(Acquire, 1)
    AIE.useToken @token0(Release, 2)
    AIE.end
  }

  %c33 = AIE.core(%t33) {
    AIE.useToken @token3(Acquire, 0)
    AIE.useToken @token2(Acquire, 0)
    AIE.useToken @token1(Acquire, 0)
    AIE.useToken @token0(Acquire, 0)
    AIE.useToken @token0(Release, 1)
    AIE.useToken @token1(Release, 1)
    AIE.useToken @token2(Release, 1)
    AIE.useToken @token3(Release, 1)
    AIE.end
  }

  %c34 = AIE.core(%t34) {
    AIE.useToken @token1(Acquire, 1)
    AIE.useToken @token1(Release, 2)
    AIE.end
  }

  %c43 = AIE.core(%t43) {
    AIE.useToken @token2(Acquire, 1)
    AIE.useToken @token2(Release, 2)
    AIE.end
  }

  %c32 = AIE.core(%t32) {
    AIE.useToken @token3(Acquire, 1)
    AIE.useToken @token3(Release, 2)
    AIE.end
  }
}
