//===- locks1.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-logical-lock %s | FileCheck %s

// CHECK:  module @test_xaie0 {
// CHECK:    %0 = AIE.tile(1, 1)
// CHECK:    %1 = AIE.tile(3, 4)
// CHECK:    %2 = AIE.tile(3, 2)
// CHECK:    %3 = AIE.tile(3, 3)
// CHECK:    %4 = AIE.tile(4, 3)
// CHECK:    %5 = AIE.lock(%0, 0)
// CHECK:    %6 = AIE.lock(%3, 0)
// CHECK:    %7 = AIE.lock(%3, 1)
// CHECK:    %8 = AIE.lock(%4, 0)
// CHECK:    %9 = AIE.core(%0) {
// CHECK:      AIE.useLock(%5, Acquire, 0)
// CHECK:      AIE.useLock(%5, Release, 1)
// CHECK:      AIE.end
// CHECK:    }
// CHECK:    %10 = AIE.core(%1) {
// CHECK:      AIE.useLock(%6, Acquire, 0)
// CHECK:      AIE.useLock(%6, Release, 1)
// CHECK:      AIE.end
// CHECK:    }
// CHECK:    %11 = AIE.core(%2) {
// CHECK:      AIE.useLock(%6, Acquire, 0)
// CHECK:      AIE.useLock(%6, Release, 1)
// CHECK:      AIE.end
// CHECK:    }
// CHECK:    %12 = AIE.core(%3) {
// CHECK:      AIE.useLock(%6, Acquire, 0)
// CHECK:      AIE.useLock(%6, Release, 1)
// CHECK:      AIE.end
// CHECK:    }
// CHECK:    %13 = AIE.core(%4) {
// CHECK:      AIE.useLock(%6, Acquire, 0)
// CHECK:      AIE.useLock(%6, Release, 1)
// CHECK:      AIE.useLock(%8, Acquire, 0)
// CHECK:      AIE.useLock(%8, Release, 1)
// CHECK:      AIE.end
// CHECK:    }
// CHECK:  }

module @test_assign_lockIDs {
  %t11 = AIE.tile(1, 1)
  %t34 = AIE.tile(3, 4)
  %t32 = AIE.tile(3, 2)
  %t33 = AIE.tile(3, 3)
  %t43 = AIE.tile(4, 3)

  %l11_8 = AIE.lock(%t11, 1)
  %l33_0 = AIE.lock(%t33, 2)
  %l33_1 = AIE.lock(%t33)
  %l43_8 = AIE.lock(%t43)  
}
