//===- locks-ve2802.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-localize-locks %s | FileCheck %s

// CHECK:  module @test_xaie0  {
// CHECK:    %0 = AIE.tile(1, 3)
// CHECK:    %1 = AIE.tile(3, 5)
// CHECK:    %2 = AIE.tile(3, 3)
// CHECK:    %3 = AIE.tile(3, 4)
// CHECK:    %4 = AIE.tile(4, 4)
// CHECK:    %5 = AIE.lock(%0, 0)
// CHECK:    %6 = AIE.lock(%3, 8)
// CHECK:    %7 = AIE.lock(%4, 8)
// CHECK:    %8 = AIE.core(%0)  {
// CHECK:      %c48 = arith.constant 48 : index
// CHECK:      AIE.useLock(%c48, Acquire, 0)
// CHECK:      AIE.useLock(%c48, Release, 1)
// CHECK:      AIE.end
// CHECK:    }
// CHECK:    %9 = AIE.core(%1)  {
// CHECK:      %c8 = arith.constant 8 : index
// CHECK:      AIE.useLock(%c8, Acquire, 0)
// CHECK:      AIE.useLock(%c8, Release, 1)
// CHECK:      AIE.end
// CHECK:    }
// CHECK:    %10 = AIE.core(%2)  {
// CHECK:      %c40 = arith.constant 40 : index
// CHECK:      AIE.useLock(%c40, Acquire, 0)
// CHECK:      AIE.useLock(%c40, Release, 1)
// CHECK:      AIE.end
// CHECK:    }
// CHECK:    %11 = AIE.core(%3)  {
// CHECK:      %c56 = arith.constant 56 : index
// CHECK:      AIE.useLock(%c56, Acquire, 0)
// CHECK:      AIE.useLock(%c56, Release, 1)
// CHECK:      AIE.end
// CHECK:    }
// CHECK:    %12 = AIE.core(%4)  {
// CHECK:      %c56 = arith.constant 56 : index
// CHECK:      %c24 = arith.constant 24 : index
// CHECK:      AIE.useLock(%c24, Acquire, 0)
// CHECK:      AIE.useLock(%c24, Release, 1)
// CHECK:      AIE.useLock(%c56, Acquire, 0)
// CHECK:      AIE.useLock(%c56, Release, 1)
// CHECK:      AIE.end
// CHECK:    }
// CHECK:  }

module @test_xaie0 {
 AIE.device(xcve2802) {
  %t13 = AIE.tile(1, 3)
  %t35 = AIE.tile(3, 5)
  %t33 = AIE.tile(3, 3)
  %t34 = AIE.tile(3, 4)
  %t44 = AIE.tile(4, 4)

  %l11_8 = AIE.lock(%t13, 0)
  %l33_8 = AIE.lock(%t34, 8)
  %l43_8 = AIE.lock(%t44, 8)

  AIE.core(%t13) {
    AIE.useLock(%l11_8, Acquire, 0)
    AIE.useLock(%l11_8, Release, 1)
    AIE.end
  }
  AIE.core(%t35) {
    AIE.useLock(%l33_8, Acquire, 0)
    AIE.useLock(%l33_8, Release, 1)
    AIE.end
  }
  AIE.core(%t33) {
    AIE.useLock(%l33_8, Acquire, 0)
    AIE.useLock(%l33_8, Release, 1)
    AIE.end
  }
  AIE.core(%t34) {
    AIE.useLock(%l33_8, Acquire, 0)
    AIE.useLock(%l33_8, Release, 1)
    AIE.end
  }
  AIE.core(%t44) {
    AIE.useLock(%l33_8, Acquire, 0)
    AIE.useLock(%l33_8, Release, 1)
    AIE.useLock(%l43_8, Acquire, 0)
    AIE.useLock(%l43_8, Release, 1)
    AIE.end
  }
 }
}
