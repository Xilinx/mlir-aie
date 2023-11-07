//===- test_lock2.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_1:.*]] = AIE.lock(%[[VAL_0]], 0)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(2, 3)
// CHECK:           %[[VAL_3:.*]] = AIE.lock(%[[VAL_2]], 0)
// CHECK:           %[[VAL_4:.*]] = AIE.tile(3, 4)
// CHECK:           %[[VAL_5:.*]] = AIE.lock(%[[VAL_4]], 0)
// CHECK:           %[[VAL_6:.*]] = AIE.tile(4, 3)
// CHECK:           %[[VAL_7:.*]] = AIE.tile(3, 2)
// CHECK:           %[[VAL_8:.*]] = AIE.lock(%[[VAL_7]], 0)
// CHECK:           AIEX.token(0) {sym_name = "token0"}
// CHECK:           AIEX.token(0) {sym_name = "token1"}
// CHECK:           AIEX.token(0) {sym_name = "token2"}
// CHECK:           AIEX.token(0) {sym_name = "token3"}
// CHECK:           %[[VAL_14:.*]] = AIE.core(%[[VAL_2]]) {
// CHECK:             AIE.useLock(%[[VAL_3]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_3]], Release, 0)
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:             AIE.useLock(%[[VAL_8]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_1]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_5]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_3]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_3]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_1]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_8]], Release, 1)
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = AIE.core(%[[VAL_4]]) {
// CHECK:             AIE.useLock(%[[VAL_5]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_5]], Release, 0)
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = AIE.core(%[[VAL_6]]) {
// CHECK:             AIE.useLock(%[[VAL_1]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_1]], Release, 0)
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = AIE.core(%[[VAL_7]]) {
// CHECK:             AIE.useLock(%[[VAL_8]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_8]], Release, 0)
// CHECK:           }
// CHECK:         }

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
//      Tile
//       |
// Tile-Tile-Tile
//       |
//      Tile
// single producer (tile(3, 3)), multiple consumers
module @test_lock2 {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)
  %t23 = AIE.tile(2, 3)
  %t34 = AIE.tile(3, 4)
  %t43 = AIE.tile(4, 3)
  %t32 = AIE.tile(3, 2)

  AIEX.token(0) {sym_name = "token0"}
  AIEX.token(0) {sym_name = "token1"}
  AIEX.token(0) {sym_name = "token2"}
  AIEX.token(0) {sym_name = "token3"}

  %m33 = AIE.mem(%t33) {
      AIE.end
  }

  %m23 = AIE.mem(%t23) {
      AIE.end
  }

  %m34 = AIE.mem(%t34) {
      AIE.end
  }

  %m43 = AIE.mem(%t43) {
      AIE.end
  }

  %m32 = AIE.mem(%t32) {
      AIE.end
  }

  %c23 = AIE.core(%t23) {
    AIEX.useToken @token0(Acquire, 1)
    AIEX.useToken @token0(Release, 2)
    AIE.end
  }

  %c33 = AIE.core(%t33) {
    AIEX.useToken @token3(Acquire, 0)
    AIEX.useToken @token2(Acquire, 0)
    AIEX.useToken @token1(Acquire, 0)
    AIEX.useToken @token0(Acquire, 0)
    AIEX.useToken @token0(Release, 1)
    AIEX.useToken @token1(Release, 1)
    AIEX.useToken @token2(Release, 1)
    AIEX.useToken @token3(Release, 1)
    AIE.end
  }

  %c34 = AIE.core(%t34) {
    AIEX.useToken @token1(Acquire, 1)
    AIEX.useToken @token1(Release, 2)
    AIE.end
  }

  %c43 = AIE.core(%t43) {
    AIEX.useToken @token2(Acquire, 1)
    AIEX.useToken @token2(Release, 2)
    AIE.end
  }

  %c32 = AIE.core(%t32) {
    AIEX.useToken @token3(Acquire, 1)
    AIEX.useToken @token3(Release, 2)
    AIE.end
  }
 }
}
