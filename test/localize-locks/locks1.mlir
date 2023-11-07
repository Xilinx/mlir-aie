//===- locks1.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// RUN: aie-opt --aie-localize-locks %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(1, 1)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(3, 4)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(3, 2)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_4:.*]] = AIE.tile(4, 3)
// CHECK:           %[[VAL_5:.*]] = AIE.lock(%[[VAL_0]], 0)
// CHECK:           %[[VAL_6:.*]] = AIE.lock(%[[VAL_3]], 8)
// CHECK:           %[[VAL_7:.*]] = AIE.lock(%[[VAL_4]], 8)
// CHECK:           %[[VAL_8:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_9:.*]] = arith.constant 48 : index
// CHECK:             AIE.useLock(%[[VAL_9]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_9]], Release, 1)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = AIE.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_11:.*]] = arith.constant 8 : index
// CHECK:             AIE.useLock(%[[VAL_11]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = AIE.core(%[[VAL_2]]) {
// CHECK:             %[[VAL_13:.*]] = arith.constant 40 : index
// CHECK:             AIE.useLock(%[[VAL_13]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_13]], Release, 1)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = AIE.core(%[[VAL_3]]) {
// CHECK:             %[[VAL_15:.*]] = arith.constant 56 : index
// CHECK:             AIE.useLock(%[[VAL_15]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_15]], Release, 1)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = AIE.core(%[[VAL_4]]) {
// CHECK:             %[[VAL_17:.*]] = arith.constant 56 : index
// CHECK:             %[[VAL_18:.*]] = arith.constant 24 : index
// CHECK:             AIE.useLock(%[[VAL_18]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_18]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_17]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_17]], Release, 1)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @test_xaie0 {
 AIE.device(xcvc1902) {
  %t11 = AIE.tile(1, 1)
  %t34 = AIE.tile(3, 4)
  %t32 = AIE.tile(3, 2)
  %t33 = AIE.tile(3, 3)
  %t43 = AIE.tile(4, 3)

  %l11_8 = AIE.lock(%t11, 0)
  %l33_8 = AIE.lock(%t33, 8)
  %l43_8 = AIE.lock(%t43, 8)

  AIE.core(%t11) {
    AIE.useLock(%l11_8, Acquire, 0)
    AIE.useLock(%l11_8, Release, 1)
    AIE.end
  }
  AIE.core(%t34) {
    AIE.useLock(%l33_8, Acquire, 0)
    AIE.useLock(%l33_8, Release, 1)
    AIE.end
  }
  AIE.core(%t32) {
    AIE.useLock(%l33_8, Acquire, 0)
    AIE.useLock(%l33_8, Release, 1)
    AIE.end
  }
  AIE.core(%t33) {
    AIE.useLock(%l33_8, Acquire, 0)
    AIE.useLock(%l33_8, Release, 1)
    AIE.end
  }
  AIE.core(%t43) {
    AIE.useLock(%l33_8, Acquire, 0)
    AIE.useLock(%l33_8, Release, 1)
    AIE.useLock(%l43_8, Acquire, 0)
    AIE.useLock(%l43_8, Release, 1)
    AIE.end
  }
 }
}
