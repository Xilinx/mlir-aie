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

// CHECK-LABEL:   aie.device(xcve2802) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_1:.*]] = aie.tile(3, 5)
// CHECK:           %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_3:.*]] = aie.tile(3, 4)
// CHECK:           %[[VAL_4:.*]] = aie.tile(4, 4)
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]], 0)
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_3]], 8)
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_4]], 8)
// CHECK:           %[[VAL_8:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_9:.*]] = arith.constant 48 : index
// CHECK:             aie.use_lock(%[[VAL_9]], Acquire, 0)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_11:.*]] = arith.constant 8 : index
// CHECK:             aie.use_lock(%[[VAL_11]], Acquire, 0)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = aie.core(%[[VAL_2]]) {
// CHECK:             %[[VAL_13:.*]] = arith.constant 40 : index
// CHECK:             aie.use_lock(%[[VAL_13]], Acquire, 0)
// CHECK:             aie.use_lock(%[[VAL_13]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = aie.core(%[[VAL_3]]) {
// CHECK:             %[[VAL_15:.*]] = arith.constant 56 : index
// CHECK:             aie.use_lock(%[[VAL_15]], Acquire, 0)
// CHECK:             aie.use_lock(%[[VAL_15]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = aie.core(%[[VAL_4]]) {
// CHECK:             %[[VAL_17:.*]] = arith.constant 56 : index
// CHECK:             %[[VAL_18:.*]] = arith.constant 24 : index
// CHECK:             aie.use_lock(%[[VAL_18]], Acquire, 0)
// CHECK:             aie.use_lock(%[[VAL_18]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_17]], Acquire, 0)
// CHECK:             aie.use_lock(%[[VAL_17]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @test_xaie0 {
 aie.device(xcve2802) {
  %t13 = aie.tile(1, 3)
  %t35 = aie.tile(3, 5)
  %t33 = aie.tile(3, 3)
  %t34 = aie.tile(3, 4)
  %t44 = aie.tile(4, 4)

  %l11_8 = aie.lock(%t13, 0)
  %l33_8 = aie.lock(%t34, 8)
  %l43_8 = aie.lock(%t44, 8)

  aie.core(%t13) {
    aie.use_lock(%l11_8, Acquire, 0)
    aie.use_lock(%l11_8, Release, 1)
    aie.end
  }
  aie.core(%t35) {
    aie.use_lock(%l33_8, Acquire, 0)
    aie.use_lock(%l33_8, Release, 1)
    aie.end
  }
  aie.core(%t33) {
    aie.use_lock(%l33_8, Acquire, 0)
    aie.use_lock(%l33_8, Release, 1)
    aie.end
  }
  aie.core(%t34) {
    aie.use_lock(%l33_8, Acquire, 0)
    aie.use_lock(%l33_8, Release, 1)
    aie.end
  }
  aie.core(%t44) {
    aie.use_lock(%l33_8, Acquire, 0)
    aie.use_lock(%l33_8, Release, 1)
    aie.use_lock(%l43_8, Acquire, 0)
    aie.use_lock(%l43_8, Release, 1)
    aie.end
  }
 }
}
