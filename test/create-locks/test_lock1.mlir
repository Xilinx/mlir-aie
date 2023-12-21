//===- test_lock1.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_1:.*]] = aie.lock(%[[VAL_0]], 0)
// CHECK:           %[[VAL_2:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_2]], 0)
// CHECK:           %[[VAL_4:.*]] = aie.tile(4, 3)
// CHECK:           aiex.token(0) {sym_name = "token0"}
// CHECK:           %[[VAL_8:.*]] = aie.core(%[[VAL_2]]) {
// CHECK:             aie.use_lock(%[[VAL_3]], Acquire, 0)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             aie.use_lock(%[[VAL_1]], Acquire, 0)
// CHECK:             aie.use_lock(%[[VAL_3]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_1]], Release, 1)
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = aie.core(%[[VAL_4]]) {
// CHECK:             aie.use_lock(%[[VAL_1]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_1]], Release, 0)
// CHECK:           }
// CHECK:         }

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// Tile-Tile-Tile
module @test_lock1 {
 aie.device(xcvc1902) {
  %t33 = aie.tile(3, 3)
  %t23 = aie.tile(2, 3)
  %t43 = aie.tile(4, 3)

  aiex.token(0) {sym_name = "token0"}

  %m33 = aie.mem(%t33) {
      aie.end
  }

  %m23 = aie.mem(%t23) {
      aie.end
  }

  %m43 = aie.mem(%t43) {
      aie.end
  }

  %c23 = aie.core(%t23) {
    aiex.useToken @token0(Acquire, 0)
    aiex.useToken @token0(Release, 1)
    aie.end
  }

  %c33 = aie.core(%t33) {
    aiex.useToken @token0(Acquire, 1)
    aiex.useToken @token0(Release, 2)
    aie.end
  }

  %c43 = aie.core(%t43) {
    aiex.useToken @token0(Acquire, 2)
    aiex.useToken @token0(Release, 3)
    aie.end
  }
 }
}
