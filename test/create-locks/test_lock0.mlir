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

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_2:.*]] = aie.lock(%[[VAL_1]], 0)
// CHECK:           %[[VAL_5:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             aie.use_lock(%[[VAL_2]], Acquire, 0)
// CHECK:             aie.use_lock(%[[VAL_2]], Release, 1)
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             aie.use_lock(%[[VAL_2]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_2]], Release, 0)
// CHECK:           }
// CHECK:         }

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// Tile-Tile
module @test_lock0 {
 aie.device(xcvc1902) {
  %t33 = aie.tile(3, 3)
  %t23 = aie.tile(2, 3)

  aiex.token(0) {sym_name = "token0"}

  %m33 = aie.mem(%t33) {
      aie.end
  }

  %m23 = aie.mem(%t23) {
      aie.end
  }

  %c33 = aie.core(%t33) {
    aiex.useToken @token0(Acquire, 0)
    aiex.useToken @token0(Release, 1)
    aie.end
  }

  %c23 = aie.core(%t23) {
    aiex.useToken @token0(Acquire, 1)
    aiex.useToken @token0(Release, 2)
    aie.end
  }
 }
}
