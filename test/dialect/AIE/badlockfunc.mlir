//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aiecc.py %s |& FileCheck %s
// CHECK: error{{.*}}'AIE.lock' op is accessed outside of a tile
module @test {
  %t1 = AIE.tile(1, 1)
  %t2 = AIE.tile(4, 4)
  %lock = AIE.lock(%t2, 3) { sym_name = "lock1" }

  func.func @task3() -> () {
    AIE.useLock(%lock, "Acquire", 1)
    return
  }
}
