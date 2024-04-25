//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not %PYTHON aiecc.py --basic-alloc-scheme %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.lock' op is accessed outside of a tile
module @test {
  %t1 = aie.tile(1, 1)
  %t2 = aie.tile(4, 4)
  %lock = aie.lock(%t2, 3) { sym_name = "lock1" }

  func.func @task3() -> () {
    aie.use_lock(%lock, "Acquire", 1)
    return
  }
}
