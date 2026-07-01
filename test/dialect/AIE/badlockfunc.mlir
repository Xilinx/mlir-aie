//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
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
