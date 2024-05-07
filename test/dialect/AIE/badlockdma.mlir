//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.lock' op in Column 4 and Row 4 is accessed from an unreachable tile in Column 1 and Row 1
module @test {
  %t1 = aie.tile(1, 1)
  %t2 = aie.tile(4, 4)
  %lock = aie.lock(%t2, 3) { sym_name = "lock1" }

  %mem13 = aie.mem(%t1) {
    %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lock, "Acquire", 1)
      aie.use_lock(%lock, "Release", 0)
      aie.next_bd ^end // point to the next BD, or termination
    ^end:
      aie.end
  }
}
