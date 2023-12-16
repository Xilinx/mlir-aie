//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --canonicalize %s 2>&1 | FileCheck %s
// CHECK: 'cf.br' op is not an allowed terminator

module @test {
  %t1 = AIE.tile(1, 1)

  %mem13 = AIE.mem(%t1) {
    %dma0 = AIE.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      cf.br ^end
    ^end:
      AIE.end
  }
}
