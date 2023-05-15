//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --canonicalize %s |& FileCheck %s
// CHECK: 'AIE.dmaStart' op duplicate DMA channel MM2S0 not allowed

module @test {
  %t1 = AIE.tile(1, 1)

  %mem13 = AIE.mem(%t1) {
    AIE.dmaStart("MM2S", 0, ^bd0, ^dma1)
    ^dma1:
    AIE.dmaStart("MM2S", 0, ^bd0, ^dma1)
    ^bd0:
      AIE.end
  }
}
