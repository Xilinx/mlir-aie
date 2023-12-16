//===- badshim-notnoc.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --canonicalize %s 2>&1 | FileCheck %s
// CHECK: 'AIE.shim_dma' op failed to verify that op exists in a shim tile with NOC connection

module @test {
  %t1 = AIE.tile(1, 0)

  %mem13 = AIE.shim_dma(%t1) {
    AIE.dma_start("MM2S", 0, ^bd0, ^dma1)
    ^dma1:
    AIE.dma_start("MM2S", 0, ^bd0, ^dma1)
    ^bd0:
      AIE.end
  }
}
