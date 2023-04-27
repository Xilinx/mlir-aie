//===- badmemtiledma.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --canonicalize %s |& FileCheck %s
// CHECK: 'AIE.dmaStart' op duplicate DMA channel MM2S0 not allowed

AIE.device(xcve2802) {
  %t1 = AIE.tile(1, 1)
  %buf = AIE.buffer(%t1) : memref<256xi32>
  %mem = AIE.memTileDMA(%t1) {
    AIE.dmaStart("MM2S", 0, ^bd0, ^dma1)
    ^dma1:
    AIE.dmaStart("MM2S", 0, ^bd0, ^dma1)
    ^bd0:
      AIE.end
  }
}
