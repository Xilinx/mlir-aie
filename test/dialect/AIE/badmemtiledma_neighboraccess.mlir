//===- memtiledma.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aiecc.py %s |& FileCheck %s
// CHECK: error: 'AIE.buffer' op in Column 3 and Row 1 is accessed from an unreachable tile in Column 1 and Row 1

// memtiles can only access neighboring memtiles

AIE.device(xcve2802) {
  %t1 = AIE.tile(1, 1)
  %t2 = AIE.tile(3, 1)
  %t0 = AIE.tile(0, 1)
  %buf = AIE.buffer(%t1) : memref<256xi32>
  %buf2 = AIE.buffer(%t2) : memref<256xi32>
  %buf0 = AIE.buffer(%t0) : memref<256xi32>
  %mem = AIE.memTileDMA(%t1) {
    AIE.dmaStart("MM2S", 0, ^bd0, ^dma1)
    ^dma1:
    AIE.dmaStart("MM2S", 1, ^bd15, ^dma1)
    ^bd0:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd1
    ^bd1:
      AIE.dmaBd(<%buf0 : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd2
    ^bd2:
      AIE.dmaBd(<%buf2 : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd15
    ^bd15:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd16
    ^bd16:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.end
  }
}
