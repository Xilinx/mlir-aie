//===- badmem_toomany_bds.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --canonicalize %s |& FileCheck %s
// CHECK: 'AIE.shimDMA' op has more than 16 blocks

AIE.device(xcvc1902) {
  %t1 = AIE.tile(2, 0)
  %buf = AIE.external_buffer : memref<256xi32>
  %mem = AIE.shimDMA(%t1) {
    %dma0 = AIE.dmaStart("MM2S", 0, ^bd0, ^bd15)
    ^bd0:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd1
    ^bd1:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd2
    ^bd2:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd3
    ^bd3:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd4
    ^bd4:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd5
    ^bd5:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd6
    ^bd6:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd7
    ^bd7:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd8
    ^bd8:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd9
    ^bd9:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd10
    ^bd10:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd11
    ^bd11:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd12
    ^bd12:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd13
    ^bd13:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd14
    ^bd14:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd15
    ^bd15:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.nextBd ^bd16
    ^bd16:
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.end
  }
}
