//===- badmem_toomany_bds.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --canonicalize %s 2>&1 | FileCheck %s
// CHECK: 'AIE.shim_dma' op has more than 16 blocks

AIE.device(xcvc1902) {
  %t1 = AIE.tile(2, 0)
  %buf = AIE.external_buffer : memref<256xi32>
  %mem = AIE.shim_dma(%t1) {
    %dma0 = AIE.dma_start("MM2S", 0, ^bd0, ^bd15)
    ^bd0:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd1
    ^bd1:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd2
    ^bd2:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd3
    ^bd3:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd4
    ^bd4:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd5
    ^bd5:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd6
    ^bd6:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd7
    ^bd7:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd8
    ^bd8:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd9
    ^bd9:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd10
    ^bd10:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd11
    ^bd11:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd12
    ^bd12:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd13
    ^bd13:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd14
    ^bd14:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd15
    ^bd15:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.next_bd ^bd16
    ^bd16:
      AIE.dma_bd(<%buf : memref<256xi32>, 0, 256>, A)
      AIE.end
  }
}
