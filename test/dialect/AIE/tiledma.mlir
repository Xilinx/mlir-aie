//===- tiledma.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// Test that we can fill all 16 BDs

// CHECK-LABEL: module {
// CHECK:       }

AIE.device(xcvc1902) {
  %t1 = AIE.tile(1, 1)
  %buf = AIE.buffer(%t1) : memref<256xi32>
  %mem = AIE.mem(%t1) {
    AIE.dma_start("MM2S", 0, ^bd0, ^dma1)
    ^dma1:
    AIE.dma_start("MM2S", 1, ^bd15, ^dma1)
    ^bd0:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd1
    ^bd1:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd2
    ^bd2:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd3
    ^bd3:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd4
    ^bd4:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd5
    ^bd5:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd6
    ^bd6:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd7
    ^bd7:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd8
    ^bd8:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd9
    ^bd9:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd10
    ^bd10:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd11
    ^bd11:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd12
    ^bd12:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd13
    ^bd13:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd14
    ^bd14:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd15
    ^bd15:
      AIE.dma_bd(%buf : memref<256xi32>, 0, 256)
      AIE.end
  }
}
