//===- memtiledma.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// memtiles have 48 BDs, not just 16

// CHECK-LABEL: module {
// CHECK:       }

aie.device(xcve2802) {
  %t1 = aie.tile(1, 1)
  %buf = aie.buffer(%t1) : memref<256xi32>
  %mem = aie.memtile_dma(%t1) {
    aie.dma_start("MM2S", 0, ^bd0, ^dma1)
    ^dma1:
    aie.dma_start("MM2S", 1, ^bd15, ^dma1)
    ^bd0:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd4
    ^bd4:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd5
    ^bd5:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd6
    ^bd6:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd7
    ^bd7:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd8
    ^bd8:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd9
    ^bd9:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd10
    ^bd10:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd11
    ^bd11:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd12
    ^bd12:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd13
    ^bd13:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^bd14
    ^bd14:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256, [<size = 2, stride = 128>])
      aie.next_bd ^bd15
    ^bd15:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256, [<size = 2, stride = 128>], [<const_pad_before = 1, const_pad_after = 1>])
      aie.next_bd ^bd16
    ^bd16:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256, [<size = 2, stride = 128>], [<const_pad_before = 1, const_pad_after = 1>], pad_value = 0)
      aie.end
  }
}
