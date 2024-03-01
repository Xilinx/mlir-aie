//===- badmemtiledma2.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: 'aie.memtile_dma' op uses more output channels than available on this tile

module @test {
  aie.device(xcve2802) {
    %t12 = aie.tile(1, 2)

      %buf_0 = aie.buffer(%t12) : memref<256xi32>
      %buf_1 = aie.buffer(%t12) : memref<256xi32>
      %buf_2 = aie.buffer(%t12) : memref<256xi32>
      %buf_3 = aie.buffer(%t12) : memref<256xi32>
      %buf_4 = aie.buffer(%t12) : memref<256xi32>
      %buf_5 = aie.buffer(%t12) : memref<256xi32>
      %buf_6 = aie.buffer(%t12) : memref<256xi32>

      %lock_0 = aie.lock(%t12, 0)
      %lock_1 = aie.lock(%t12, 1)
      %lock_2 = aie.lock(%t12, 2)
      %lock_3 = aie.lock(%t12, 3)
      %lock_4 = aie.lock(%t12, 4)
      %lock_5 = aie.lock(%t12, 5)
      %lock_6 = aie.lock(%t12, 6)

      // Mem Tile DMA
      %m12 = aie.memtile_dma(%t12) {
          %dma = aie.dma_start("MM2S", 0, ^bd0, ^dma1)
        ^dma1:
          %dma1 = aie.dma_start("MM2S", 1, ^bd1, ^dma2)
        ^dma2:
          %dma2 = aie.dma_start("MM2S", 2, ^bd2, ^dma3)
        ^dma3:
          %dma3 = aie.dma_start("MM2S", 3, ^bd3, ^dma4)
        ^dma4:
          %dma4 = aie.dma_start("MM2S", 4, ^bd4, ^dma5)
        ^dma5:
          %dma5 = aie.dma_start("MM2S", 5, ^bd5, ^dma6)
        ^dma6:
          %dma6 = aie.dma_start("MM2S", 6, ^bd6, ^end)
        ^bd0:
          aie.dma_bd(%buf_0 : memref<256xi32>, 0, 256)
          aie.use_lock(%lock_0, Release, 1)
          aie.next_bd ^bd0
        ^bd1:
          aie.dma_bd(%buf_1 : memref<256xi32>, 0, 256)
          aie.use_lock(%lock_1, Release, 1)
          aie.next_bd ^bd1
        ^bd2:
          aie.dma_bd(%buf_2 : memref<256xi32>, 0, 256)
          aie.use_lock(%lock_2, Release, 1)
          aie.next_bd ^bd2
        ^bd3:
          aie.dma_bd(%buf_3 : memref<256xi32>, 0, 256)
          aie.use_lock(%lock_3, Release, 1)
          aie.next_bd ^bd3
        ^bd4:
          aie.dma_bd(%buf_4 : memref<256xi32>, 0, 256)
          aie.use_lock(%lock_4, Release, 1)
          aie.next_bd ^bd4
        ^bd5:
          aie.dma_bd(%buf_5 : memref<256xi32>, 0, 256)
          aie.use_lock(%lock_5, Release, 1)
          aie.next_bd ^bd5
        ^bd6:
          aie.dma_bd(%buf_6 : memref<256xi32>, 0, 256)
          aie.use_lock(%lock_6, Release, 1)
          aie.next_bd ^bd6
        ^end:
          aie.end
      }
  }
}