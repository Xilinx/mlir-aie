//===- badmemtiledma2.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: 'aie.memtile_dma' op uses more output channels than available on this tile

module @test {
  aie.device(xcve2802) {
    %t12 = aie.tile(1, 2)

      aie.buffer @buf_0(%t12) : memref<256xi32>
      aie.buffer @buf_1(%t12) : memref<256xi32>
      aie.buffer @buf_2(%t12) : memref<256xi32>
      aie.buffer @buf_3(%t12) : memref<256xi32>
      aie.buffer @buf_4(%t12) : memref<256xi32>
      aie.buffer @buf_5(%t12) : memref<256xi32>
      aie.buffer @buf_6(%t12) : memref<256xi32>

      aie.lock @lock_0(%t12, 0)
      aie.lock @lock_1(%t12, 1)
      aie.lock @lock_2(%t12, 2)
      aie.lock @lock_3(%t12, 3)
      aie.lock @lock_4(%t12, 4)
      aie.lock @lock_5(%t12, 5)
      aie.lock @lock_6(%t12, 6)

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
          %buf_0 = aie.get_buffer @buf_0 : memref<256xi32>
          aie.dma_bd(%buf_0 : memref<256xi32> offset = 0 len = 256)
          %c1_ul1 = arith.constant 1 : i32
          aie.use_lock(@lock_0, Release, %c1_ul1)
          aie.next_bd ^bd0
        ^bd1:
          %buf_1 = aie.get_buffer @buf_1 : memref<256xi32>
          aie.dma_bd(%buf_1 : memref<256xi32> offset = 0 len = 256)
          %c1_ul2 = arith.constant 1 : i32
          aie.use_lock(@lock_1, Release, %c1_ul2)
          aie.next_bd ^bd1
        ^bd2:
          %buf_2 = aie.get_buffer @buf_2 : memref<256xi32>
          aie.dma_bd(%buf_2 : memref<256xi32> offset = 0 len = 256)
          %c1_ul3 = arith.constant 1 : i32
          aie.use_lock(@lock_2, Release, %c1_ul3)
          aie.next_bd ^bd2
        ^bd3:
          %buf_3 = aie.get_buffer @buf_3 : memref<256xi32>
          aie.dma_bd(%buf_3 : memref<256xi32> offset = 0 len = 256)
          %c1_ul4 = arith.constant 1 : i32
          aie.use_lock(@lock_3, Release, %c1_ul4)
          aie.next_bd ^bd3
        ^bd4:
          %buf_4 = aie.get_buffer @buf_4 : memref<256xi32>
          aie.dma_bd(%buf_4 : memref<256xi32> offset = 0 len = 256)
          %c1_ul5 = arith.constant 1 : i32
          aie.use_lock(@lock_4, Release, %c1_ul5)
          aie.next_bd ^bd4
        ^bd5:
          %buf_5 = aie.get_buffer @buf_5 : memref<256xi32>
          aie.dma_bd(%buf_5 : memref<256xi32> offset = 0 len = 256)
          %c1_ul6 = arith.constant 1 : i32
          aie.use_lock(@lock_5, Release, %c1_ul6)
          aie.next_bd ^bd5
        ^bd6:
          %buf_6 = aie.get_buffer @buf_6 : memref<256xi32>
          aie.dma_bd(%buf_6 : memref<256xi32> offset = 0 len = 256)
          %c1_ul7 = arith.constant 1 : i32
          aie.use_lock(@lock_6, Release, %c1_ul7)
          aie.next_bd ^bd6
        ^end:
          aie.end
      }
  }
}
