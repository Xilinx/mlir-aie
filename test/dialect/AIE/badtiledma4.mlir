//===- badtiledma4.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.mem' op uses more output channels than available on this tile

module @test {
  aie.device(npu1) {
    %t03 = aie.tile(0, 3)

    aie.buffer @buf_e(%t03) : memref<256xi32>
    aie.buffer @buf_l(%t03) : memref<256xi32>
    aie.buffer @buf_n(%t03) : memref<256xi32>

    aie.lock @lock_e(%t03, 0)
    aie.lock @lock_l(%t03, 1)
    aie.lock @lock_n(%t03, 2)

    // Tile DMA
    %m03 = aie.mem(%t03) {
        %dma = aie.dma_start("MM2S", 0, ^bd0, ^dma1)
      ^dma1:
        %dma2 = aie.dma_start("MM2S", 1, ^bd1, ^dma2)
      ^dma2:
        %dma3 = aie.dma_start("MM2S", 2, ^bd2, ^end)
      ^bd0:
        %buf_l = aie.get_buffer @buf_l : memref<256xi32>
        aie.dma_bd(%buf_l : memref<256xi32> offset = 0 len = 256)
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(@lock_e, Release, %c1_ul1)
        aie.next_bd ^bd0
      ^bd1:
        %buf_l_1 = aie.get_buffer @buf_l : memref<256xi32>
        aie.dma_bd(%buf_l_1 : memref<256xi32> offset = 0 len = 256)
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(@lock_l, Release, %c1_ul2)
        aie.next_bd ^bd1
      ^bd2:
        %buf_l_2 = aie.get_buffer @buf_l : memref<256xi32>
        aie.dma_bd(%buf_l_2 : memref<256xi32> offset = 0 len = 256)
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(@lock_n, Release, %c1_ul3)
        aie.next_bd ^bd2
      ^end:
        aie.end
    }
  }
}
