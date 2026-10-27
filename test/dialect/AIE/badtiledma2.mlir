//===- badtile.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.dma_bd' op Core tile DMAs can only access a buffer in the same tile.

module @test {
    %t63 = aie.tile(6, 3)
    %t73 = aie.tile(7, 3)
    %t74 = aie.tile(7, 4)
    %t72 = aie.tile(7, 2)

    aie.buffer @east(%t63) : memref<256xi32>
    aie.buffer @local(%t73) : memref<256xi32>
    aie.buffer @north(%t74) : memref<256xi32>
    aie.buffer @south(%t72) : memref<256xi32>

    aie.lock @lock_e(%t63, 0)
    aie.lock @lock_l(%t73, 0)
    aie.lock @lock_n(%t74, 0)
    aie.lock @lock_s(%t72, 0)

    // Tile DMA
    %m73 = aie.mem(%t73) {
        %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^dma1)
      ^dma1:
        %dstDma = aie.dma_start("S2MM", 0, ^bd2, ^end)
      ^bd0:
        %buf_e = aie.get_buffer @east : memref<256xi32>
        aie.dma_bd(%buf_e : memref<256xi32> offset = 0 len = 256)
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(@lock_e, Release, %c1_ul1)
        aie.next_bd ^bd1
      ^bd1:
        %buf_l = aie.get_buffer @local : memref<256xi32>
        aie.dma_bd(%buf_l : memref<256xi32> offset = 0 len = 256)
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(@lock_l, Release, %c1_ul2)
        aie.next_bd ^end
      ^bd2:
        %buf_n = aie.get_buffer @north : memref<256xi32>
        aie.dma_bd(%buf_n : memref<256xi32> offset = 0 len = 256)
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(@lock_n, Release, %c1_ul3)
        aie.next_bd ^bd3
      ^bd3:
        %buf_s = aie.get_buffer @south : memref<256xi32>
        aie.dma_bd(%buf_s : memref<256xi32> offset = 0 len = 256)
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(@lock_s, Release, %c1_ul4)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
