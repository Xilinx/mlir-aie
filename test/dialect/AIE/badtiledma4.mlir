//===- badtiledma4.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.mem' op uses more output channels than available on this tile

module @test {
  aie.device(npu) {
    %t03 = aie.tile(0, 3)

    %buf_e = aie.buffer(%t03) : memref<256xi32>
    %buf_l = aie.buffer(%t03) : memref<256xi32>
    %buf_n = aie.buffer(%t03) : memref<256xi32>

    %lock_e = aie.lock(%t03, 0)
    %lock_l = aie.lock(%t03, 1)
    %lock_n = aie.lock(%t03, 2)

    // Tile DMA
    %m03 = aie.mem(%t03) {
        %dma = aie.dma_start("MM2S", 0, ^bd0, ^dma1)
      ^dma1:
        %dma2 = aie.dma_start("MM2S", 1, ^bd1, ^dma2)
      ^dma2:
        %dma3 = aie.dma_start("MM2S", 2, ^bd2, ^end)
      ^bd0:
        aie.dma_bd(%buf_l : memref<256xi32>, 0, 256)
        aie.use_lock(%lock_e, Release, 1)
        aie.next_bd ^bd0
      ^bd1:
        aie.dma_bd(%buf_l : memref<256xi32>, 0, 256)
        aie.use_lock(%lock_l, Release, 1)
        aie.next_bd ^bd1
      ^bd2:
        aie.dma_bd(%buf_l : memref<256xi32>, 0, 256)
        aie.use_lock(%lock_n, Release, 1)
        aie.next_bd ^bd2
      ^end:
        aie.end
    }
  }
}
