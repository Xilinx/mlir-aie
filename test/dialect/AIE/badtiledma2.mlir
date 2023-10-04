//===- badtile.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aiecc.py %s |& FileCheck %s
// CHECK: error{{.*}}'AIE.dmaBd' op can only access a buffer in the same tile.

module @test {
    %t63 = AIE.tile(6, 3)
    %t73 = AIE.tile(7, 3)
    %t74 = AIE.tile(7, 4)
    %t72 = AIE.tile(7, 2)

    %buf_e = AIE.buffer(%t63) { sym_name = "east" } : memref<256xi32>
    %buf_l = AIE.buffer(%t73) { sym_name = "local" } : memref<256xi32>
    %buf_n = AIE.buffer(%t74) { sym_name = "north" } : memref<256xi32>
    %buf_s = AIE.buffer(%t72) { sym_name = "south" } : memref<256xi32>

    %lock_e = AIE.lock(%t63, 0)
    %lock_l = AIE.lock(%t73, 0)
    %lock_n = AIE.lock(%t74, 0)
    %lock_s = AIE.lock(%t72, 0)

    // Tile DMA
    %m73 = AIE.mem(%t73) {
        %srcDma = AIE.dmaStart("MM2S", 0, ^bd0, ^dma1)
      ^dma1:
        %dstDma = AIE.dmaStart("S2MM", 0, ^bd2, ^end)
      ^bd0:
        AIE.dmaBd(<%buf_e : memref<256xi32>, 0, 256>, 0)
        AIE.useLock(%lock_e, Release, 1)
        AIE.nextBd ^bd1
      ^bd1:
        AIE.dmaBd(<%buf_l : memref<256xi32>, 0, 256>, 0)
        AIE.useLock(%lock_l, Release, 1)
        AIE.nextBd ^end
      ^bd2:
        AIE.dmaBd(<%buf_n : memref<256xi32>, 0, 256>, 0)
        AIE.useLock(%lock_n, Release, 1)
        AIE.nextBd ^bd3
      ^bd3:
        AIE.dmaBd(<%buf_s : memref<256xi32>, 0, 256>, 0)
        AIE.useLock(%lock_s, Release, 1)
        AIE.nextBd ^end
      ^end:
        AIE.end
    }
  }
