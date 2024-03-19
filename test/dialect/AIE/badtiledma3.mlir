//===- badtile.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not %PYTHON aiecc.py %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.use_lock' op can only access a lock in the same tile

module @test {
    %t63 = aie.tile(6, 3)
    %t73 = aie.tile(7, 3)
    %t74 = aie.tile(7, 4)
    %t72 = aie.tile(7, 2)

    %buf_e = aie.buffer(%t63) { sym_name = "east" } : memref<256xi32>
    %buf_l = aie.buffer(%t73) { sym_name = "local" } : memref<256xi32>
    %buf_n = aie.buffer(%t74) { sym_name = "north" } : memref<256xi32>
    %buf_s = aie.buffer(%t72) { sym_name = "south" } : memref<256xi32>

    %lock_e = aie.lock(%t63, 0)
    %lock_l = aie.lock(%t73, 0)
    %lock_n = aie.lock(%t74, 0)
    %lock_s = aie.lock(%t72, 0)

    // Tile DMA
    %m73 = aie.mem(%t73) {
        %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^dma1)
      ^dma1:
        %dstDma = aie.dma_start("S2MM", 0, ^bd2, ^end)
      ^bd0:
        aie.dma_bd(%buf_l : memref<256xi32>) { len = 256 : i32 }
        aie.use_lock(%lock_e, Release, 1)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%buf_l : memref<256xi32>) { len = 256 : i32 }
        aie.use_lock(%lock_l, Release, 1)
        aie.next_bd ^end
      ^bd2:
        aie.dma_bd(%buf_l : memref<256xi32>) { len = 256 : i32 }
        aie.use_lock(%lock_n, Release, 1)
        aie.next_bd ^bd3
      ^bd3:
        aie.dma_bd(%buf_l : memref<256xi32>) { len = 256 : i32 }
        aie.use_lock(%lock_s, Release, 1)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
