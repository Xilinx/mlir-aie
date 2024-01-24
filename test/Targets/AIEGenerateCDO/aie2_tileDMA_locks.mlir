//===- aie2_tileDMA.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-cdo %s | FileCheck %s

// CHECK: XAie_DmaDesc [[bd0:.*]];
// CHECK: XAie_DmaDescInit(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(2,4))
// CHECK: XAie_DmaSetLock(&([[bd0]]), XAie_LockInit(3,-1),XAie_LockInit(4,1))
// CHECK: XAie_DmaSetAddrLen(&([[bd0]]),  /* addrA */ 0x720,  /* len */ 251 * 4)
// CHECK: XAie_DmaSetNextBd(&([[bd0]]),  /* nextbd */ 1,  /* enableNextBd */ 1)
// CHECK: XAie_DmaEnableBd(&([[bd0]]))
// CHECK: XAie_DmaWriteBd(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(2,4),  /* bd */ 0)
// CHECK: XAie_DmaChannelSetStartQueue(&(ctx->DevInst), XAie_TileLoc(7,4), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */0, /* Repeat */ 1, /* EnToken */ XAIE_DISABLE)
// CHECK: XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(2,4), /* ChNum */ 0, /* dmaDir */ DMA_S2MM)

module @aie_module  {
  aie.device(ipu) {
    %t13 = aie.tile(1, 4)
    %t23 = aie.tile(2, 4)
    %t22 = aie.tile(2, 3)
    %t24 = aie.tile(2, 5)

    %buf_e = aie.buffer(%t13) {address = 0 : i32, sym_name = "east" } : memref<256xi32>
    %buf_l = aie.buffer(%t23) {address = 1824 : i32, sym_name = "local" } : memref<256xi32>
    %buf_n = aie.buffer(%t24) {address = 0 : i32, sym_name = "north" } : memref<256xi32>
    %buf_s = aie.buffer(%t22) {address = 0 : i32, sym_name = "south" } : memref<256xi32>

    %lock_e = aie.lock(%t13, 0)
    %lock_l1 = aie.lock(%t23, 3)
    %lock_l2 = aie.lock(%t23, 4)
    %lock_n = aie.lock(%t24, 0)
    %lock_s = aie.lock(%t22, 0)
    
    // Tile DMA
    %m23 = aie.mem(%t23) {
        %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^end)
      ^bd0:
        aie.use_lock(%lock_l1, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_l : memref<256xi32>, 0, 256)
        aie.use_lock(%lock_l2, Release, 1)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%buf_l : memref<256xi32>, 0, 256)
        aie.use_lock(%lock_l1, Release, 1)
        aie.next_bd ^bd2
      ^bd2:
        aie.dma_bd(%buf_l : memref<256xi32>, 0, 256)
        aie.use_lock(%lock_l1, Release, 1)
        aie.next_bd ^bd3
      ^bd3:
        aie.dma_bd(%buf_l : memref<256xi32>, 0, 256)
        aie.use_lock(%lock_l1, Release, 1)
        aie.next_bd ^end
      ^end:
        aie.end
    }
 }
}
