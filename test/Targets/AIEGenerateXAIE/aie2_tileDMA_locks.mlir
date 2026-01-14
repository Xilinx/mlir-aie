//===- aie2_tileDMA_locks.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: XAie_DmaDesc [[bd0:.*]];
// CHECK: XAie_DmaDescInit(ctx->XAieDevInst, &([[bd0]]), XAie_TileLoc(7,4))
// CHECK: XAie_DmaSetLock(&([[bd0]]), XAie_LockInit(3,-1),XAie_LockInit(4,1))
// CHECK: XAie_DmaSetAddrLen(&([[bd0]]),  /* addrA */ 0x720,  /* len */ 1024)
// CHECK: XAie_DmaSetNextBd(&([[bd0]]),  /* nextbd */ 1,  /* enableNextBd */ 1)
// CHECK: XAie_DmaEnableBd(&([[bd0]]))
// CHECK: XAie_DmaWriteBd(ctx->XAieDevInst, &([[bd0]]), XAie_TileLoc(7,4),  /* bd */ 0)
// CHECK: XAie_DmaChannelSetStartQueue(ctx->XAieDevInst, XAie_TileLoc(7,4), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */0, /* Repeat */ 1, /* EnToken */ XAIE_DISABLE)
// CHECK: XAie_DmaChannelEnable(ctx->XAieDevInst, XAie_TileLoc(7,4), /* ChNum */ 0, /* dmaDir */ DMA_S2MM)

module @aie_module  {
  aie.device(xcve2802) {
    %t63 = aie.tile(6, 4)
    %t73 = aie.tile(7, 4)
    %t72 = aie.tile(7, 3)
    %t74 = aie.tile(7, 5)

    %buf_e = aie.buffer(%t63) {address = 0 : i32, sym_name = "east" } : memref<256xi32>
    %buf_l = aie.buffer(%t73) {address = 1824 : i32, sym_name = "local" } : memref<256xi32>
    %buf_n = aie.buffer(%t74) {address = 0 : i32, sym_name = "north" } : memref<256xi32>
    %buf_s = aie.buffer(%t72) {address = 0 : i32, sym_name = "south" } : memref<256xi32>

    %lock_e = aie.lock(%t63, 0)
    %lock_l1 = aie.lock(%t73, 3)
    %lock_l2 = aie.lock(%t73, 4)
    %lock_n = aie.lock(%t74, 0)
    %lock_s = aie.lock(%t72, 0)
    
    // Tile DMA
    %m73 = aie.mem(%t73) {
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
