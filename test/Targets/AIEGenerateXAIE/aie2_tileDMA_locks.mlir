//===- aie2_tileDMA_locks.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

    aie.buffer @east(%t63) {address = 0 : i32} : memref<256xi32>
    aie.buffer @local(%t73) {address = 1824 : i32} : memref<256xi32>
    aie.buffer @north(%t74) {address = 0 : i32} : memref<256xi32>
    aie.buffer @south(%t72) {address = 0 : i32} : memref<256xi32>

    aie.lock @lock_e(%t63, 0)
    aie.lock @lock_l1(%t73, 3)
    aie.lock @lock_l2(%t73, 4)
    aie.lock @lock_n(%t74, 0)
    aie.lock @lock_s(%t72, 0)

    // Tile DMA
    %m73 = aie.mem(%t73) {
        %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^end)
      ^bd0:
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(@lock_l1, AcquireGreaterEqual, %c1_ul1)
        %buf_l = aie.get_buffer @local : memref<256xi32>
        aie.dma_bd(%buf_l : memref<256xi32> offset = 0 len = 256)
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(@lock_l2, Release, %c1_ul2)
        aie.next_bd ^bd1
      ^bd1:
        %buf_l_1 = aie.get_buffer @local : memref<256xi32>
        aie.dma_bd(%buf_l_1 : memref<256xi32> offset = 0 len = 256)
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(@lock_l1, Release, %c1_ul3)
        aie.next_bd ^bd2
      ^bd2:
        %buf_l_2 = aie.get_buffer @local : memref<256xi32>
        aie.dma_bd(%buf_l_2 : memref<256xi32> offset = 0 len = 256)
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(@lock_l1, Release, %c1_ul4)
        aie.next_bd ^bd3
      ^bd3:
        %buf_l_3 = aie.get_buffer @local : memref<256xi32>
        aie.dma_bd(%buf_l_3 : memref<256xi32> offset = 0 len = 256)
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(@lock_l1, Release, %c1_ul5)
        aie.next_bd ^end
      ^end:
        aie.end
    }
 }
}
