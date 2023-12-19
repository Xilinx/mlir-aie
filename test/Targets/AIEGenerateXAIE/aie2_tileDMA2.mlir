//===- aie2_tileDMA.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: XAie_DmaDesc [[bd0:.*]];
// CHECK: __mlir_aie_try(XAie_DmaDescInit(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(7,3)));
// CHECK: __mlir_aie_try(XAie_DmaSetLock(&([[bd0]]), XAie_LockInit(3,-1),XAie_LockInit({{.*}},0)));
// CHECK: [[bd0]].LockDesc.LockRelEn = XAIE_DISABLE;
// CHECK: __mlir_aie_try(XAie_DmaSetAddrLen(&([[bd0]]),  /* addrA */ 0x720,  /* len */ 256 * 4));
// CHECK: __mlir_aie_try(XAie_DmaSetNextBd(&([[bd0]]),  /* nextbd */ 0,  /* enableNextBd */ 0));
// CHECK: __mlir_aie_try(XAie_DmaEnableBd(&([[bd0]])));
// CHECK: __mlir_aie_try(XAie_DmaWriteBd(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(7,3),  /* bd */ 0));
// CHECK: __mlir_aie_try(XAie_DmaChannelPushBdToQueue(&(ctx->DevInst), XAie_TileLoc(7,3), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */0));
// CHECK: __mlir_aie_try(XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(7,3), /* ChNum */ 0, /* dmaDir */ DMA_S2MM));

module @aie_module  {
  AIE.device(xcve2802) {
    %t73 = AIE.tile(7, 3)

    %buf_a_ping = AIE.buffer(%t73) {address = 1824 : i32, sym_name = "a_ping" } : memref<256xi32>

    %lock_a_write = AIE.lock(%t73, 3) { init = 1 : i32 }
    %lock_a_read = AIE.lock(%t73, 4)

    // Tile DMA
    %m73 = AIE.mem(%t73) {
        %srcDma = AIE.dma_start("S2MM", 0, ^bd0, ^end)
      ^bd0:
        // Note: acquire and release are different locks.
        AIE.use_lock(%lock_a_write, AcquireGreaterEqual, 1)
        AIE.dma_bd(<%buf_a_ping : memref<256xi32>, 0, 256>, A)
        // AIE.use_lock(%lock_a_read, Release, 1)
        AIE.next_bd ^end
      ^end:
        AIE.end
    }
 }
}
