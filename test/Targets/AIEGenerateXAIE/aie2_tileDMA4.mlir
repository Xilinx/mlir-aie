//===- aie2_tileDMA4.mlir --------------------------------------*- MLIR -*-===//
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
// CHECK: __mlir_aie_try(XAie_DmaDescInit(ctx->XAieDevInst, &([[bd0]]), XAie_TileLoc(7,3)));
// CHECK: __mlir_aie_try(XAie_DmaSetLock(&([[bd0]]), XAie_LockInit(3,-1),XAie_LockInit(4,1)));
// CHECK: __mlir_aie_try(XAie_DmaSetAddrLen(&([[bd0]]),  /* addrA */ 0x720,  /* len */ 1024));
// CHECK: __mlir_aie_try(XAie_DmaSetNextBd(&([[bd0]]),  /* nextbd */ 0,  /* enableNextBd */ 0));
// CHECK: __mlir_aie_try(XAie_DmaEnableBd(&([[bd0]])));
// CHECK: __mlir_aie_try(XAie_DmaWriteBd(ctx->XAieDevInst, &([[bd0]]), XAie_TileLoc(7,3),  /* bd */ 0));
// CHECK: __mlir_aie_try(XAie_DmaChannelSetStartQueue(ctx->XAieDevInst, XAie_TileLoc(7,3), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */0, /* Repeat */ 4, /* EnToken */ XAIE_DISABLE));
// CHECK: __mlir_aie_try(XAie_DmaChannelEnable(ctx->XAieDevInst, XAie_TileLoc(7,3), /* ChNum */ 0, /* dmaDir */ DMA_S2MM));

 module @aie_module  {
   aie.device(xcve2802) {
     %t73 = aie.tile(7, 3)

     %buf_a_ping = aie.buffer(%t73) {address = 1824 : i32, sym_name = "a_ping" } : memref<256xi32>

     %lock_a_write = aie.lock(%t73, 3) { init = 1 : i32 }
     %lock_a_read = aie.lock(%t73, 4)

     // Tile DMA
     %m73 = aie.mem(%t73) {
         %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^end, repeat_count = 3)
       ^bd0:
         // Note: acquire and release are different locks.
         aie.use_lock(%lock_a_write, AcquireGreaterEqual, 1)
         aie.dma_bd(%buf_a_ping : memref<256xi32>, 0, 256)
         aie.use_lock(%lock_a_read, Release, 1)
         aie.next_bd ^end
       ^end:
         aie.end
     }
  }
 }