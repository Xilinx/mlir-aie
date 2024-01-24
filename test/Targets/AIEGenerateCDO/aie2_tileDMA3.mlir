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
// CHECK: XAie_DmaDescInit(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(2,3));
// CHECK: XAie_DmaSetLock(&([[bd0]]), XAie_LockInit({{.*}},0),XAie_LockInit(4,1));
// CHECK: [[bd0]].LockDesc.LockAcqEn = XAIE_DISABLE;
// CHECK: XAie_DmaSetAddrLen(&([[bd0]]),  /* addrA */ 0x720,  /* len */ 256 * 4);
// CHECK: XAie_DmaSetNextBd(&([[bd0]]),  /* nextbd */ 0,  /* enableNextBd */ 0);
// CHECK: XAie_DmaEnableBd(&([[bd0]]));
// CHECK: XAie_DmaWriteBd(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(2,3),  /* bd */ 0);
// CHECK: XAie_DmaChannelSetStartQueue(&(ctx->DevInst), XAie_TileLoc(2,3), /* ChNum */ 0, /* dmaDir */ DMA_S2MM, /* BdNum */ 0, /* Repeat */ 1, /* EnToken */ XAIE_DISABLE);
// CHECK: XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(2,3), /* ChNum */ 0, /* dmaDir */ DMA_S2MM);

module @aie_module  {
  aie.device(ipu) {
    %t23 = aie.tile(2, 3)

    %buf_a_ping = aie.buffer(%t23) {address = 1824 : i32, sym_name = "a_ping" } : memref<256xi32>

    %lock_a_write = aie.lock(%t23, 3) { init = 1 : i32 }
    %lock_a_read = aie.lock(%t23, 4)

    // Tile DMA
    %m23 = aie.mem(%t23) {
        %srcDma = aie.dmaStart("S2MM", 0, ^bd0, ^end)
      ^bd0:
        // Note: acquire and release are different locks.
        //aie.useLock(%lock_a_write, AcquireGreaterEqual, 1)
        aie.dmaBd(%buf_a_ping : memref<256xi32>, 0, 256)
        aie.useLock(%lock_a_read, Release, 1)
        aie.nextBd ^end
      ^end:
        aie.end
    }
 }
}
