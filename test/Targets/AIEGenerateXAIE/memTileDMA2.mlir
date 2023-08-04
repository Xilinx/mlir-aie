//===- memTileDMA.mlir ------------------------------------------*- MLIR -*-===//
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
// CHECK: XAie_DmaDescInit(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(2,1))
// CHECK: XAie_DmaSetLock(&([[bd0]]), XAie_LockInit(0,0),XAie_LockInit(0,1))
// CHECK: [[bd0]].LockDesc.LockAcqEn = XAIE_DISABLE;
// CHECK: XAie_DmaSetAddrLen(&([[bd0]]), /* addrA */ 0x0,  /* len */ 16 * 4)
// CHECK: XAie_DmaSetNextBd(&([[bd0]]),  /* nextbd */ 1,  /* enableNextBd */ 1)
// CHECK: XAie_DmaEnableBd(&([[bd0]]))
// CHECK: XAie_DmaWriteBd(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(2,1),  /* bd */ 0)

// CHECK: XAie_DmaDesc [[bd1:.*]];
// CHECK: XAie_DmaDescInit(&(ctx->DevInst), &([[bd1]]), XAie_TileLoc(2,1))
// CHECK: XAie_DmaSetLock(&([[bd1]]), XAie_LockInit(0,0),XAie_LockInit(64,1))
// CHECK: [[bd1]].LockDesc.LockAcqEn = XAIE_DISABLE;
// CHECK: XAie_DmaSetAddrLen(&([[bd1]]), /* addrA */ 0x80000,  /* len */ 16 * 4)
// CHECK: XAie_DmaSetNextBd(&([[bd1]]),  /* nextbd */ 2,  /* enableNextBd */ 1)
// CHECK: XAie_DmaEnableBd(&([[bd1]]))
// CHECK: XAie_DmaWriteBd(&(ctx->DevInst), &([[bd1]]), XAie_TileLoc(2,1),  /* bd */ 1)

// CHECK: XAie_DmaDesc [[bd2:.*]];
// CHECK: XAie_DmaDescInit(&(ctx->DevInst), &([[bd2]]), XAie_TileLoc(2,1))
// CHECK: XAie_DmaSetLock(&([[bd2]]), XAie_LockInit(0,0),XAie_LockInit(128,1))
// CHECK: [[bd2]].LockDesc.LockAcqEn = XAIE_DISABLE;
// CHECK: XAie_DmaSetAddrLen(&([[bd2]]), /* addrA */ 0x100000,  /* len */ 16 * 4)
// CHECK: XAie_DmaSetNextBd(&([[bd2]]),  /* nextbd */ 0,  /* enableNextBd */ 0)
// CHECK: XAie_DmaEnableBd(&([[bd2]]))
// CHECK: XAie_DmaWriteBd(&(ctx->DevInst), &([[bd2]]), XAie_TileLoc(2,1),  /* bd */ 2)

module @aie_module  {
 AIE.device(xcve2802) {
  %t00 = AIE.tile(1, 1)
  %t01 = AIE.tile(2, 1)
  %t02 = AIE.tile(3, 1)
  %buf_w = AIE.buffer(%t00) { address = 0 : i32, sym_name = "west" } : memref<16xi32>
  %buf_l = AIE.buffer(%t01) { address = 0 : i32, sym_name = "local" } : memref<16xi32>
  %buf_e = AIE.buffer(%t02) { address = 0 : i32, sym_name = "east" } : memref<16xi32>

  %lock_w = AIE.lock(%t00, 0)
  %lock_l = AIE.lock(%t01, 0)
  %lock_e = AIE.lock(%t02, 0)

  %m01 = AIE.memTileDMA(%t01) {
      %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
    ^bd0:
      AIE.dmaBd(<%buf_w : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%lock_w, "Release", 1)
      AIE.nextBd ^bd1
    ^bd1:
      AIE.dmaBd(<%buf_l : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%lock_l, "Release", 1)
      AIE.nextBd ^bd2
    ^bd2:
      AIE.dmaBd(<%buf_e : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%lock_e, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }
 }
}
