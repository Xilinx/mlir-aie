//===- memTileDMA2.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: XAie_DmaDesc [[bd0:.*]];
// CHECK: XAie_DmaDescInit(ctx->XAieDevInst, &([[bd0]]), XAie_TileLoc(2,1))
// CHECK: XAie_DmaSetLock(&([[bd0]]), XAie_LockInit(0,0),XAie_LockInit(0,1))
// CHECK: [[bd0]].LockDesc.LockAcqEn = XAIE_DISABLE;
// CHECK: XAie_DmaSetAddrLen(&([[bd0]]), /* addrA */ 0x0,  /* len */ 64)
// CHECK: XAie_DmaSetNextBd(&([[bd0]]),  /* nextbd */ 1,  /* enableNextBd */ 1)
// CHECK: XAie_DmaEnableBd(&([[bd0]]))
// CHECK: XAie_DmaWriteBd(ctx->XAieDevInst, &([[bd0]]), XAie_TileLoc(2,1),  /* bd */ 0)

// CHECK: XAie_DmaDesc [[bd1:.*]];
// CHECK: XAie_DmaDescInit(ctx->XAieDevInst, &([[bd1]]), XAie_TileLoc(2,1))
// CHECK: XAie_DmaSetLock(&([[bd1]]), XAie_LockInit(0,0),XAie_LockInit(64,1))
// CHECK: [[bd1]].LockDesc.LockAcqEn = XAIE_DISABLE;
// CHECK: XAie_DmaSetAddrLen(&([[bd1]]), /* addrA */ 0x80000,  /* len */ 64)
// CHECK: XAie_DmaSetNextBd(&([[bd1]]),  /* nextbd */ 2,  /* enableNextBd */ 1)
// CHECK: XAie_DmaEnableBd(&([[bd1]]))
// CHECK: XAie_DmaWriteBd(ctx->XAieDevInst, &([[bd1]]), XAie_TileLoc(2,1),  /* bd */ 1)

// CHECK: XAie_DmaDesc [[bd2:.*]];
// CHECK: XAie_DmaDescInit(ctx->XAieDevInst, &([[bd2]]), XAie_TileLoc(2,1))
// CHECK: XAie_DmaSetLock(&([[bd2]]), XAie_LockInit(0,0),XAie_LockInit(128,1))
// CHECK: [[bd2]].LockDesc.LockAcqEn = XAIE_DISABLE;
// CHECK: XAie_DmaSetAddrLen(&([[bd2]]), /* addrA */ 0x100000,  /* len */ 64)
// CHECK: XAie_DmaSetNextBd(&([[bd2]]),  /* nextbd */ 0,  /* enableNextBd */ 0)
// CHECK: XAie_DmaEnableBd(&([[bd2]]))
// CHECK: XAie_DmaWriteBd(ctx->XAieDevInst, &([[bd2]]), XAie_TileLoc(2,1),  /* bd */ 2)

module @aie_module  {
 aie.device(xcve2802) {
  %t00 = aie.tile(1, 1)
  %t01 = aie.tile(2, 1)
  %t02 = aie.tile(3, 1)
  aie.buffer @west(%t00) { address = 0 : i32} : memref<16xi32>
  aie.buffer @local(%t01) { address = 0 : i32} : memref<16xi32>
  aie.buffer @east(%t02) { address = 0 : i32} : memref<16xi32>

  aie.lock @lock_w(%t00, 0)
  aie.lock @lock_l(%t01, 0)
  aie.lock @lock_e(%t02, 0)

  %m01 = aie.memtile_dma(%t01) {
      %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      %buf_w = aie.get_buffer @west : memref<16xi32>
      aie.dma_bd(%buf_w : memref<16xi32> offset = 0 len = 16)
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(@lock_w, "Release", %c1_ul1)
      aie.next_bd ^bd1
    ^bd1:
      %buf_l = aie.get_buffer @local : memref<16xi32>
      aie.dma_bd(%buf_l : memref<16xi32> offset = 0 len = 16)
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(@lock_l, "Release", %c1_ul2)
      aie.next_bd ^bd2
    ^bd2:
      %buf_e = aie.get_buffer @east : memref<16xi32>
      aie.dma_bd(%buf_e : memref<16xi32> offset = 0 len = 16)
      %c1_ul3 = arith.constant 1 : i32
      aie.use_lock(@lock_e, "Release", %c1_ul3)
      aie.next_bd ^end
    ^end:
      aie.end
  }
 }
}
