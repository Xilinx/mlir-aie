//===- shim_dma_packet.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK-LABEL: void mlir_aie_configure_dmas(aie_libxaie_ctx_t* ctx) {
// CHECK: XAieDma_Shim ShimDMAInst_7_0;
// CHECK: XAieDma_ShimInitialize(&(ctx->TileInst[7][0]), &ShimDMAInst_7_0);
// CHECK: XAieDma_ShimBdSetLock(&ShimDMAInst_7_0,  /* bd */ 0,  /* lockID */ 0, XAIE_ENABLE,  /* release */ 0, XAIE_ENABLE,  /* acquire */ 1);
// CHECK: XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0, HIGH_ADDR((u64)0x4000), LOW_ADDR((u64)0x4000),  /* len */ 1024 * 4);
// CHECK: XAieDma_ShimBdSetAxi(&ShimDMAInst_7_0, /* bd */ 0, /* smid */ 0, /* burstlen */ 4, /* QOS */ 0, /* Cache */ 0, /* secure */ XAIE_ENABLE);
// CHECK: XAieDma_ShimBdSetNext(&ShimDMAInst_7_0,  /* bd */ 0,  /* nextbd */ 0);
// CHECK: XAieDma_ShimBdSetPkt(&ShimDMAInst_7_0,  /* bd */ 0,  /* en */ 1,  /* type */ 0,  /* id */ 2);
// CHECK: XAieDma_ShimBdWrite(&ShimDMAInst_7_0,  /* bd */ 0);
// CHECK: XAieDma_ShimSetStartBd(&ShimDMAInst_7_0, XAIEDMA_SHIM_CHNUM_MM2S0,  /* bd */ 0);
// CHECK: XAieDma_ShimChControl(&ShimDMAInst_7_0, XAIEDMA_TILE_CHNUM_MM2S0, /* PauseStream */ XAIE_DISABLE, /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

module {
  %buf = AIE.external_buffer 16384 : memref<32x32xi32>

  %tile70 = AIE.tile(7, 0)
  %lock70 = AIE.lock(%tile70, 0)

  %shimdma70 = AIE.shimDMA(%tile70)  {
    AIE.dmaStart(MM2S0, ^bb1, ^bb2)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%lock70, Acquire, 1)
    AIE.dmaBdPacket(0, 2)
    AIE.dmaBd(<%buf : memref<32x32xi32>, 0, 1024>, 0)
    AIE.useLock(%lock70, Release, 0)
    br ^bb1
  ^bb2:  // pred: ^bb0
    AIE.end
  }
}
