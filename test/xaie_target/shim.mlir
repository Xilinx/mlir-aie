//===- shim.mlir -----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_aie_configure_cores
// CHECK: XAieTile_CoreControl(&(ctx->TileInst[2][1]), XAIE_DISABLE, XAIE_ENABLE);
// CHECK-NOT: XAieTile_CoreControl(&(ctx->TileInst[2][0])
// CHECK: for (int l=0; l<16; l++)
// CHECK:   XAieTile_LockRelease(&(ctx->TileInst[2][1]), l, 0x0, 0);
// CHECK: XAieTile_ShimColumnReset(&(ctx->TileInst[2][0]), XAIE_RESETENABLE);
// CHECK: XAieTile_ShimColumnReset(&(ctx->TileInst[2][0]), XAIE_RESETDISABLE);

// CHECK: mlir_aie_start_cores
// CHECK: XAieTile_CoreControl(&(ctx->TileInst[2][1]), XAIE_ENABLE, XAIE_DISABLE);
// CHECK-NOT: XAieTile_CoreControl(&(ctx->TileInst[2][0])

// CHECK: mlir_aie_configure_dmas
// CHECK: XAieDma_Shim ShimDMAInst_2_0;
// CHECK: XAieDma_ShimInitialize(&(ctx->TileInst[2][0]), &ShimDMAInst_2_0);
// CHECK: XAieDma_ShimBdSetLock(&ShimDMAInst_2_0,  /* bd */ 0,  /* lockID */ 0, XAIE_ENABLE,  /* release */ 1, XAIE_ENABLE,  /* acquire */ 0);
// CHECK: XAieDma_ShimBdSetAxi(&ShimDMAInst_2_0, /* bd */ 0, /* smid */ 0, /* burstlen */ 4, /* QOS */ 0, /* Cache */ 0, /* secure */ XAIE_ENABLE);
// CHECK: XAieDma_ShimBdSetNext(&ShimDMAInst_2_0,  /* bd */ 0,  /* nextbd */ 0);
// CHECK: XAieDma_ShimBdWrite(&ShimDMAInst_2_0,  /* bd */ 0);
// XAieDma_ShimBdSetLock(&ShimDMAInst_2_0,  /* bd */ 1,  /* lockID */ 1, XAIE_ENABLE,  /* release */ 0, XAIE_ENABLE,  /* acquire */ 1);
// CHECK: XAieDma_ShimBdSetAxi(&ShimDMAInst_2_0, /* bd */ 1, /* smid */ 0, /* burstlen */ 4, /* QOS */ 0, /* Cache */ 0, /* secure */ XAIE_ENABLE);
// CHECK: XAieDma_ShimBdSetNext(&ShimDMAInst_2_0,  /* bd */ 1,  /* nextbd */ 1);
// CHECK: XAieDma_ShimBdWrite(&ShimDMAInst_2_0,  /* bd */ 1);
// CHECK: XAieDma_ShimSetStartBd(&ShimDMAInst_2_0, XAIEDMA_SHIM_CHNUM_S2MM0,  /* bd */ 0);
// CHECK: XAieDma_ShimChControl(&ShimDMAInst_2_0, XAIEDMA_TILE_CHNUM_S2MM0, /* PauseStream */ XAIE_DISABLE, /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);
// CHECK: XAieDma_ShimSetStartBd(&ShimDMAInst_2_0, XAIEDMA_SHIM_CHNUM_MM2S0,  /* bd */ 1);
// CHECK: XAieDma_ShimChControl(&ShimDMAInst_2_0, XAIEDMA_TILE_CHNUM_MM2S0, /* PauseStream */ XAIE_DISABLE, /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

// CHECK: mlir_aie_configure_switchboxes
// CHECK: x = 2;
// CHECK: y = 0;
// CHECK: XAieTile_ShimStrmDemuxConfig(&(ctx->TileInst[x][y]),
// CHECK:        XAIETILE_SHIM_STRM_DEM_SOUTH2,
// CHECK:        XAIETILE_SHIM_STRM_DEM_DMA);

module {
 AIE.device(xcvc1902) {
  %buffer = AIE.external_buffer : memref<16 x f32>
  %t21 = AIE.tile(2, 1)
  %t20 = AIE.tile(2, 0)
  %c21 = AIE.core(%t21)  {
    AIE.end
  }
  %s21 = AIE.switchbox(%t21)  {
    AIE.connect<Core : 0, South : 0>
  }
  %s20 = AIE.switchbox(%t20)  {
    AIE.connect<North : 0, South : 2>
  }
  %mux = AIE.shimmux(%t20)  {
    AIE.connect<North : 2, DMA : 0>
  }
  %dma = AIE.shimDMA(%t20)  {
      %lock0 = AIE.lock(%t20, 0)
      %lock1 = AIE.lock(%t20, 1)
   
      AIE.dmaStart(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      AIE.dmaStart(MM2S, 0, ^bd1, ^end)
    ^bd0:
      AIE.useLock(%lock0, Acquire, 0)
      AIE.dmaBd(<%buffer : memref<16 x f32>, 0, 16>, 0)
      AIE.useLock(%lock0, Release, 1)
      AIE.nextBd ^bd0
    ^bd1:
      // AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer : memref<16 x f32>, 0, 4>, 0)
      // AIE.useLock(%lock1, Release, 0)
      AIE.nextBd ^bd1
    ^end:
      AIE.end
  }
  AIE.wire(%s21 : South, %s20 : North)
  AIE.wire(%s20 : South, %mux : North)
  AIE.wire(%mux : DMA, %dma : DMA)
  AIE.wire(%mux : South, %t20 : DMA)
  AIE.wire(%s21 : Core, %c21 : Core)
  AIE.wire(%s21 : Core, %t21 : Core)
 }
}
