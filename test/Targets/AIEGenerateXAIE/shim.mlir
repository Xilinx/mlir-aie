//===- shim.mlir -----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_aie_configure_shimdma_20
// CHECK: XAie_DmaDesc [[bd0:.*]];
// CHECK: XAie_DmaDescInit(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(2,0));
// CHECK: XAie_DmaSetLock(&([[bd0]]), XAie_LockInit(0,0),XAie_LockInit(0,1));
// CHECK: XAie_DmaSetAddrLen(&([[bd0]]), {{.*}} mlir_aie_external_get_addr_myBuffer_20_0(), {{.*}} 16 * 4);
// CHECK: XAie_DmaSetAxi(&([[bd0]]), {{.*}} 0, {{.*}} 4, {{.*}} 0, {{.*}} 0, {{.*}} XAIE_ENABLE);
// CHECK: XAie_DmaSetNextBd(&([[bd0]]), {{.*}} 0, {{.*}} 1);
// CHECK: XAie_DmaEnableBd(&([[bd0]]));
// CHECK: XAie_DmaWriteBd(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(2,0), {{.*}} 0);
// CHECK: XAie_DmaDesc [[bd1:.*]];
// CHECK: XAie_DmaDescInit(&(ctx->DevInst), &([[bd1]]), XAie_TileLoc(2,0));
// CHECK: XAie_DmaSetAddrLen(&([[bd1]]), {{.*}} mlir_aie_external_get_addr_myBuffer_20_1(), {{.*}} 4 * 4);
// CHECK: XAie_DmaSetAxi(&([[bd1]]), {{.*}} 0, {{.*}} 4, {{.*}} 0, {{.*}} 0, {{.*}} XAIE_ENABLE);
// CHECK: XAie_DmaSetNextBd(&([[bd1]]), {{.*}} 1, {{.*}} 1);
// CHECK: XAie_DmaEnableBd(&([[bd1]]));
// CHECK: XAie_DmaWriteBd(&(ctx->DevInst), &([[bd1]]), XAie_TileLoc(2,0), {{.*}} 1);
// CHECK: XAie_DmaChannelPushBdToQueue(&(ctx->DevInst), XAie_TileLoc(2,0), {{.*}}0, {{.*}} DMA_S2MM, {{.*}}0);
// CHECK: XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(2,0), {{.*}} 0, {{.*}} DMA_S2MM);
// CHECK: XAie_DmaChannelPushBdToQueue(&(ctx->DevInst), XAie_TileLoc(2,0), {{.*}}0, {{.*}} DMA_MM2S, {{.*}}1);
// CHECK: XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(2,0), {{.*}} 0, {{.*}} DMA_MM2S);

// CHECK: mlir_aie_configure_switchboxes
// CHECK: x = 2;
// CHECK: y = 0;
// CHECK: XAie_StrmConnCctEnable(&(ctx->DevInst), XAie_TileLoc(x,y), NORTH, 0, SOUTH, 2);
// CHECK: XAie_EnableAieToShimDmaStrmPort(&(ctx->DevInst), XAie_TileLoc(x,y), 2);


module {
 AIE.device(xcvc1902) {
  %buffer = AIE.external_buffer { sym_name = "buf" } : memref<16 x f32>
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
