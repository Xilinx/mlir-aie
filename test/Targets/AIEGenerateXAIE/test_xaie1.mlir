//===- test_xaie1.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: XAie_DmaDesc dma_tile33_bd0;
// CHECK: XAie_DmaDescInit(&(ctx->DevInst), &(dma_tile33_bd0), XAie_TileLoc(3,3));
// CHECK: XAie_DmaSetLock(&(dma_tile33_bd0), XAie_LockInit(0,0),XAie_LockInit(0,1));
// CHECK: XAie_DmaSetAddrLen(&(dma_tile33_bd0), {{.*}} 0x1400, {{.*}} 256 * 4);
// CHECK: XAie_DmaSetNextBd(&(dma_tile33_bd0), {{.*}} 0, {{.*}} 0);
// CHECK: XAie_DmaEnableBd(&(dma_tile33_bd0));
// CHECK: XAie_DmaWriteBd(&(ctx->DevInst), &(dma_tile33_bd0), XAie_TileLoc(3,3), {{.*}} 0);
// CHECK: XAie_DmaChannelPushBdToQueue(&(ctx->DevInst), XAie_TileLoc(3,3), {{.*}}0, {{.*}} DMA_MM2S, {{.*}}0);
// CHECK: XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(3,3), {{.*}} 0, /* dmaDir */ DMA_MM2S);

module @test_xaie1 {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)

  %buf33_0 = AIE.buffer(%t33) {address = 4096 : i32, sym_name = "buf33_0"} : memref<256xi32>
  %buf33_1 = AIE.buffer(%t33) {address = 5120 : i32, sym_name = "buf33_1"} : memref<256xi32>
  %l33_0 = AIE.lock(%t33, 0)

  %m33 = AIE.mem(%t33) {
      %srcDma = AIE.dmaStart(MM2S, 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l33_0, Acquire, 0)
      AIE.dmaBd(<%buf33_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l33_0, Release, 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

 }
}
