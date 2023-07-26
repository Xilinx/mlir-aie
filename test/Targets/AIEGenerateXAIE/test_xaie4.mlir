//===- test_xaie4.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// Test shared BD list.
// CHECK: XAie_DmaDesc [[bd0:.*]];
// CHECK: __mlir_aie_try(XAie_DmaDescInit(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(3,3)));
// CHECK: __mlir_aie_try(XAie_DmaSetLock(&([[bd0]]), XAie_LockInit(0,1),XAie_LockInit(0,0)));
// CHECK: __mlir_aie_try(XAie_DmaSetAddrLen(&([[bd0]]), {{.*}}0x1000, {{.*}}256 * 4));
// CHECK: __mlir_aie_try(XAie_DmaSetNextBd(&([[bd0]]), {{.*}}0, {{.*}}0));
// CHECK: __mlir_aie_try(XAie_DmaEnableBd(&([[bd0]])));
// CHECK: __mlir_aie_try(XAie_DmaWriteBd(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(3,3), {{.*}}0));
// CHECK: XAie_DmaDesc [[bd1:.*]];
// CHECK: __mlir_aie_try(XAie_DmaDescInit(&(ctx->DevInst), &([[bd1]]), XAie_TileLoc(3,3)));
// CHECK: __mlir_aie_try(XAie_DmaSetLock(&([[bd1]]), XAie_LockInit(1,1),XAie_LockInit(1,0)));
// CHECK: __mlir_aie_try(XAie_DmaSetAddrLen(&([[bd1]]), {{.*}}0x1400, {{.*}}256 * 4));
// CHECK: __mlir_aie_try(XAie_DmaSetNextBd(&([[bd1]]), {{.*}}0, {{.*}}0));
// CHECK: __mlir_aie_try(XAie_DmaEnableBd(&([[bd1]])));
// CHECK: __mlir_aie_try(XAie_DmaWriteBd(&(ctx->DevInst), &([[bd1]]), XAie_TileLoc(3,3), {{.*}}1));
// CHECK: __mlir_aie_try(XAie_DmaChannelPushBdToQueue(&(ctx->DevInst), XAie_TileLoc(3,3), {{.*}}0, {{.*}}DMA_MM2S, {{.*}}0));
// CHECK: __mlir_aie_try(XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(3,3), {{.*}}0, {{.*}}DMA_MM2S));
// CHECK: __mlir_aie_try(XAie_DmaChannelPushBdToQueue(&(ctx->DevInst), XAie_TileLoc(3,3), {{.*}}0, {{.*}}DMA_S2MM, {{.*}}1));
// CHECK: __mlir_aie_try(XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(3,3), {{.*}}0, {{.*}}DMA_S2MM));

module @test_xaie3 {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)
  %t44 = AIE.tile(4, 4)

  %buf33_0 = AIE.buffer(%t33) {address = 0x1000, sym_name = "buf33_0" } : memref<256xi32>
  %buf33_1 = AIE.buffer(%t33) {address = 0x1400, sym_name = "buf33_1" } : memref<256xi32>

  %l33_0 = AIE.lock(%t33, 0)
  %l33_1 = AIE.lock(%t33, 1)

  %m33 = AIE.mem(%t33) {
      %srcDma = AIE.dmaStart(MM2S, 0, ^bd0, ^dma0)
    ^dma0:
      %destDma = AIE.dmaStart(S2MM, 0, ^bd1, ^end)
    ^bd0:
      AIE.useLock(%l33_0, Acquire, 1)
      AIE.dmaBd(<%buf33_0 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l33_0, Release, 0)
      AIE.nextBd ^end
    ^bd1:
      AIE.useLock(%l33_1, Acquire, 1)
      AIE.dmaBd(<%buf33_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l33_1, Release, 0)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }
 }
}
