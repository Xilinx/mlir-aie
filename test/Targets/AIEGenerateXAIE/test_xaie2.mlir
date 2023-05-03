//===- test_xaie2.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// Test S2MM, BD chaining.
// CHECK: XAie_DmaDesc [[bd0:.*]];
// CHECK: XAie_DmaDescInit(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(3,3));
// CHECK: XAie_DmaSetLock(&([[bd0]]), XAie_LockInit(0,0),XAie_LockInit(0,1));
// CHECK: XAie_DmaSetAddrLen(&([[bd0]]), {{.*}}0x1000, {{.*}}256 * 4);
// CHECK: XAie_DmaSetNextBd(&([[bd0]]), {{.*}}1, {{.*}}1);
// CHECK: XAie_DmaEnableBd(&([[bd0]]));
// CHECK: XAie_DmaWriteBd(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(3,3), {{.*}}0);
// CHECK: XAie_DmaDesc [[bd1:.*]];
// CHECK: XAie_DmaDescInit(&(ctx->DevInst), &([[bd1]]), XAie_TileLoc(3,3));
// CHECK: XAie_DmaSetLock(&([[bd1]]), XAie_LockInit(0,0),XAie_LockInit(0,1));
// CHECK: XAie_DmaSetAddrLen(&([[bd1]]), {{.*}}0x1400, {{.*}}4 * 4);
// CHECK: XAie_DmaSetNextBd(&([[bd1]]), {{.*}}0, {{.*}}1);
// CHECK: XAie_DmaEnableBd(&([[bd1]]));
// CHECK: XAie_DmaWriteBd(&(ctx->DevInst), &([[bd1]]), XAie_TileLoc(3,3), {{.*}}1);
// CHECK: XAie_DmaChannelPushBdToQueue(&(ctx->DevInst), XAie_TileLoc(3,3), {{.*}}0, {{.*}}DMA_S2MM, {{.*}}0);
// CHECK: XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(3,3), {{.*}}0, {{.*}}DMA_S2MM);

module @test_xaie2 {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)

  %buf33_0 = AIE.buffer(%t33) { address = 0x1000, sym_name = "buff33_0" }: memref<256xi32>
  %buf33_1 = AIE.buffer(%t33) { address = 0x1400, sym_name = "buff33_1" }: memref<16xi32>

  %l33_0 = AIE.lock(%t33, 0)
  %l33_1 = AIE.lock(%t33, 1)

  %m33 = AIE.mem(%t33) {
      %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l33_0, Acquire, 0)
      AIE.dmaBd(<%buf33_0 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l33_0, Release, 1)
      AIE.nextBd ^bd1
    ^bd1:
      AIE.useLock(%l33_0, Acquire, 0)
      AIE.dmaBd(<%buf33_1 : memref<16xi32>, 0, 4>, 0)
      AIE.useLock(%l33_0, Release, 1)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }
 }
}
