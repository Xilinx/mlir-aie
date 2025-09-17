//===- test_xaie1.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: XAie_DmaDesc dma_tile33_bd0;
// CHECK: __mlir_aie_try(XAie_DmaDescInit(ctx->XAieDevInst, &(dma_tile33_bd0), XAie_TileLoc(3,3)));
// CHECK: __mlir_aie_try(XAie_DmaSetLock(&(dma_tile33_bd0), XAie_LockInit(0,0),XAie_LockInit(0,1)));
// CHECK: __mlir_aie_try(XAie_DmaSetAddrLen(&(dma_tile33_bd0), {{.*}} 0x1400, {{.*}} 1024));
// CHECK: __mlir_aie_try(XAie_DmaSetNextBd(&(dma_tile33_bd0), {{.*}} 0, {{.*}} 0));
// CHECK: __mlir_aie_try(XAie_DmaEnableBd(&(dma_tile33_bd0)));
// CHECK: __mlir_aie_try(XAie_DmaWriteBd(ctx->XAieDevInst, &(dma_tile33_bd0), XAie_TileLoc(3,3), {{.*}} 0));
// CHECK: __mlir_aie_try(XAie_DmaChannelPushBdToQueue(ctx->XAieDevInst, XAie_TileLoc(3,3), {{.*}}0, {{.*}} DMA_MM2S, {{.*}}0));
// CHECK: __mlir_aie_try(XAie_DmaChannelEnable(ctx->XAieDevInst, XAie_TileLoc(3,3), {{.*}} 0, /* dmaDir */ DMA_MM2S));

module @test_xaie1 {
 aie.device(xcvc1902) {
  %t33 = aie.tile(3, 3)

  %buf33_0 = aie.buffer(%t33) {address = 4096 : i32, sym_name = "buf33_0"} : memref<256xi32>
  %buf33_1 = aie.buffer(%t33) {address = 5120 : i32, sym_name = "buf33_1"} : memref<256xi32>
  %l33_0 = aie.lock(%t33, 0)

  %m33 = aie.mem(%t33) {
      %srcDma = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%l33_0, Acquire, 0)
      aie.dma_bd(%buf33_1 : memref<256xi32>, 0, 256)
      aie.use_lock(%l33_0, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

 }
}
