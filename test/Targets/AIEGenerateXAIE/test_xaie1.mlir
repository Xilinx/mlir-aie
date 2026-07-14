//===- test_xaie1.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
      %c0_ul1 = arith.constant 0 : i32
      aie.use_lock(%l33_0, Acquire, %c0_ul1)
      aie.dma_bd(%buf33_1 : memref<256xi32> offset = 0 len = 256)
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(%l33_0, Release, %c1_ul2)
      aie.next_bd ^end
    ^end:
      aie.end
  }

 }
}
