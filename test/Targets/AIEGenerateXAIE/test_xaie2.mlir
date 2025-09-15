//===- test_xaie2.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// Test S2MM, BD chaining.
// CHECK: XAie_DmaDesc [[bd0:.*]];
// CHECK: __mlir_aie_try(XAie_DmaDescInit(ctx->XAieDevInst, &([[bd0]]), XAie_TileLoc(3,3)));
// CHECK: __mlir_aie_try(XAie_DmaSetLock(&([[bd0]]), XAie_LockInit(0,0),XAie_LockInit(0,1)));
// CHECK: __mlir_aie_try(XAie_DmaSetAddrLen(&([[bd0]]), {{.*}}0x1000, {{.*}}1024));
// CHECK: __mlir_aie_try(XAie_DmaSetNextBd(&([[bd0]]), {{.*}}1, {{.*}}1));
// CHECK: __mlir_aie_try(XAie_DmaEnableBd(&([[bd0]])));
// CHECK: __mlir_aie_try(XAie_DmaWriteBd(ctx->XAieDevInst, &([[bd0]]), XAie_TileLoc(3,3), {{.*}}0));
// CHECK: XAie_DmaDesc [[bd1:.*]];
// CHECK: __mlir_aie_try(XAie_DmaDescInit(ctx->XAieDevInst, &([[bd1]]), XAie_TileLoc(3,3)));
// CHECK: __mlir_aie_try(XAie_DmaSetLock(&([[bd1]]), XAie_LockInit(0,0),XAie_LockInit(0,1)));
// CHECK: __mlir_aie_try(XAie_DmaSetAddrLen(&([[bd1]]), {{.*}}0x1400, {{.*}}16));
// CHECK: __mlir_aie_try(XAie_DmaSetNextBd(&([[bd1]]), {{.*}}0, {{.*}}1));
// CHECK: __mlir_aie_try(XAie_DmaEnableBd(&([[bd1]])));
// CHECK: __mlir_aie_try(XAie_DmaWriteBd(ctx->XAieDevInst, &([[bd1]]), XAie_TileLoc(3,3), {{.*}}1));
// CHECK: __mlir_aie_try(XAie_DmaChannelPushBdToQueue(ctx->XAieDevInst, XAie_TileLoc(3,3), {{.*}}0, {{.*}}DMA_S2MM, {{.*}}0));
// CHECK: __mlir_aie_try(XAie_DmaChannelEnable(ctx->XAieDevInst, XAie_TileLoc(3,3), {{.*}}0, {{.*}}DMA_S2MM));

module @test_xaie2 {
 aie.device(xcvc1902) {
  %t33 = aie.tile(3, 3)

  %buf33_0 = aie.buffer(%t33) { address = 0x1000 : i32, sym_name = "buff33_0" }: memref<256xi32>
  %buf33_1 = aie.buffer(%t33) { address = 0x1400 : i32, sym_name = "buff33_1" }: memref<16xi32>

  %l33_0 = aie.lock(%t33, 0)
  %l33_1 = aie.lock(%t33, 1)

  %m33 = aie.mem(%t33) {
      %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%l33_0, Acquire, 0)
      aie.dma_bd(%buf33_0 : memref<256xi32>, 0, 256)
      aie.use_lock(%l33_0, Release, 1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%l33_0, Acquire, 0)
      aie.dma_bd(%buf33_1 : memref<16xi32>, 0, 4)
      aie.use_lock(%l33_0, Release, 1)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }
 }
}
