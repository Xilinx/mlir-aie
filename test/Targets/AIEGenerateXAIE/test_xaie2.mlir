//===- test_xaie2.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
      %c0_ul1 = arith.constant 0 : i32
      aie.use_lock(%l33_0, Acquire, %c0_ul1)
      aie.dma_bd(%buf33_0 : memref<256xi32> offset = 0 len = 256)
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(%l33_0, Release, %c1_ul2)
      aie.next_bd ^bd1
    ^bd1:
      %c0_ul3 = arith.constant 0 : i32
      aie.use_lock(%l33_0, Acquire, %c0_ul3)
      aie.dma_bd(%buf33_1 : memref<16xi32> offset = 0 len = 4)
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(%l33_0, Release, %c1_ul4)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }
 }
}
