//===- shim.mlir -----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_aie_configure_shimdma_20
// CHECK: XAie_DmaDesc [[bd0:.*]];
// CHECK: __mlir_aie_try(XAie_DmaDescInit(ctx->XAieDevInst, &([[bd0]]), XAie_TileLoc(2,0)));
// CHECK: __mlir_aie_try(XAie_DmaSetLock(&([[bd0]]), XAie_LockInit(0,0),XAie_LockInit(0,1)));
// CHECK: __mlir_aie_try(XAie_DmaSetAddrLen(&([[bd0]]), {{.*}} mlir_aie_external_get_addr_myBuffer_20_0(), {{.*}} 64));
// CHECK: __mlir_aie_try(XAie_DmaSetAxi(&([[bd0]]), {{.*}} 0, {{.*}} 4, {{.*}} 0, {{.*}} 0, {{.*}} XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAie_DmaSetNextBd(&([[bd0]]), {{.*}} 0, {{.*}} 1));
// CHECK: __mlir_aie_try(XAie_DmaEnableBd(&([[bd0]])));
// CHECK: __mlir_aie_try(XAie_DmaWriteBd(ctx->XAieDevInst, &([[bd0]]), XAie_TileLoc(2,0), {{.*}} 0));
// CHECK: XAie_DmaDesc [[bd1:.*]];
// CHECK: __mlir_aie_try(XAie_DmaDescInit(ctx->XAieDevInst, &([[bd1]]), XAie_TileLoc(2,0)));
// CHECK: __mlir_aie_try(XAie_DmaSetAddrLen(&([[bd1]]), {{.*}} mlir_aie_external_get_addr_myBuffer_20_1(), {{.*}} 16));
// CHECK: __mlir_aie_try(XAie_DmaSetAxi(&([[bd1]]), {{.*}} 0, {{.*}} 4, {{.*}} 0, {{.*}} 0, {{.*}} XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAie_DmaSetNextBd(&([[bd1]]), {{.*}} 1, {{.*}} 1));
// CHECK: __mlir_aie_try(XAie_DmaEnableBd(&([[bd1]])));
// CHECK: __mlir_aie_try(XAie_DmaWriteBd(ctx->XAieDevInst, &([[bd1]]), XAie_TileLoc(2,0), {{.*}} 1));
// CHECK: __mlir_aie_try(XAie_DmaChannelPushBdToQueue(ctx->XAieDevInst, XAie_TileLoc(2,0), {{.*}}0, {{.*}} DMA_S2MM, {{.*}}0));
// CHECK: __mlir_aie_try(XAie_DmaChannelEnable(ctx->XAieDevInst, XAie_TileLoc(2,0), {{.*}} 0, {{.*}} DMA_S2MM));
// CHECK: __mlir_aie_try(XAie_DmaChannelPushBdToQueue(ctx->XAieDevInst, XAie_TileLoc(2,0), {{.*}}0, {{.*}} DMA_MM2S, {{.*}}1));
// CHECK: __mlir_aie_try(XAie_DmaChannelEnable(ctx->XAieDevInst, XAie_TileLoc(2,0), {{.*}} 0, {{.*}} DMA_MM2S));

// CHECK: mlir_aie_configure_switchboxes
// CHECK: x = 2;
// CHECK: y = 0;
// CHECK: __mlir_aie_try(XAie_StrmConnCctEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), NORTH, 0, SOUTH, 2));
// CHECK: __mlir_aie_try(XAie_EnableAieToShimDmaStrmPort(ctx->XAieDevInst, XAie_TileLoc(x,y), 2));


module {
 aie.device(xcvc1902) {
  %buffer = aie.external_buffer { sym_name = "buf" } : memref<16 x f32>
  %t21 = aie.tile(2, 1)
  %t20 = aie.tile(2, 0)
  %c21 = aie.core(%t21)  {
    aie.end
  }
  %s21 = aie.switchbox(%t21)  {
    aie.connect<Core : 0, South : 0>
  }
  %s20 = aie.switchbox(%t20)  {
    aie.connect<North : 0, South : 2>
  }
  %mux = aie.shim_mux(%t20)  {
    aie.connect<North : 2, DMA : 0>
  }
  %dma = aie.shim_dma(%t20)  {
      %lock0 = aie.lock(%t20, 0)
      %lock1 = aie.lock(%t20, 1)

      aie.dma_start(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      aie.dma_start(MM2S, 0, ^bd1, ^end)
    ^bd0:
      aie.use_lock(%lock0, Acquire, 0)
      aie.dma_bd(%buffer : memref<16 x f32>, 0, 16)
      aie.use_lock(%lock0, Release, 1)
      aie.next_bd ^bd0
    ^bd1:
      // aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer : memref<16 x f32>, 0, 4)
      // aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd1
    ^end:
      aie.end
  }
  aie.wire(%s21 : South, %s20 : North)
  aie.wire(%s20 : South, %mux : North)
  aie.wire(%mux : DMA, %dma : DMA)
  aie.wire(%mux : South, %t20 : DMA)
  aie.wire(%s21 : Core, %c21 : Core)
  aie.wire(%s21 : Core, %t21 : Core)
 }
}
