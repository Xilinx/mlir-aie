//===- aie2_tileDMA5.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// Check for ordered queuing of BD tasks.

// CHECK: __mlir_aie_try(XAie_DmaChannelSetStartQueue(ctx->XAieDevInst, XAie_TileLoc(7,3), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */0, /* Repeat */ 1, /* EnToken */ XAIE_DISABLE));
// CHECK: __mlir_aie_try(XAie_DmaChannelSetStartQueue(ctx->XAieDevInst, XAie_TileLoc(7,3), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */1, /* Repeat */ 8, /* EnToken */ XAIE_DISABLE));
// CHECK: __mlir_aie_try(XAie_DmaChannelSetStartQueue(ctx->XAieDevInst, XAie_TileLoc(7,3), /* ChNum */0, /* dmaDir */ DMA_MM2S, /* BdNum */3, /* Repeat */ 1, /* EnToken */ XAIE_DISABLE));
// CHECK: __mlir_aie_try(XAie_DmaChannelSetStartQueue(ctx->XAieDevInst, XAie_TileLoc(7,3), /* ChNum */0, /* dmaDir */ DMA_MM2S, /* BdNum */2, /* Repeat */ 8, /* EnToken */ XAIE_DISABLE));

module @aie_module {
  aie.device(xcve2802) {
    %tile_7_3 = aie.tile(7, 3)
    %buf8 = aie.buffer(%tile_7_3) {address = 1824 : i32, sym_name = "buf8"} : memref<12xi32, 1 : i32> 
    %buf7 = aie.buffer(%tile_7_3) {address = 1872 : i32, sym_name = "buf7"} : memref<5xi32, 1 : i32> 
    %lock_7_3 = aie.lock(%tile_7_3, 1) {init = 1 : i32}
    %lock_7_3_0 = aie.lock(%tile_7_3, 0) {init = 0 : i32}
    %mem_7_3 = aie.mem(%tile_7_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%lock_7_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf7 : memref<5xi32, 1 : i32>, 0, 5)
      aie.use_lock(%lock_7_3_0, Release, 1)
      aie.next_bd ^bb4
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb7, repeat_count = 7)
    ^bb3:  // pred: ^bb2
      aie.use_lock(%lock_7_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<12xi32, 1 : i32>, 0, 12)
      aie.use_lock(%lock_7_3_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // 5 preds: ^bb1, ^bb3, ^bb5, ^bb6, ^bb8
      aie.end
    ^bb5:  // pred: ^bb7
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb4, repeat_count = 7)
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_7_3_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<12xi32, 1 : i32>, 0, 12)
      aie.use_lock(%lock_7_3, Release, 1)
      aie.next_bd ^bb4
    ^bb7:  // pred: ^bb2
      %3 = aie.dma_start(MM2S, 0, ^bb8, ^bb5)
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_7_3_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf7 : memref<5xi32, 1 : i32>, 0, 5)
      aie.use_lock(%lock_7_3, Release, 1)
      aie.next_bd ^bb4
    }
  }
}
