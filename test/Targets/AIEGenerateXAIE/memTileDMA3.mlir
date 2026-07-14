//===- memTileDMA3.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// Check for ordered queuing of BD tasks.

// CHECK: __mlir_aie_try(XAie_DmaChannelSetStartQueue(ctx->XAieDevInst, XAie_TileLoc(2,1), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */0, /* Repeat */ 1, /* EnToken */ XAIE_DISABLE));
// CHECK: __mlir_aie_try(XAie_DmaChannelSetStartQueue(ctx->XAieDevInst, XAie_TileLoc(2,1), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */1, /* Repeat */ 8, /* EnToken */ XAIE_DISABLE));
// CHECK: __mlir_aie_try(XAie_DmaChannelSetStartQueue(ctx->XAieDevInst, XAie_TileLoc(2,1), /* ChNum */0, /* dmaDir */ DMA_MM2S, /* BdNum */3, /* Repeat */ 1, /* EnToken */ XAIE_DISABLE));
// CHECK: __mlir_aie_try(XAie_DmaChannelSetStartQueue(ctx->XAieDevInst, XAie_TileLoc(2,1), /* ChNum */0, /* dmaDir */ DMA_MM2S, /* BdNum */2, /* Repeat */ 8, /* EnToken */ XAIE_DISABLE));

module @aie_module {
  aie.device(xcve2802) {
    %tile_2_1 = aie.tile(2, 1)
    %buf8 = aie.buffer(%tile_2_1) {address = 1824 : i32, sym_name = "buf8"} : memref<12xi32, 1 : i32> 
    %buf7 = aie.buffer(%tile_2_1) {address = 1872 : i32, sym_name = "buf7"} : memref<5xi32, 1 : i32> 
    %lock_2_1 = aie.lock(%tile_2_1, 1) {init = 1 : i32}
    %lock_2_1_0 = aie.lock(%tile_2_1, 0) {init = 0 : i32}
    %memtile_dma_2_1 = aie.memtile_dma(%tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // pred: ^bb0
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, %c1_ul1)
      aie.dma_bd(%buf7 : memref<5xi32, 1 : i32> offset = 0 len = 5)
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(%lock_2_1_0, Release, %c1_ul2)
      aie.next_bd ^bb4
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb7, repeat_count = 7)
    ^bb3:  // pred: ^bb2
      %c1_ul3 = arith.constant 1 : i32
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, %c1_ul3)
      aie.dma_bd(%buf8 : memref<12xi32, 1 : i32> offset = 0 len = 12)
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(%lock_2_1_0, Release, %c1_ul4)
      aie.next_bd ^bb4
    ^bb4:  // 5 preds: ^bb1, ^bb3, ^bb5, ^bb6, ^bb8
      aie.end
    ^bb5:  // pred: ^bb7
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb4, repeat_count = 7)
    ^bb6:  // pred: ^bb5
      %c1_ul5 = arith.constant 1 : i32
      aie.use_lock(%lock_2_1_0, AcquireGreaterEqual, %c1_ul5)
      aie.dma_bd(%buf8 : memref<12xi32, 1 : i32> offset = 0 len = 12)
      %c1_ul6 = arith.constant 1 : i32
      aie.use_lock(%lock_2_1, Release, %c1_ul6)
      aie.next_bd ^bb4
    ^bb7:  // pred: ^bb2
      %3 = aie.dma_start(MM2S, 0, ^bb8, ^bb5)
    ^bb8:  // pred: ^bb7
      %c1_ul7 = arith.constant 1 : i32
      aie.use_lock(%lock_2_1_0, AcquireGreaterEqual, %c1_ul7)
      aie.dma_bd(%buf7 : memref<5xi32, 1 : i32> offset = 0 len = 5)
      %c1_ul8 = arith.constant 1 : i32
      aie.use_lock(%lock_2_1, Release, %c1_ul8)
      aie.next_bd ^bb4
    }
  }
}
