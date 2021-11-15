//===- tileDMA.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: void mlir_configure_dmas() {
// CHECK: XAieDma_TileBdSetXy2d(&(TileDMAInst[7][2]),  /* bd */ 0,  /* xyType */ XAIEDMA_TILE_BD_2DDMA_X, /* incr */ 1, /* wrap */ 8, /* ofst */ 4);
// CHECK: XAieDma_TileBdSetXy2d(&(TileDMAInst[7][2]),  /* bd */ 0,  /* xyType */ XAIEDMA_TILE_BD_2DDMA_Y, /* incr */ 8, /* wrap */ 4, /* ofst */ 1);
// CHECK: XAieDma_TileBdWrite(&(TileDMAInst[7][2]),  /* bd */ 0);
// CHECK: XAieDma_TileBdSetXy2d(&(TileDMAInst[7][2]),  /* bd */ 1,  /* xyType */ XAIEDMA_TILE_BD_2DDMA_X, /* incr */ 1, /* wrap */ 8, /* ofst */ 4);
// CHECK: XAieDma_TileBdSetXy2d(&(TileDMAInst[7][2]),  /* bd */ 1,  /* xyType */ XAIEDMA_TILE_BD_2DDMA_Y, /* incr */ 8, /* wrap */ 4, /* ofst */ 1);
// CHECK: XAieDma_TileBdWrite(&(TileDMAInst[7][2]),  /* bd */ 1);

//
// This test configures the tile DMA buffer descriptor to transpose an 8x4xi32
// input matrix as it is writted to the tile's local memory. It returns the
// matrix to its original arrangement when it is output to the stream.
//
module @aie_module  {
  %0 = AIE.tile(7, 2)
  %24 = AIE.buffer(%0) {address = 4096 : i32, sym_name = "buf6"} : memref<4x8xi32, 2>
  %25 = AIE.lock(%0, 0)
  %26 = AIE.buffer(%0) {address = 4224 : i32, sym_name = "buf7"} : memref<4x8xi32, 2>
  %27 = AIE.lock(%0, 1)
  %28 = AIE.mem(%0)  {
    %38 = AIE.dmaStart(S2MM0, ^bb1, ^bb3)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%25, Acquire, 0, 0)
    AIE.dmaBd2D(4, 1, 8, 1, 8, 4)
    AIE.dmaBd(<%24 : memref<4x8xi32, 2>, 0, 32>, 0)
    AIE.useLock(%25, Release, 1, 0)
    br ^bb1
  ^bb2:  // pred: ^bb3
    AIE.end
  ^bb3:  // pred: ^bb0
    %39 = AIE.dmaStart(MM2S0, ^bb4, ^bb2)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    AIE.useLock(%27, Acquire, 1, 0)
    AIE.dmaBd2D(4, 1, 8, 1, 8, 4)
    AIE.dmaBd(<%26 : memref<4x8xi32, 2>, 0, 32>, 0)
    AIE.useLock(%27, Release, 0, 0)
    br ^bb4
  }
}
