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

// AIE.end is not the last block.

// CHECK: XAieDma_TileBdSetNext(&(ctx->TileDMAInst[8][3]),  /* bd */ 0,  /* nextbd */ 0);
// CHECK: XAieDma_TileBdWrite(&(ctx->TileDMAInst[8][3]),  /* bd */ 0);
// CHECK: XAieDma_TileBdSetNext(&(ctx->TileDMAInst[8][3]),  /* bd */ 1,  /* nextbd */ 1);
// CHECK: XAieDma_TileBdWrite(&(ctx->TileDMAInst[8][3]),  /* bd */ 1);

module @aie_module  {
 AIE.device(xcvc1902) {
  %0 = AIE.tile(8, 3)
  %24 = AIE.buffer(%0) {address = 4096 : i32, sym_name = "buf6"} : memref<64xi32, 2>
  %25 = AIE.lock(%0, 0)
  %26 = AIE.buffer(%0) {address = 4352 : i32, sym_name = "buf7"} : memref<64xi32, 2>
  %27 = AIE.lock(%0, 1)
  %28 = AIE.mem(%0)  {
    %38 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%25, Acquire, 0)
    AIE.dmaBd(<%24 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%25, Release, 1)
    AIE.nextBd ^bb1
  ^bb2:  // pred: ^bb3
    AIE.end
  ^bb3:  // pred: ^bb0
    %39 = AIE.dmaStart(MM2S, 0, ^bb4, ^bb2)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    AIE.useLock(%27, Acquire, 1)
    AIE.dmaBd(<%26 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%27, Release, 0)
    AIE.nextBd ^bb4
  }
 }
}
