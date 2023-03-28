//===- test_xaie1.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: XAieTile_CoreControl(&(ctx->TileInst[3][3]), XAIE_ENABLE, XAIE_DISABLE);
// CHECK: XAieDma_TileInitialize(&(ctx->TileInst[3][3]), &(ctx->TileDMAInst[3][3]));
// CHECK: XAieDma_TileBdClearAll(&(ctx->TileDMAInst[3][3]));
// CHECK: XAieDma_TileChResetAll(&(ctx->TileDMAInst[3][3]));
// CHECK: XAieDma_TileBdSetLock(&(ctx->TileDMAInst[3][3]), {{.*}}0, XAIEDMA_TILE_BD_ADDRA, {{.*}}0, XAIE_ENABLE, {{.*}}1, XAIE_ENABLE, {{.*}}0);
// CHECK: XAieDma_TileBdSetAdrLenMod(&(ctx->TileDMAInst[3][3]), {{.*}}0, {{.*}}0x1400, {{.*}}0x0, {{.*}}256 * 4, {{.*}}XAIE_DISABLE, {{.*}}XAIE_DISABLE);
// CHECK: XAieDma_TileBdWrite(&(ctx->TileDMAInst[3][3]), {{.*}}0);
// CHECK: XAieDma_TileSetStartBd((&(ctx->TileDMAInst[3][3])), XAIEDMA_TILE_CHNUM_MM2S0, {{.*}}0);
// CHECK: XAieDma_TileChControl(&(ctx->TileDMAInst[3][3]), XAIEDMA_TILE_CHNUM_MM2S0, XAIE_RESETDISABLE, XAIE_ENABLE);

module @test_xaie1 {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)

  %buf33_0 = AIE.buffer(%t33) {address = 4096 : i32, sym_name = "buf33_0"} : memref<256xi32>
  %buf33_1 = AIE.buffer(%t33) {address = 5120 : i32, sym_name = "buf33_1"} : memref<256xi32>
  %l33_0 = AIE.lock(%t33, 0)

  %m33 = AIE.mem(%t33) {
      %srcDma = AIE.dmaStart(MM2S, 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l33_0, Acquire, 0)
      AIE.dmaBd(<%buf33_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l33_0, Release, 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

 }
}
