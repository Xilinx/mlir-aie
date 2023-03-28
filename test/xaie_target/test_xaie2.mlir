//===- test_xaie2.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// Test S2MM, BD chaining.
// CHECK: XAieTile_CoreControl(&(ctx->TileInst[3][3]), XAIE_ENABLE, XAIE_DISABLE);
// CHECK: XAieDma_TileInitialize(&(ctx->TileInst[3][3]), &(ctx->TileDMAInst[3][3]));
// CHECK: XAieDma_TileBdClearAll(&(ctx->TileDMAInst[3][3]));
// CHECK: XAieDma_TileChResetAll(&(ctx->TileDMAInst[3][3]));
// CHECK: XAieDma_TileBdSetLock(&(ctx->TileDMAInst[3][3]), {{.*}}0, XAIEDMA_TILE_BD_ADDRA, {{.*}}0, XAIE_ENABLE, {{.*}}1, XAIE_ENABLE, {{.*}}0);
// CHECK: XAieDma_TileBdSetAdrLenMod(&(ctx->TileDMAInst[3][3]), {{.*}}0, {{.*}}0x1000, {{.*}}0x0, {{.*}}256 * 4, {{.*}}XAIE_DISABLE, {{.*}}XAIE_DISABLE);
// CHECK: XAieDma_TileBdSetNext(&(ctx->TileDMAInst[3][3]), {{.*}}0, {{.*}}1);
// CHECK: XAieDma_TileBdWrite(&(ctx->TileDMAInst[3][3]), {{.*}}0);
// CHECK: XAieDma_TileBdSetLock(&(ctx->TileDMAInst[3][3]), {{.*}}1, XAIEDMA_TILE_BD_ADDRA, {{.*}}0, XAIE_ENABLE, {{.*}}1, XAIE_ENABLE, {{.*}}0);
// CHECK: XAieDma_TileBdSetAdrLenMod(&(ctx->TileDMAInst[3][3]), {{.*}}1, {{.*}}0x1400, {{.*}}0x0, {{.*}}4 * 4, {{.*}}XAIE_DISABLE, {{.*}}XAIE_DISABLE);
// CHECK: XAieDma_TileBdSetNext(&(ctx->TileDMAInst[3][3]), {{.*}}1, {{.*}}0);
// CHECK: XAieDma_TileBdWrite(&(ctx->TileDMAInst[3][3]), {{.*}}1);
// CHECK: XAieDma_TileSetStartBd((&(ctx->TileDMAInst[3][3])), XAIEDMA_TILE_CHNUM_S2MM0, {{.*}}0);
// CHECK: XAieDma_TileChControl(&(ctx->TileDMAInst[3][3]), XAIEDMA_TILE_CHNUM_S2MM0, XAIE_RESETDISABLE, XAIE_ENABLE);

module @test_xaie2 {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)

  %buf33_0 = AIE.buffer(%t33) { address = 0x1000, sym_name = "buff33_0" }: memref<256xi32>
  %buf33_1 = AIE.buffer(%t33) { address = 0x1400, sym_name = "buff33_1" }: memref<16xi32>

  %l33_0 = AIE.lock(%t33, 0)
  %l33_1 = AIE.lock(%t33, 1)

  %m33 = AIE.mem(%t33) {
      %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l33_0, Acquire, 0)
      AIE.dmaBd(<%buf33_0 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l33_0, Release, 1)
      AIE.nextBd ^bd1
    ^bd1:
      AIE.useLock(%l33_0, Acquire, 0)
      AIE.dmaBd(<%buf33_1 : memref<16xi32>, 0, 4>, 0)
      AIE.useLock(%l33_0, Release, 1)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }
 }
}
