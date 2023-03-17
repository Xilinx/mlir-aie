//===- test_xaie3.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// Test acquire with '1'.   Single BD.
// CHECK: XAieDma_TileBdSetLock(&(ctx->TileDMAInst[3][3]), {{.*}}0, XAIEDMA_TILE_BD_ADDRA, {{.*}}0, XAIE_ENABLE, {{.*}}0, XAIE_ENABLE, {{.*}}1);
// CHECK: XAieDma_TileBdSetAdrLenMod(&(ctx->TileDMAInst[3][3]), {{.*}}0, {{.*}}0x0, {{.*}}0x0, {{.*}}256 * 4, {{.*}}XAIE_DISABLE, {{.*}}XAIE_DISABLE);
// CHECK: XAieDma_TileBdWrite(&(ctx->TileDMAInst[3][3]), {{.*}}0);
// CHECK: XAieDma_TileSetStartBd((&(ctx->TileDMAInst[3][3])), XAIEDMA_TILE_CHNUM_MM2S0, {{.*}}0);

module @test_xaie3 {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)
  %t44 = AIE.tile(4, 4)

  %buf33_0 = AIE.buffer(%t33) { address = 0, sym_name = "buf33_0" }: memref<256xi32>

  %l33_0 = AIE.lock(%t33, 0)

  %m33 = AIE.mem(%t33) {
      %srcDma = AIE.dmaStart(MM2S, 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l33_0, Acquire, 1)
      AIE.dmaBd(<%buf33_0 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l33_0, Release, 0)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }
 }
}
