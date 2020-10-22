// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: XAieTile_CoreControl(&(TileInst[3][3]), XAIE_ENABLE, XAIE_DISABLE);
// CHECK: XAieDma_TileInitialize(&(TileInst[3][3]), &(TileDMAInst[3][3]));
// CHECK: XAieDma_TileBdClearAll(&(TileDMAInst[3][3]));
// CHECK: XAieDma_TileChResetAll(&(TileDMAInst[3][3]));
// CHECK: XAieDma_TileBdSetLock(&(TileDMAInst[3][3]), {{.*}}0, XAIEDMA_TILE_BD_ADDRA, {{.*}}0, XAIE_ENABLE, {{.*}}1, XAIE_ENABLE, {{.*}}0);
// CHECK: XAieDma_TileBdSetAdrLenMod(&(TileDMAInst[3][3]), {{.*}}0, {{.*}}0x400, {{.*}}0x0, {{.*}}256, {{.*}}XAIE_ENABLE, {{.*}}XAIE_DISABLE);
// CHECK: XAieDma_TileBdWrite(&(TileDMAInst[3][3]), {{.*}}0);
// CHECK: XAieDma_TileSetStartBd(&(TileDMAInst[3][3]), XAIEDMA_TILE_CHNUM_MM2S0, {{.*}}0);
// CHECK: XAieDma_TileChControl(&(TileDMAInst[3][3]), XAIEDMA_TILE_CHNUM_MM2S0, XAIE_RESETDISABLE, XAIE_ENABLE);

module @test_xaie1 {
  %t33 = AIE.tile(3, 3)

  %buf33_0 = AIE.buffer(%t33) : memref<256xi32>
  %buf33_1 = AIE.buffer(%t33) : memref<256xi32>
  %l33_0 = AIE.lock(%t33, 0)

  %m33 = AIE.mem(%t33) {
    %srcDma = AIE.dmaStart("MM2S0")
    AIE.terminator(^dma0, ^end)
    ^dma0:
      cond_br %srcDma, ^bd0, ^end
    ^bd0:
      AIE.useLock(%l33_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf33_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l33_0, "Release", 1, 0)
      br ^end
    ^end:
      AIE.end
  }

}
