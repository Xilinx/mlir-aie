// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// Test shared BD list.
// CHECK: XAieDma_TileBdSetLock(&(TileDMAInst[3][3]), {{.*}}0, XAIEDMA_TILE_BD_ADDRA, {{.*}}0, XAIE_ENABLE, {{.*}}0, XAIE_ENABLE, {{.*}}1);
// CHECK: XAieDma_TileBdSetAdrLenMod(&(TileDMAInst[3][3]), {{.*}}0, {{.*}}0x0, {{.*}}0x0, {{.*}}256, {{.*}}XAIE_DISABLE, {{.*}}XAIE_DISABLE);
// CHECK: XAieDma_TileBdWrite(&(TileDMAInst[3][3]), {{.*}}0);
// CHECK: XAieDma_TileBdSetLock(&(TileDMAInst[3][3]), {{.*}}2, XAIEDMA_TILE_BD_ADDRA, {{.*}}1, XAIE_ENABLE, {{.*}}0, XAIE_ENABLE, {{.*}}1);
// CHECK: XAieDma_TileBdSetAdrLenMod(&(TileDMAInst[3][3]), {{.*}}2, {{.*}}0x400, {{.*}}0x0, {{.*}}256, {{.*}}XAIE_DISABLE, {{.*}}XAIE_DISABLE);
// CHECK: XAieDma_TileBdWrite(&(TileDMAInst[3][3]), {{.*}}2);
// CHECK: XAieDma_TileSetStartBd(&(TileDMAInst[3][3]), XAIEDMA_TILE_CHNUM_MM2S0, {{.*}}0);
// CHECK: XAieDma_TileSetStartBd(&(TileDMAInst[3][3]), XAIEDMA_TILE_CHNUM_S2MM0, {{.*}}2);

module @test_xaie3 {
  %t33 = AIE.tile(3, 3)
  %t44 = AIE.tile(4, 4)

  %buf33_0 = AIE.buffer(%t33) : memref<256xi32>
  %buf33_1 = AIE.buffer(%t33) : memref<256xi32>

  %l33_0 = AIE.lock(%t33, 0)
  %l33_1 = AIE.lock(%t33, 1)

  %m33 = AIE.mem(%t33) {
    %srcDma = AIE.dmaStart("MM2S0")
    %destDma = AIE.dmaStart("S2MM0")
    AIE.terminator(^dma0, ^dma1, ^end)
    ^dma0:
      cond_br %srcDma, ^bd0, ^end
    ^bd0:
      AIE.useLock(%l33_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf33_0 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l33_0, "Release", 0, 0)
      br ^end
    ^dma1:
      cond_br %destDma, ^bd1, ^end
    ^bd1:
      AIE.useLock(%l33_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf33_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l33_1, "Release", 0, 0)
      br ^end
    ^end:
      AIE.end
  }
}
