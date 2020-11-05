// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// Test S2MM, BD chaining.
// CHECK: XAieTile_CoreControl(&(TileInst[3][3]), XAIE_ENABLE, XAIE_DISABLE);
// CHECK: XAieDma_TileInitialize(&(TileInst[3][3]), &(TileDMAInst[3][3]));
// CHECK: XAieDma_TileBdClearAll(&(TileDMAInst[3][3]));
// CHECK: XAieDma_TileChResetAll(&(TileDMAInst[3][3]));
// CHECK: XAieDma_TileBdSetLock(&(TileDMAInst[3][3]), {{.*}}0, XAIEDMA_TILE_BD_ADDRA, {{.*}}0, XAIE_ENABLE, {{.*}}1, XAIE_ENABLE, {{.*}}0);
// CHECK: XAieDma_TileBdSetAdrLenMod(&(TileDMAInst[3][3]), {{.*}}0, {{.*}}0x0, {{.*}}0x0, {{.*}}256, {{.*}}XAIE_DISABLE, {{.*}}XAIE_DISABLE);
// CHECK: XAieDma_TileBdSetNext(&(TileDMAInst[3][3]), {{.*}}0, {{.*}}1);
// CHECK: XAieDma_TileBdWrite(&(TileDMAInst[3][3]), {{.*}}0);
// CHECK: XAieDma_TileBdSetLock(&(TileDMAInst[3][3]), {{.*}}1, XAIEDMA_TILE_BD_ADDRA, {{.*}}0, XAIE_ENABLE, {{.*}}1, XAIE_ENABLE, {{.*}}0);
// CHECK: XAieDma_TileBdSetAdrLenMod(&(TileDMAInst[3][3]), {{.*}}1, {{.*}}0x400, {{.*}}0x0, {{.*}}16, {{.*}}XAIE_DISABLE, {{.*}}XAIE_DISABLE);
// CHECK: XAieDma_TileBdSetNext(&(TileDMAInst[3][3]), {{.*}}1, {{.*}}0);
// CHECK: XAieDma_TileBdWrite(&(TileDMAInst[3][3]), {{.*}}1);
// CHECK: XAieDma_TileSetStartBd(&(TileDMAInst[3][3]), XAIEDMA_TILE_CHNUM_S2MM0, {{.*}}0);
// CHECK: XAieDma_TileChControl(&(TileDMAInst[3][3]), XAIEDMA_TILE_CHNUM_S2MM0, XAIE_RESETDISABLE, XAIE_ENABLE);

module @test_xaie2 {
  %t33 = AIE.tile(3, 3)

  %buf33_0 = AIE.buffer(%t33) : memref<256xi32>
  %buf33_1 = AIE.buffer(%t33) : memref<16xi32>

  %l33_0 = AIE.lock(%t33, 0)
  %l33_1 = AIE.lock(%t33, 1)

  %m33 = AIE.mem(%t33) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l33_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf33_0 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l33_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l33_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf33_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l33_0, "Release", 1, 0)
      br ^bd0
    ^end:
      AIE.end
  }
}
