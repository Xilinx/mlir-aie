// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// Verifier: useLock should be contained in a core or a memory.
// CHECK: XAieTile_CoreControl(&(TileInst[3][3]), XAIE_ENABLE, XAIE_DISABLE);
// CHECK: XAieTile_LockAcquire(&(TileDMAInst[3][3]), 8, 0, 0);
// CHECK: XAieTile_LockRelease(&(TileDMAInst[3][3]), 8, 1, 0);

module @test_xaie0 {
  %t33 = AIE.tile(3, 3)

  %l33_8 = AIE.lock(%t33, 8)

  AIE.useLock(%l33_8, "Acquire", 0, 0)
  AIE.useLock(%l33_8, "Release", 1, 0)
}
