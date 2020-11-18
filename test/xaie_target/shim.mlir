// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_start_cores
// CHECK: XAieTile_CoreControl(&(TileInst[2][1]), XAIE_ENABLE, XAIE_DISABLE);
// CHECK: XAieTile_CoreControl(&(TileInst[2][0]), XAIE_ENABLE, XAIE_DISABLE);
// CHECK: mlir_configure_switchboxes
// CHECK: x = 2;
// CHECK: y = 0;
// CHECK: XAieTile_ShimStrmMuxConfig(&(TileInst[x][y]),
// CHECK:        XAIETILE_SHIM_STRM_MUX_NORTH2, DMA);

module {
  %t21 = AIE.tile(2, 1)
  %t20 = AIE.tile(2, 0)
  %c21 = AIE.core(%t21)  {
    AIE.end
  }
  %s21 = AIE.switchbox(%t21)  {
    AIE.connect<"ME" : 0, "South" : 0>
  }
  %c20 = AIE.core(%t20)  {
    AIE.end
  }
  %s20 = AIE.switchbox(%t20)  {
    AIE.connect<"North" : 0, "South" : 2>
  }
  %mux = AIE.shimmux(%t20)  {
    AIE.connect<"North" : 2, "DMA" : 0>
  }
  %dma = AIE.shimDMA(%t20)  {
    AIE.end
  }
  AIE.wire(%s21 : "South", %s20 : "North")
  AIE.wire(%s20 : "South", %mux : "North")
  AIE.wire(%mux : "DMA", %dma : "DMA")
  AIE.wire(%mux : "South", %t20 : "DMA")
  AIE.wire(%s21 : "ME", %c21 : "ME")
  AIE.wire(%s21 : "ME", %t21 : "ME")
}