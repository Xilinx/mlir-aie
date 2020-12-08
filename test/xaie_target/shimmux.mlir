// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_configure_switchboxes
// CHECK: XAieTile_ShimStrmMuxConfig(&(TileInst[x][y]),
// CHECK:        XAIETILE_SHIM_STRM_MUX_SOUTH3,
// CHECK:        XAIETILE_SHIM_STRM_MUX_DMA);

module {
  %t20 = AIE.tile(2, 0)
  %mux = AIE.shimmux(%t20)  {
    AIE.connect<"DMA" : 0, "South" : 3>
  }
}