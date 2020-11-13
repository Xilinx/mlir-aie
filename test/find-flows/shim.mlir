// RUN: aie-opt --aie-find-flows %s | FileCheck %s

// CHECK: %0 = AIE.tile(2, 1)
// CHECK: %1 = AIE.tile(2, 0)
// CHECK: %7 = AIE.shimDMA(%1)
// CHECK: AIE.flow(%0, "ME" : 0, %7, "DMA" : 0)
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