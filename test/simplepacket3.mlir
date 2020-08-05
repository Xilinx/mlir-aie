// RUN: aie-opt %s | FileCheck %s
// CHECK-LABEL: module {
// CHECK:       }

module {
  %0 = AIE.tile(0, 0)
  %1 = AIE.tile(1, 1)
  %2 = AIE.tile(0, 1)
  %3 = AIE.tile(1, 0)
  %4 = AIE.switchbox(%0) {
    AIE.connect<"DMA" : 0, "North" : 0>
  }
  %5 = AIE.switchbox(%2) {
    // AIE.connect<"South" : 0, "East" : 0>
    %m1 = AIE.masterset(1, "East" : 0 )
    AIE.packetrules("South" : 0) {
      AIE.rule(0x1F, 0x10, %m1)
    }
  }
  %6 = AIE.switchbox(%3) {
    AIE.connect<"North" : 0, "ME" : 1>
  }
  %7 = AIE.switchbox(%1) {
    %m1 = AIE.masterset(1, "South" : 0 )
    AIE.packetrules("West" : 0) {
      AIE.rule(0x10, 0x0, %m1)
    }
  }
  %8 = AIE.shimswitchbox(0) {
  }
  %9 = AIE.shimswitchbox(1) {
  }
  %10 = AIE.plio(0)
  %11 = AIE.plio(1)
  AIE.wire(%0 : "ME", %4 : "ME")
  AIE.wire(%0 : "DMA", %4 : "DMA")
  AIE.wire(%8 : "North", %4 : "South")
  AIE.wire(%10 : "North", %8 : "South")
  AIE.wire(%2 : "ME", %5 : "ME")
  AIE.wire(%2 : "DMA", %5 : "DMA")
  AIE.wire(%4 : "North", %5 : "South")
  AIE.wire(%3 : "ME", %6 : "ME")
  AIE.wire(%3 : "DMA", %6 : "DMA")
  AIE.wire(%4 : "East", %6 : "West")
  AIE.wire(%9 : "North", %6 : "South")
  AIE.wire(%8 : "East", %9 : "West")
  AIE.wire(%11 : "North", %9 : "South")
  AIE.wire(%1 : "ME", %7 : "ME")
  AIE.wire(%1 : "DMA", %7 : "DMA")
  AIE.wire(%5 : "East", %7 : "West")
  AIE.wire(%6 : "North", %7 : "South")

}
