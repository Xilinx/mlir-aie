module {
  %0 = aie.core(0, 0)
  %1 = aie.core(1, 1)
  %2 = aie.core(0, 1)
  %3 = aie.core(1, 0)
  %4 = aie.switchbox(0, 0) {
    aie.connect<"DMA" : 0, "North" : 0>
  }
  %5 = aie.switchbox(0, 1) {
    // aie.connect<"South" : 0, "East" : 0>
	 %m1 = aie.masterset(1, "East" : 0 )
	 aie.packetrules("South" : 0) {
		aie.rule(0x1F, 0x10, %m1)
	 }
  }
  %6 = aie.switchbox(1, 0) {
    aie.connect<"North" : 0, "ME" : 1>
  }
  %7 = aie.switchbox(1, 1) {
	 %m1 = aie.masterset(1, "South" : 0 )
	 aie.packetrules("West" : 0) {
		aie.rule(0x10, 0x0, %m1)
	 }
  }
  %8 = aie.shimswitchbox(0) {
  }
  %9 = aie.shimswitchbox(1) {
  }
  %10 = aie.plio(0)
  %11 = aie.plio(1)
  aie.wire(%0 : "ME", %4 : "ME")
  aie.wire(%0 : "DMA", %4 : "DMA")
  aie.wire(%8 : "North", %4 : "South")
  aie.wire(%10 : "North", %8 : "South")
  aie.wire(%2 : "ME", %5 : "ME")
  aie.wire(%2 : "DMA", %5 : "DMA")
  aie.wire(%4 : "North", %5 : "South")
  aie.wire(%3 : "ME", %6 : "ME")
  aie.wire(%3 : "DMA", %6 : "DMA")
  aie.wire(%4 : "East", %6 : "West")
  aie.wire(%9 : "North", %6 : "South")
  aie.wire(%8 : "East", %9 : "West")
  aie.wire(%11 : "North", %9 : "South")
  aie.wire(%1 : "ME", %7 : "ME")
  aie.wire(%1 : "DMA", %7 : "DMA")
  aie.wire(%5 : "East", %7 : "West")
  aie.wire(%6 : "North", %7 : "South")

}
