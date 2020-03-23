

module {
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %0 = aie.switchbox(%c2, %c3) {
    aie.connect<"ME0", "South1">
	 aie.connect<"South1", "ME1">
	 aie.connect<"South1", "North2">
  }
  %1 = aie.switchbox(%c3, %c2) {
    aie.connect<"North1", "ME1">
	 aie.connect<"ME1", "North1">
	 aie.connect<"ME0", "ME0">
  }
  %core0 = aie.core(%c2, %c3)
  %core1 = aie.core(%c3, %c2)
  aie.wire(%0: "ME0", %core0: "ME0")
  aie.wire(%0: "ME1", %core0: "ME1")
  aie.wire(%1: "ME0", %core1: "ME0")
  aie.wire(%1: "ME1", %core1: "ME1")
  aie.wire(%0: "South1", %1: "North1")
  aie.wire(%0: "South2", %1: "North2")
  aie.wire(%0: "South3", %1: "North3")
  aie.wire(%core0: "ME0", %0: "ME0")
  aie.wire(%core0: "ME1", %0: "ME1")
  aie.wire(%core1: "ME0", %1: "ME0")
  aie.wire(%core1: "ME1", %1: "ME1")
  aie.wire(%1: "North1", %0: "South1")
  aie.wire(%1: "North2", %0: "South2")
  aie.wire(%1: "North3", %0: "South3")
  }
