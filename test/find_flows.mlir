

module {
  %0 = aie.switchbox(2, 3) {
    aie.connect<"ME":0, "South":1>
	 aie.connect<"South":1, "ME":1>
	 aie.connect<"South":1, "North":2>
  }
  %1 = aie.switchbox(2, 2) {
    aie.connect<"North":1, "ME":1>
	 aie.connect<"ME":1, "North":1>
	 aie.connect<"ME":0, "ME":0>
  }
  %core0 = aie.core(2, 3)
  %core1 = aie.core(2, 2)
  aie.wire(%0: "ME", %core0: "ME")
  aie.wire(%1: "ME", %core1: "ME")
  aie.wire(%0: "South", %1: "North")
  }
