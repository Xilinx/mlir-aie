

module {
  %0 = AIE.switchbox(2, 3) {
    AIE.connect<"ME":0, "South":1>
	 AIE.connect<"South":1, "ME":1>
	 AIE.connect<"South":1, "North":2>
  }
  %1 = AIE.switchbox(2, 2) {
    AIE.connect<"North":1, "ME":1>
	 AIE.connect<"ME":1, "North":1>
	 AIE.connect<"ME":0, "ME":0>
  }
  %core0 = AIE.core(2, 3)
  %core1 = AIE.core(2, 2)
  AIE.wire(%0: "ME", %core0: "ME")
  AIE.wire(%1: "ME", %core1: "ME")
  AIE.wire(%0: "South", %1: "North")
  }
