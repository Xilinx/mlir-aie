// RUN: aie-opt %s | FileCheck %s
// CHECK-LABEL: module {
// CHECK:       }

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
  %tile0 = AIE.tile(2, 3)
  %tile1 = AIE.tile(2, 2)
  AIE.wire(%0: "ME", %tile0: "ME")
  AIE.wire(%1: "ME", %tile1: "ME")
  AIE.wire(%0: "South", %1: "North")
}
