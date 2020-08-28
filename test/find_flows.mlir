// RUN: aie-opt -aie-find-flows %s | FileCheck %s
// CHECK: %[[T23:.*]] = AIE.tile(2, 3)
// CHECK: %[[T22:.*]] = AIE.tile(2, 2)
// CHECK: AIE.flow(%[[T23]], "ME" : 0, %[[T22]], "ME" : 1)
// CHECK: AIE.flow(%[[T22]], "ME" : 0, %[[T22]], "ME" : 0)
// CHECK: AIE.flow(%[[T22]], "ME" : 1, %[[T23]], "ME" : 1)

module {
  %tile0 = AIE.tile(2, 3)
  %tile1 = AIE.tile(2, 2)

  %0 = AIE.switchbox(%tile0) {
    AIE.connect<"ME":0, "South":1>
    AIE.connect<"South":1, "ME":1>
    AIE.connect<"South":1, "North":2>
  }
  %1 = AIE.switchbox(%tile1) {
    AIE.connect<"North":1, "ME":1>
    AIE.connect<"ME":1, "North":1>
    AIE.connect<"ME":0, "ME":0>
  }
  AIE.wire(%0: "ME", %tile0: "ME")
  AIE.wire(%1: "ME", %tile1: "ME")
  AIE.wire(%0: "South", %1: "North")
}
