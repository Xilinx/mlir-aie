// RUN: aie-opt --aie-create-flows --aie-find-flows %s | FileCheck %s
// CHECK: %[[T23:.*]] = AIE.tile(2, 3)
// CHECK: %[[T32:.*]] = AIE.tile(3, 2)
// CHECK: AIE.flow(%[[T23]], "ME" : 1, %[[T32]], "DMA" : 0)

module {
  %0 = AIE.tile(2, 3)
  %1 = AIE.tile(3, 2)
  AIE.flow(%0, "ME" : 1, %1, "DMA" : 0)
}
