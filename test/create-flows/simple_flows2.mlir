// RUN: aie-opt --aie-create-flows --aie-find-flows %s | FileCheck %s
// CHECK: %[[T23:.*]] = AIE.tile(2, 3)
// CHECK: %[[T22:.*]] = AIE.tile(2, 2)
// CHECK: %[[T11:.*]] = AIE.tile(1, 1)
// CHECK: AIE.flow(%[[T23]], Core : 0, %[[T22]], Core : 1)
// CHECK: AIE.flow(%[[T22]], Core : 0, %[[T11]], Core : 0)

module {
  %t23 = AIE.tile(2, 3)
  %t22 = AIE.tile(2, 2)
  %t11 = AIE.tile(1, 1)
  AIE.flow(%t23, Core : 0, %t22, Core : 1)
  AIE.flow(%t22, Core : 0, %t11, Core : 0)
}