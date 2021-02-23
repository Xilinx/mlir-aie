// RUN: aie-opt --aie-create-flows --aie-find-flows %s | FileCheck %s
// CHECK: %[[T23:.*]] = AIE.tile(2, 3)
// CHECK: %[[T22:.*]] = AIE.tile(2, 2)
// CHECK: AIE.flow(%[[T23]], Core : 0, %[[T22]], Core : 1)
// CHECK: AIE.flow(%[[T22]], Core : 0, %[[T22]], Core : 0)
// CHECK: AIE.flow(%[[T22]], Core : 1, %[[T23]], Core : 1)

module {
  %t23 = AIE.tile(2, 3)
  %t22 = AIE.tile(2, 2)
  AIE.flow(%t23, Core : 0, %t22, Core : 1)
  AIE.flow(%t22, Core : 0, %t22, Core : 0)
  AIE.flow(%t22, Core : 1, %t23, Core : 1)
}