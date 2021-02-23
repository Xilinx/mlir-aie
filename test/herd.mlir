// RUN: aie-opt %s | FileCheck %s
// CHECK-LABEL: module {
// CHECK:       }

module {
  %2 = AIE.tile(2, 3)
  %3 = AIE.tile(2, 2)
  AIE.flow(%2, Core : 0, %3, Core : 1)
  AIE.flow(%3, Core : 0, %3, Core : 0)
  AIE.flow(%3, Core : 1, %2, Core : 1)
}
