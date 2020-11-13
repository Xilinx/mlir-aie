// RUN: aie-opt --split-input-file --aie-create-flows %s | FileCheck %s
// CHECK: %[[T21:.*]] = AIE.tile(2, 1)
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK:  %{{.*}} = AIE.switchbox(%[[T20]])  {
// CHECK:    AIE.connect<"North" : 0, "South" : 4>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.switchbox(%[[T21]])  {
// CHECK:    AIE.connect<"North" : 0, "South" : 0>
// CHECK:  }

module {
  %t23 = AIE.tile(2, 1)
  %t22 = AIE.tile(2, 0)
  AIE.flow(%t23, "North" : 0, %t22, "South" : 4)
}

// -----

// CHECK: %[[T21:.*]] = AIE.tile(2, 1)
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK:  %{{.*}} = AIE.switchbox(%[[T20]])  {
// CHECK:    AIE.connect<"DMA" : 0, "DMA" : 0>
// CHECK:  }
module {
  %t23 = AIE.tile(2, 1)
  %t22 = AIE.tile(2, 0)
  AIE.flow(%t22, "DMA" : 0, %t22, "DMA" : 0)
}

// -----

// CHECK: %[[T21:.*]] = AIE.tile(2, 1)
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK:  %{{.*}} = AIE.switchbox(%[[T20]])  {
// CHECK:    AIE.connect<"North" : 0, "South" : 5>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.switchbox(%[[T21]])  {
// CHECK:    AIE.connect<"DMA" : 0, "South" : 0>
// CHECK:  }
module {
  %t23 = AIE.tile(2, 1)
  %t22 = AIE.tile(2, 0)
  AIE.flow(%t23, "DMA" : 0, %t22, "South" : 5)
//  AIE.flow(%t23, "ME" : 0, %p0, "South" : 0)
}

// -----

// CHECK: %[[T21:.*]] = AIE.tile(2, 1)
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK:  %{{.*}} = AIE.switchbox(%[[T20]])  {
// CHECK:    AIE.connect<"North" : 0, "South" : 5>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.switchbox(%[[T21]])  {
// CHECK:    AIE.connect<"DMA" : 0, "South" : 0>
// CHECK:  }
module {
  %t23 = AIE.tile(2, 1)
  %t22 = AIE.tile(2, 0)
  AIE.flow(%t23, "DMA" : 0, %t22, "DMA" : 0)
}

// -----

// CHECK: %[[T21:.*]] = AIE.tile(2, 1)
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK:  %{{.*}} = AIE.switchbox(%[[T20]])  {
// CHECK:    AIE.connect<"North" : 0, "South" : 5>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.switchbox(%[[T21]])  {
// CHECK:    AIE.connect<"DMA" : 0, "South" : 0>
// CHECK:  }
module {
  %t23 = AIE.tile(2, 1)
  %t22 = AIE.tile(2, 0)
  AIE.flow(%t23, "DMA" : 0, %t22, "PLIO" : 0)
}
