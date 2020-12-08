// RUN: aie-opt --split-input-file --aie-create-flows %s | FileCheck %s
// CHECK: module
// CHECK: %[[T21:.*]] = AIE.tile(2, 1)
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK:  %{{.*}} = AIE.switchbox(%[[T20]])  {
// CHECK:    AIE.connect<"North" : 0, "South" : 0>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.switchbox(%[[T21]])  {
// CHECK:    AIE.connect<"North" : 0, "South" : 0>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.shimmux(%[[T20]])  {
// CHECK:    AIE.connect<"South" : 0, "PLIO" : 0>
// CHECK:  }

module {
  %t23 = AIE.tile(2, 1)
  %t22 = AIE.tile(2, 0)
  AIE.flow(%t23, "North" : 0, %t22, "PLIO" : 0)
}

// -----

// CHECK: module
// CHECK: %[[T21:.*]] = AIE.tile(2, 1)
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK:  %{{.*}} = AIE.switchbox(%[[T20]])  {
// CHECK:    AIE.connect<"North" : 0, "South" : 0>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.switchbox(%[[T21]])  {
// CHECK:    AIE.connect<"ME" : 0, "South" : 0>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.shimmux(%[[T20]])  {
// FIXME: should be "South":7
// CHECK:    AIE.connect<"South" : 0, "DMA" : 1>
// CHECK:  }
module {
  %t21 = AIE.tile(2, 1)
  %t20 = AIE.tile(2, 0)
  %c21 = AIE.core(%t21)  {
    AIE.end
  }
  %dma = AIE.shimDMA(%t20)  {
    AIE.end
  }
  AIE.flow(%c21, "ME" : 0, %dma, "DMA" : 1)
}

// -----

// CHECK: module
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK:  %{{.*}} = AIE.switchbox(%[[T20]])  {
// CHECK:    AIE.connect<"South" : 0, "South" : 0>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.shimmux(%[[T20]])  {
// FIXME: these connections are wrong.
// CHECK:    AIE.connect<"DMA" : 0, "South" : 0>
// CHECK:    AIE.connect<"South" : 0, "DMA" : 1>
// CHECK:  }
module {
  %t20 = AIE.tile(2, 0)
  %dma = AIE.shimDMA(%t20)  {
    AIE.end
  }
  AIE.flow(%dma, "DMA" : 0, %dma, "DMA" : 1)
}
