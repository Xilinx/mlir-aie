// RUN: aie-opt --aie-lower-multicast %s | FileCheck %s

// CHECK-LABEL: module @test_multicast { 
// CHECK:         %0 = AIE.tile(7, 0)
// CHECK-NEXT:    %1 = AIE.tile(7, 3)
// CHECK-NEXT:    %2 = AIE.tile(7, 4)
// CHECK-NEXT:    %3 = AIE.tile(6, 3)
// CHECK-NEXT:    %4 = AIE.tile(6, 4)
// CHECK-NEXT:    AIE.flow(%0, DMA : 0, %1, DMA : 0)
// CHECK-NEXT:    AIE.flow(%0, DMA : 0, %2, DMA : 0)
// CHECK-NEXT:    AIE.flow(%0, DMA : 0, %3, DMA : 0)
// CHECK-NEXT:    AIE.flow(%0, DMA : 0, %4, DMA : 0)
// CHECK-NEXT:  }

module @test_multicast {
 AIE.device(xcvc1902) {
  %70 = AIE.tile(7, 0)
  %73 = AIE.tile(7, 3)
  %74 = AIE.tile(7, 4)
  %63 = AIE.tile(6, 3)
  %64 = AIE.tile(6, 4)
  AIEX.multicast(%70, "DMA" : 0){
    AIEX.multi_dest<%73, "DMA" : 0>
    AIEX.multi_dest<%74, "DMA" : 0>
    AIEX.multi_dest<%63, "DMA" : 0>
    AIEX.multi_dest<%64, "DMA" : 0>
  }
 }
}