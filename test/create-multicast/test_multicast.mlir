// RUN: aie-opt --aie-lower-multicast %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(7, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(7, 3)
// CHECK:           %[[VAL_2:.*]] = aie.tile(7, 4)
// CHECK:           %[[VAL_3:.*]] = aie.tile(6, 3)
// CHECK:           %[[VAL_4:.*]] = aie.tile(6, 4)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:         }

module @test_multicast {
 aie.device(xcvc1902) {
  %70 = aie.tile(7, 0)
  %73 = aie.tile(7, 3)
  %74 = aie.tile(7, 4)
  %63 = aie.tile(6, 3)
  %64 = aie.tile(6, 4)
  aiex.multicast(%70, "DMA" : 0){
    aiex.multi_dest<%73, "DMA" : 0>
    aiex.multi_dest<%74, "DMA" : 0>
    aiex.multi_dest<%63, "DMA" : 0>
    aiex.multi_dest<%64, "DMA" : 0>
  }
 }
}