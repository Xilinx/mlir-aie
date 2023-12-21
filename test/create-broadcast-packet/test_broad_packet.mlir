// RUN: aie-opt --aie-lower-broadcast-packet %s | FileCheck %s
// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(6, 3)
// CHECK:           %[[VAL_1:.*]] = aie.tile(6, 4)
// CHECK:           %[[VAL_2:.*]] = aie.tile(7, 2)
// CHECK:           %[[VAL_3:.*]] = aie.tile(7, 3)
// CHECK:           %[[VAL_4:.*]] = aie.tile(7, 4)
// CHECK:           aie.packet_flow(1) {
// CHECK:             aie.packet_source<%[[VAL_2]], DMA : 0>
// CHECK:             aie.packet_dest<%[[VAL_4]], DMA : 0>
// CHECK:             aie.packet_dest<%[[VAL_1]], DMA : 0>
// CHECK:           }
// CHECK:           aie.packet_flow(0) {
// CHECK:             aie.packet_source<%[[VAL_2]], DMA : 0>
// CHECK:             aie.packet_dest<%[[VAL_3]], DMA : 0>
// CHECK:             aie.packet_dest<%[[VAL_0]], DMA : 0>
// CHECK:           }
// CHECK:         }

//Sending data from tile DMA 0 of (7,2) to tile (6, 3), (6, 4), (7, 3), (7, 4)
//in which (6, 3) and (7, 3) are broadcasted so do (6, 4) and (7, 4).
//Besides, (6, 3) and (7, 3) are in stream with ID 0,
//(6, 4) and (7, 4) are in stream with ID 1. That means that pair (6, 3), (7, 3)
//and pair (6, 4) and (7, 4) will time-multiplexed use tile DMA 0 of (7,2).
module @test_broadcast_packet {
 aie.device(xcvc1902) {
  %t63 = aie.tile(6, 3)
  %t64 = aie.tile(6, 4)
  %t72 = aie.tile(7, 2)
  %t73 = aie.tile(7, 3)
  %t74 = aie.tile(7, 4)
  aiex.broadcast_packet(%t72, "DMA" : 0){
    aiex.bp_id(0x0){
      aiex.bp_dest<%t73, "DMA" : 0>
      aiex.bp_dest<%t63, "DMA" : 0>
    }
    aiex.bp_id(0x1){
      aiex.bp_dest<%t74, "DMA" : 0>
      aiex.bp_dest<%t64, "DMA" : 0>
    }
  }
 }
}