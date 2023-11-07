// RUN: aie-opt --aie-lower-broadcast-packet %s | FileCheck %s
// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(6, 3)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(6, 4)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(7, 2)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(7, 3)
// CHECK:           %[[VAL_4:.*]] = AIE.tile(7, 4)
// CHECK:           AIE.packet_flow(1) {
// CHECK:             AIE.packet_source<%[[VAL_2]], DMA : 0>
// CHECK:             AIE.packet_dest<%[[VAL_4]], DMA : 0>
// CHECK:             AIE.packet_dest<%[[VAL_1]], DMA : 0>
// CHECK:           }
// CHECK:           AIE.packet_flow(0) {
// CHECK:             AIE.packet_source<%[[VAL_2]], DMA : 0>
// CHECK:             AIE.packet_dest<%[[VAL_3]], DMA : 0>
// CHECK:             AIE.packet_dest<%[[VAL_0]], DMA : 0>
// CHECK:           }
// CHECK:         }

//Sending data from tile DMA 0 of (7,2) to tile (6, 3), (6, 4), (7, 3), (7, 4)
//in which (6, 3) and (7, 3) are broadcasted so do (6, 4) and (7, 4).
//Besides, (6, 3) and (7, 3) are in stream with ID 0,
//(6, 4) and (7, 4) are in stream with ID 1. That means that pair (6, 3), (7, 3)
//and pair (6, 4) and (7, 4) will time-multiplexed use tile DMA 0 of (7,2).
module @test_broadcast_packet {
 AIE.device(xcvc1902) {
  %t63 = AIE.tile(6, 3)
  %t64 = AIE.tile(6, 4)
  %t72 = AIE.tile(7, 2)
  %t73 = AIE.tile(7, 3)
  %t74 = AIE.tile(7, 4)
  AIEX.broadcast_packet(%t72, "DMA" : 0){
    AIEX.bp_id(0x0){
      AIEX.bp_dest<%t73, "DMA" : 0>
      AIEX.bp_dest<%t63, "DMA" : 0>
    }
    AIEX.bp_id(0x1){
      AIEX.bp_dest<%t74, "DMA" : 0>
      AIEX.bp_dest<%t64, "DMA" : 0>
    }
  }
 }
}