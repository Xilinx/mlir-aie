// RUN: aie-opt --aie-lower-broadcast-packet %s | FileCheck %s

// CHECK-LABEL: module @test_broadcast_packet {
// CHECK-NEXT:    %0 = AIE.tile(6, 3)
// CHECK-NEXT:    %1 = AIE.tile(6, 4)
// CHECK-NEXT:    %2 = AIE.tile(7, 2)
// CHECK-NEXT:    %3 = AIE.tile(7, 3)
// CHECK-NEXT:    %4 = AIE.tile(7, 4)
// CHECK-NEXT:    AIE.packet_flow(1) {
// CHECK-NEXT:      AIE.packet_source<%2, DMA : 0>
// CHECK-NEXT:      AIE.packet_dest<%4, DMA : 0>
// CHECK-NEXT:      AIE.packet_dest<%1, DMA : 0>
// CHECK-NEXT:    }
// CHECK-NEXT:    AIE.packet_flow(0) {
// CHECK-NEXT:      AIE.packet_source<%2, DMA : 0>
// CHECK-NEXT:      AIE.packet_dest<%3, DMA : 0>
// CHECK-NEXT:      AIE.packet_dest<%0, DMA : 0>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

//Sending data from tile DMA 0 of (7,2) to tile (6, 3), (6, 4), (7, 3), (7, 4)
//in which (6, 3) and (7, 3) are broadcasted so do (6, 4) and (7, 4). 
//Besides, (6, 3) and (7, 3) are in stream with ID 0,
//(6, 4) and (7, 4) are in stream with ID 1. That means that pair (6, 3), (7, 3)
//and pair (6, 4) and (7, 4) will time-multiplexed use tile DMA 0 of (7,2).
module @test_broadcast_packet {
  %t63 = AIE.tile(6, 3)
  %t64 = AIE.tile(6, 4)
  %t72 = AIE.tile(7, 2)
  %t73 = AIE.tile(7, 3)
  %t74 = AIE.tile(7, 4)
  AIE.broadcast_packet(%t72, "DMA" : 0){
    AIE.bp_id(0x0){
      AIE.bp_dest<%t73, "DMA" : 0>
      AIE.bp_dest<%t63, "DMA" : 0>
    }
    AIE.bp_id(0x1){
      AIE.bp_dest<%t74, "DMA" : 0>
      AIE.bp_dest<%t64, "DMA" : 0>
    }
  }
}