// RUN: aie-opt --aie-create-packet-flows %s | FileCheck %s

// CHECK-LABEL: module @test_create_packet_flows5 {
// CHECK:   %0 = AIE.tile(1, 1)
// CHECK:   %1 = AIE.switchbox(%0) {
// CHECK:     %2 = AIE.amsel<0> (0)
// CHECK:     %3 = AIE.masterset("ME" : 0, %2)
// CHECK:     AIE.packetrules("West" : 1) {
// CHECK:       AIE.rule(31, 2, %2)
// CHECK:     }
// CHECK:     AIE.packetrules("West" : 0) {
// CHECK:       AIE.rule(30, 1, %2)
// CHECK:     }
// CHECK:   }
// CHECK: }

// many-to-one, 3 streams
module @test_create_packet_flows5 {
  %t11 = AIE.tile(1, 1)

  AIE.packet_flow(0x0) {
    AIE.packet_source<%t11, "West" : 0>
    AIE.packet_dest<%t11, "ME" : 0>
  }

  AIE.packet_flow(0x1) {
    AIE.packet_source<%t11, "West" : 0>
    AIE.packet_dest<%t11, "ME" : 0>
  }

  AIE.packet_flow(0x2) {
    AIE.packet_source<%t11, "West" : 1>
    AIE.packet_dest<%t11, "ME" : 0>
  }
}
