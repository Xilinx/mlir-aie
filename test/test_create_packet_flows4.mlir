// RUN: aie-opt --aie-create-packet-flows %s | FileCheck %s

// CHECK-LABEL: module @test_create_packet_flows4 {
// CHECK:   %0 = AIE.tile(1, 1)
// CHECK:   %1 = AIE.switchbox(%0) {
// CHECK:     %2 = AIE.amsel<0> (1)
// CHECK:     %3 = AIE.masterset("ME" : 1, %2)
// CHECK:     %4 = AIE.amsel<0> (0)
// CHECK:     %5 = AIE.masterset("ME" : 0, %4)
// CHECK:     AIE.packetrules("West" : 1) {
// CHECK:       AIE.rule(31, 0, %2)
// CHECK:     }
// CHECK:     AIE.packetrules("West" : 0) {
// CHECK:       AIE.rule(31, 1, %2)
// CHECK:       AIE.rule(31, 0, %4)
// CHECK:     }
// CHECK:   }
// CHECK: }

// many-to-many, 3 streams
module @test_create_packet_flows4 {
  %t11 = AIE.tile(1, 1)

  AIE.packet_flow(0x0) {
    AIE.packet_source<%t11, "West" : 0>
    AIE.packet_dest<%t11, "ME" : 0>
  }

  AIE.packet_flow(0x1) {
    AIE.packet_source<%t11, "West" : 0>
    AIE.packet_dest<%t11, "ME" : 1>
  }

  AIE.packet_flow(0x0) {
    AIE.packet_source<%t11, "West" : 1>
    AIE.packet_dest<%t11, "ME" : 1>
  }
}
