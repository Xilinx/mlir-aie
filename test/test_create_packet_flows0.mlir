// Note: This test *might* fail due to the random order that the code statements are generated

// RUN: aie-opt --aie-create-packet-flows %s | FileCheck %s

// CHECK-LABEL: module @test_create_packet_flows0 {
// CHECK:   %0 = AIE.tile(1, 1)
// CHECK:   %1 = AIE.switchbox(%0) {
// CHECK:     %2 = AIE.amsel<0> (0)
// CHECK:     %3 = AIE.masterset("ME" : 0, %2)
// CHECK:     %4 = AIE.amsel<0> (1)
// CHECK:     %5 = AIE.masterset("ME" : 1, %4)
// CHECK:     AIE.packetrules("West" : 0) {
// CHECK:       AIE.rule(31, 1, %4)
// CHECK:       AIE.rule(31, 0, %2)
// CHECK:     }
// CHECK:   }
// CHECK: }

// one-to-many, single arbiter
module @test_create_packet_flows0 {
  %t11 = AIE.tile(1, 1)

  AIE.packet_flow(0x0) {
    AIE.packet_source<%t11, "West" : 0>
    AIE.packet_dest<%t11, "ME" : 0>
  }

  AIE.packet_flow(0x1) {
    AIE.packet_source<%t11, "West" : 0>
    AIE.packet_dest<%t11, "ME" : 1>
  }
}
