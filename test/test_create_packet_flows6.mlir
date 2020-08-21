// RUN: aie-opt --aie-create-packet-flows %s | FileCheck %s

// CHECK-LABEL: module @test_create_packet_flows6 {
// CHECK:   %0 = AIE.tile(2, 2)
// CHECK:   %1 = AIE.switchbox(%0) {
// CHECK:     %10 = AIE.amsel<0> (0)
// CHECK:     %11 = AIE.masterset("East" : 0, %10)
// CHECK:     AIE.packetrules("DMA" : 0) {
// CHECK:       AIE.rule(28, 3, %10)
// CHECK:     }
// CHECK:   }
// CHECK:   %2 = AIE.tile(3, 2)
// CHECK:   %3 = AIE.switchbox(%2) {
// CHECK:     %10 = AIE.amsel<0> (0)
// CHECK:     %11 = AIE.masterset("East" : 0, %10)
// CHECK:     %12 = AIE.amsel<0> (1)
// CHECK:     %13 = AIE.masterset("DMA" : 0, %12)
// CHECK:     AIE.packetrules("West" : 0) {
// CHECK:       AIE.rule(28, 3, %10)
// CHECK:       AIE.rule(31, 0, %12)
// CHECK:     }
// CHECK:   }
// CHECK:   %4 = AIE.tile(4, 2)
// CHECK:   %5 = AIE.switchbox(%4) {
// CHECK:     %10 = AIE.amsel<0> (1)
// CHECK:     %11 = AIE.masterset("DMA" : 0, %10)
// CHECK:     %12 = AIE.amsel<0> (0)
// CHECK:     %13 = AIE.masterset("East" : 0, %12)
// CHECK:     AIE.packetrules("West" : 0) {
// CHECK:       AIE.rule(30, 3, %12)
// CHECK:       AIE.rule(31, 1, %10)
// CHECK:     }
// CHECK:   }
// CHECK:   %6 = AIE.tile(5, 2)
// CHECK:   %7 = AIE.switchbox(%6) {
// CHECK:     %10 = AIE.amsel<0> (0)
// CHECK:     %11 = AIE.masterset("DMA" : 0, %10)
// CHECK:     %12 = AIE.amsel<0> (1)
// CHECK:     %13 = AIE.masterset("East" : 0, %12)
// CHECK:     AIE.packetrules("West" : 0) {
// CHECK:       AIE.rule(31, 3, %12)
// CHECK:       AIE.rule(31, 2, %10)
// CHECK:     }
// CHECK:   }
// CHECK:   %8 = AIE.tile(6, 2)
// CHECK:   %9 = AIE.switchbox(%8) {
// CHECK:     %10 = AIE.amsel<0> (0)
// CHECK:     %11 = AIE.masterset("DMA" : 0, %10)
// CHECK:     AIE.packetrules("West" : 0) {
// CHECK:       AIE.rule(31, 3, %10)
// CHECK:     }
// CHECK:   }
// CHECK: }

module @test_create_packet_flows6 {

  %tile22 = AIE.tile(2, 2)
  %tile32 = AIE.tile(3, 2)
  %tile42 = AIE.tile(4, 2)
  %tile52 = AIE.tile(5, 2)
  %tile62 = AIE.tile(6, 2)

  // [2, 2] --> [3, 2] --> [4, 2] --> [5, 2] --> [6, 2]

  AIE.packet_flow(0x0) {
    AIE.packet_source<%tile22, "DMA" : 0>
    AIE.packet_dest<%tile32, "DMA" : 0>
  }

  AIE.packet_flow(0x1) {
    AIE.packet_source<%tile22, "DMA" : 0>
    AIE.packet_dest<%tile42, "DMA" : 0>
  }

  AIE.packet_flow(0x2) {
    AIE.packet_source<%tile22, "DMA" : 0>
    AIE.packet_dest<%tile52, "DMA" : 0>
  }

  AIE.packet_flow(0x3) {
    AIE.packet_source<%tile22, "DMA" : 0>
    AIE.packet_dest<%tile62, "DMA" : 0>
  }
}
