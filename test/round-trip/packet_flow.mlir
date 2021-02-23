// RUN: aie-opt %s | FileCheck %s

module @packet_flows {
// CHECK:  %[[VAL_0:.*]] = AIE.tile(1, 1)
// CHECK:  AIE.packet_flow(0) {
// CHECK:    AIE.packet_source<%[[VAL_0]], West : 0>
// CHECK:    AIE.packet_dest<%[[VAL_0]], Core : 0>
// CHECK:  }
  %t11 = AIE.tile(1, 1)
  AIE.packet_flow(0x0) {
    AIE.packet_source<%t11, West : 0>
    AIE.packet_dest<%t11, Core : 0>
  }
}
