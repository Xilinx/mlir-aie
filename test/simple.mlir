// RUN: aie-opt %s | FileCheck %s
// CHECK-LABEL: module {
// CHECK:       }

module {
  %01 = AIE.tile(0, 1)
  %12 = AIE.tile(1, 2)
  %02 = AIE.tile(0, 2)
  AIE.flow(%01, "DMA" : 0, %12, "ME" : 1)
  AIE.packet_flow(0x10) {
    AIE.packet_source < %01, "ME" : 0>
    AIE.packet_dest < %12, "ME" : 0>
    AIE.packet_dest < %02, "DMA" : 1>
  }
}
