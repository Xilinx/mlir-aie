// RUN: aie-opt %s | FileCheck %s
// CHECK-LABEL: module {
// CHECK:       }

module {
  %00 = AIE.tile(0, 0)
  %11 = AIE.tile(1, 1)
  %01 = AIE.tile(0, 1)
  AIE.flow(%00, "DMA" : 0, %11, "ME" : 1)
  AIE.packet_flow(0x10) {
    AIE.packet_source < %00, "ME" : 0>
    AIE.packet_dest < %11, "ME" : 0>
    AIE.packet_dest < %01, "DMA" : 1>
  }
}
