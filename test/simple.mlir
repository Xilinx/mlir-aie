// RUN: aie-opt --aie-create-flows --aie-find-flows %s | FileCheck %s
// CHECK: %[[T01:.*]] = AIE.tile(0, 1)
// CHECK: %[[T12:.*]] = AIE.tile(1, 2)
// CHECK: AIE.flow(%[[T01]], DMA : 0, %[[T12]], Core : 1)

module {
  %01 = AIE.tile(0, 1)
  %12 = AIE.tile(1, 2)
  %02 = AIE.tile(0, 2)
  AIE.flow(%01, DMA : 0, %12, Core : 1)
  AIE.packet_flow(0x10) {
    AIE.packet_source < %01, Core : 0>
    AIE.packet_dest < %12, Core : 0>
    AIE.packet_dest < %02, DMA : 1>
  }
}
