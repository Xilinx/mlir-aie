// RUN: aie-opt -aie-find-flows %s | FileCheck %s
// CHECK: %[[T23:.*]] = AIE.tile(2, 3)
// CHECK: %[[T22:.*]] = AIE.tile(2, 2)
// CHECK: AIE.flow(%[[T23]], Core : 0, %[[T22]], Core : 1)
// CHECK: AIE.flow(%[[T22]], Core : 0, %[[T22]], Core : 0)
// CHECK: AIE.flow(%[[T22]], Core : 1, %[[T23]], Core : 1)
// CHECK: AIE.packet_flow(0) {
// CHECK:   AIE.packet_source<%[[T22]], DMA : 0>
// CHECK:   AIE.packet_dest<%[[T23]], DMA : 1>
// CHECK: }
module {
  %tile0 = AIE.tile(2, 3)
  %tile1 = AIE.tile(2, 2)

  %0 = AIE.switchbox(%tile0) {
    AIE.connect<Core :0, South:1>
    AIE.connect<South:1, Core :1>
    AIE.connect<South:1, North:2>
    %16 = AIE.amsel<0> (0)
    %17 = AIE.masterset(DMA : 1, %16)
    AIE.packetrules(South : 0) {
      AIE.rule(31, 0, %16)
    }
  }
  %1 = AIE.switchbox(%tile1) {
    AIE.connect<North:1, Core :1>
    AIE.connect<Core :1, North:1>
    AIE.connect<Core :0, Core :0>
    %18 = AIE.amsel<0> (0)
    %19 = AIE.masterset(North : 0, %18)
    AIE.packetrules(DMA : 0) {
      AIE.rule(31, 0, %18)
    }
  }
  AIE.wire(%0: Core, %tile0: Core)
  AIE.wire(%1: Core, %tile1: Core)
  AIE.wire(%0: DMA, %tile0: DMA)
  AIE.wire(%1: DMA, %tile1: DMA)
  AIE.wire(%0: South, %1: North)
}
