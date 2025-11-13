// RUN: aie-opt %s -aie-trace-to-config -aie-trace-pack-reg-writes -aiex-inline-trace-config | FileCheck %s

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    
    aie.trace @port_trace(%tile_0_2) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type="core"
      aie.trace.port<0> port=North channel=1 direction=S2MM
      aie.trace.event<"PORT_RUNNING_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    
    aiex.runtime_sequence @seq(%arg0: memref<32xi32>) {
      aie.trace.start_config @port_trace
    }
  }
}

// CHECK: aiex.runtime_sequence @seq
// CHECK: aiex.npu.write32 {address = 213200
// CHECK: aiex.npu.write32 {address = 213204
// CHECK: aiex.npu.write32 {address = 261888
// CHECK: aiex.npu.write32 {address = 213216
