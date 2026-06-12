// RUN: aie-opt %s -aie-trace-to-config -aie-trace-pack-reg-writes -aie-inline-trace-config | FileCheck %s

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
    
    aie.runtime_sequence @seq(%arg0: memref<32xi32>) {
      aie.trace.start_config @port_trace
    }
  }
}

// CHECK: aie.runtime_sequence @seq
// CHECK: %[[A0:.+]] = arith.constant 213200 : i32
// CHECK: aiex.npu.write32(%[[A0]],
// CHECK: %[[A1:.+]] = arith.constant 213204 : i32
// CHECK: aiex.npu.write32(%[[A1]],
// CHECK: %[[A2:.+]] = arith.constant 261888 : i32
// CHECK: aiex.npu.write32(%[[A2]],
// CHECK: %[[A3:.+]] = arith.constant 213216 : i32
// CHECK: aiex.npu.write32(%[[A3]],
