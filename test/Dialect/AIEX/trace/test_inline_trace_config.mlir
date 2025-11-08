// RUN: aie-opt %s -aie-trace-to-config -aiex-inline-trace-config | FileCheck %s

// CHECK-LABEL: module {
module {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    
    aie.trace @test_trace(%tile02) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type="core"
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    
    // CHECK: aie.trace.config @test_trace_config
    
    // Runtime sequence with trace configuration
    aiex.runtime_sequence @seq(%arg0: memref<32xi32>) {
      // CHECK: aiex.runtime_sequence
      aie.trace.start_config @test_trace
      
      // After inlining with aiex-inline-trace-config, npu.write32 is generated
      // CHECK-NOT: aie.trace.start_config
      // CHECK: aiex.npu.write32
      // CHECK-SAME: column = 0 : i32
      // CHECK-SAME: row = 2 : i32
    }
  }
}
