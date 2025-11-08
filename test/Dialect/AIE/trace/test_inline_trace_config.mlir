// RUN: aie-opt %s -aie-trace-to-config -aie-inline-trace-config | FileCheck %s

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
    
    // Simulate a runtime sequence with trace.start_config
    // CHECK: aie.trace.config @test_trace_config
    // CHECK-NOT: aie.trace.start_config
    aie.trace.start_config @test_trace
    
    // After inlining, the trace.reg ops should appear here
    // CHECK-DAG: aie.trace.reg register = "Trace_Control0" field = "Mode"
    // CHECK-DAG: aie.trace.reg register = "Trace_Control1" field = "ID"
    // CHECK-DAG: aie.trace.reg register = "Trace_Control0" field = "Trace_Start_Event"
    // CHECK-DAG: aie.trace.reg register = "Trace_Control0" field = "Trace_Stop_Event"
    // CHECK-DAG: aie.trace.reg register = "Trace_Event0" field = "Trace_Event0"
  }
}
