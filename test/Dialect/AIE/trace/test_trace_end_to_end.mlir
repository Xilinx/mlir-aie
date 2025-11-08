// RUN: aie-opt %s -aie-trace-to-config -aie-inline-trace-config | FileCheck %s

// This test demonstrates the complete trace configuration pipeline
// from high-level aie.trace operations through to inlined register specifications

// CHECK-LABEL: module {
module {
  aie.device(npu1_1col) {
    // CHECK: %[[TILE:.*]] = aie.tile(0, 2)
    %tile02 = aie.tile(0, 2)
    
    // High-level trace configuration
    aie.trace @my_trace(%tile02) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type="core"
      aie.trace.event<"INSTR_EVENT_0"> label="kernel_start"
      aie.trace.event<"INSTR_EVENT_1"> label="kernel_end"
      aie.trace.event<"INSTR_VECTOR"> label="vector_op"
      aie.trace.event<"LOCK_STALL"> label="lock_stall"
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    
    // Invocation point (would be in runtime_sequence in real code)
    aie.trace.start_config @my_trace
    
    // After Pass 1 (TraceToConfig): aie.trace → aie.trace.config
    // CHECK: aie.trace.config @my_trace_config(%[[TILE]]) {
    
    // After Pass 2 (InlineTraceConfig): trace.start_config → inlined trace.reg
    // CHECK-NOT: aie.trace.start_config
    
    // Verify register writes are present (order may vary)
    // CHECK-DAG: aie.trace.reg register = "Trace_Control0" field = "Mode" value = 0 : i32
    // CHECK-DAG: aie.trace.reg register = "Trace_Control1" field = "ID" value = 1 : i32
    // CHECK-DAG: aie.trace.reg register = "Trace_Control1" field = "Packet_Type" value = 0 : i32
    // CHECK-DAG: aie.trace.reg register = "Trace_Control0" field = "Trace_Start_Event" value = 15 : i32
    // CHECK-DAG: aie.trace.reg register = "Trace_Control0" field = "Trace_Stop_Event" value = 14 : i32
    
    // Verify event slots
    // CHECK-DAG: aie.trace.reg register = "Trace_Event0" field = "Trace_Event0" value = "INSTR_EVENT_0"
    // CHECK-DAG: aie.trace.reg register = "Trace_Event0" field = "Trace_Event1" value = "INSTR_EVENT_1"
    // CHECK-DAG: aie.trace.reg register = "Trace_Event0" field = "Trace_Event2" value = "INSTR_VECTOR"
    // CHECK-DAG: aie.trace.reg register = "Trace_Event0" field = "Trace_Event3" value = "LOCK_STALL"
  }
}
