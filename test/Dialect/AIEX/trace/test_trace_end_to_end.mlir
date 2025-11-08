// RUN: aie-opt %s -aie-trace-to-config -aiex-inline-trace-config | FileCheck %s

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
    
    // After Pass 2 (AIEXInlineTraceConfig): generates npu.write32 with col/row
    // CHECK-NOT: aie.trace.start_config
    
    // Verify npu.write32 is generated with col/row preserved
    // CHECK: aiex.npu.write32
    // CHECK-SAME: column = 0 : i32
    // CHECK-SAME: row = 2 : i32
  }
}
