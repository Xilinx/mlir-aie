// RUN: aie-opt %s -aie-trace-to-config | FileCheck %s

aie.device(npu1_1col) {
  %tile02 = aie.tile(0, 2)
  
  // CHECK-LABEL: @test_combo_and_edge
  aie.trace @test_combo_and_edge(%tile02) {
    aie.trace.mode "Event-Time"
    aie.trace.packet id=1 type=core
    
    // Combo event (using events we know exist from previous tests)
    aie.trace.combo_event<0> <"INSTR_EVENT_0"> AND_NOT <"INSTR_VECTOR">
    
    // Edge detection
    aie.trace.edge_event<0> event=<"LOCK_STALL"> trigger=BOTH
    
    // Trace the derived events
    aie.trace.event<"COMBO_EVENT_0">
    aie.trace.event<"EDGE_DETECTION_EVENT_0">
    aie.trace.event<"INSTR_EVENT_0">
    
    aie.trace.start broadcast=15
    aie.trace.stop broadcast=14
  }
  
  // Check combo event configuration
  // CHECK: aie.trace.reg register = "Combo_event_inputs" field = "eventA"
  // CHECK: aie.trace.reg register = "Combo_event_inputs" field = "eventB"
  // CHECK: aie.trace.reg register = "Combo_event_control" field = "combo0" value = 1
  // Check edge detection configuration
  // CHECK: aie.trace.reg register = "Edge_Detection_event_control" field = "Edge_Detection_Event_0"
  // CHECK: aie.trace.reg register = "Edge_Detection_event_control" field = "Edge_Detection_0_Trigger_Rising" value = 1
  // CHECK: aie.trace.reg register = "Edge_Detection_event_control" field = "Edge_Detection_0_Trigger_Falling" value = 1
  // Check trace control
  // CHECK: aie.trace.reg register = "Trace_Control0" field = "Mode" value = 0
  // CHECK: aie.trace.reg register = "Trace_Control1" field = "ID" value = 1
  // CHECK: aie.trace.reg register = "Trace_Control0" field = "Trace_Start_Event" value = 15
  // CHECK: aie.trace.reg register = "Trace_Control0" field = "Trace_Stop_Event" value = 14
  // Check event slots
  // CHECK: aie.trace.reg register = "Trace_Event0" field = "Trace_Event0"
  // CHECK: aie.trace.reg register = "Trace_Event0" field = "Trace_Event1"
  // CHECK: aie.trace.reg register = "Trace_Event0" field = "Trace_Event2"
}
