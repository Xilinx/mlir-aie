// RUN: aie-opt %s -aie-trace-to-config | FileCheck %s

aie.device(npu1_1col) {
  %tile02 = aie.tile(0, 2)
  
  // CHECK-LABEL: @test_edge_lowering
  aie.trace @test_edge_lowering(%tile02) {
    aie.trace.edge_event<0> event=<"LOCK_STALL"> trigger=RISING
    aie.trace.event<"EDGE_DETECTION_EVENT_0">
  }
  
  // CHECK: aie.trace.config @test_edge_lowering_config(%tile02)
  // CHECK:   aie.trace.reg register = "Edge_Detection_event_control" field = "Edge_Detection_Event_0"
  // CHECK:   aie.trace.reg register = "Edge_Detection_event_control" field = "Edge_Detection_0_Trigger_Rising" value = 1
  // CHECK:   aie.trace.reg register = "Edge_Detection_event_control" field = "Edge_Detection_0_Trigger_Falling" value = 0
  // CHECK:   aie.trace.reg register = "Trace_Event0" field = "Trace_Event0"
}
