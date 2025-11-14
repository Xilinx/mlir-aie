// RUN: aie-opt %s -aie-trace-to-config | FileCheck %s

aie.device(npu1_1col) {
  %tile02 = aie.tile(0, 2)
  
  // CHECK-LABEL: @test_combo_lowering
  aie.trace @test_combo_lowering(%tile02) {
    aie.trace.combo_event<0> <"INSTR_EVENT_0"> AND <"INSTR_VECTOR">
    aie.trace.event<"COMBO_EVENT_0">
  }
  
  // CHECK: aie.trace.config @test_combo_lowering_config(%tile02)
  // CHECK:   aie.trace.reg register = "Combo_event_inputs" field = "eventA"
  // CHECK:   aie.trace.reg register = "Combo_event_inputs" field = "eventB"
  // CHECK:   aie.trace.reg register = "Combo_event_control" field = "combo0" value = 0
  // CHECK:   aie.trace.reg register = "Trace_Event0" field = "Trace_Event0"
}
