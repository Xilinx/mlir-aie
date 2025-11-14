// RUN: aie-opt %s | FileCheck %s

aie.device(npu1_1col) {
  %tile02 = aie.tile(0, 2)
  
  // CHECK-LABEL: @test_combo_basic
  aie.trace @test_combo_basic(%tile02) {
    // CHECK: aie.trace.combo_event<0> <"LOCK_STALL"> AND <"DMA_S2MM_0_STALLED">
    aie.trace.combo_event<0> <"LOCK_STALL"> AND <"DMA_S2MM_0_STALLED">
    
    // CHECK: aie.trace.combo_event<1> <"INSTR_EVENT_0"> OR <"INSTR_VECTOR">
    aie.trace.combo_event<1> <"INSTR_EVENT_0"> OR <"INSTR_VECTOR">
    
    // CHECK: aie.trace.event <"COMBO_EVENT_0">
    aie.trace.event<"COMBO_EVENT_0">
  }
  
  // CHECK-LABEL: @test_edge_basic
  aie.trace @test_edge_basic(%tile02) {
    // CHECK: aie.trace.edge_event<0> event = <"LOCK_STALL"> trigger = RISING
    aie.trace.edge_event<0> event=<"LOCK_STALL"> trigger=RISING
    
    // CHECK: aie.trace.edge_event<1> event = <"DMA_MM2S_0_FINISHED_BD"> trigger = BOTH
    aie.trace.edge_event<1> event=<"DMA_MM2S_0_FINISHED_BD"> trigger=BOTH
    
    // CHECK: aie.trace.event <"EDGE_DETECTION_EVENT_0">
    aie.trace.event<"EDGE_DETECTION_EVENT_0">
  }
  
  // CHECK-LABEL: @test_hierarchical
  aie.trace @test_hierarchical(%tile02) {
    // CHECK: aie.trace.combo_event<0> <"LOCK_STALL"> AND <"DMA_S2MM_0_STALLED">
    aie.trace.combo_event<0> <"LOCK_STALL"> AND <"DMA_S2MM_0_STALLED">
    
    // CHECK: aie.trace.combo_event<1> <"INSTR_EVENT_0"> OR <"INSTR_VECTOR">
    aie.trace.combo_event<1> <"INSTR_EVENT_0"> OR <"INSTR_VECTOR">
    
    // CHECK: aie.trace.combo_event<2> <"COMBO_EVENT_0"> AND <"COMBO_EVENT_1">
    aie.trace.combo_event<2> <"COMBO_EVENT_0"> AND <"COMBO_EVENT_1">
    
    // CHECK: aie.trace.event <"COMBO_EVENT_2">
    aie.trace.event<"COMBO_EVENT_2">
  }
  
  // CHECK-LABEL: @test_all_logic
  aie.trace @test_all_logic(%tile02) {
    // CHECK: aie.trace.combo_event<0> <"LOCK_STALL"> AND <"DMA_S2MM_0_STALLED">
    aie.trace.combo_event<0> <"LOCK_STALL"> AND <"DMA_S2MM_0_STALLED">
  }
  
  // CHECK-LABEL: @test_and_not
  aie.trace @test_and_not(%tile02) {
    // CHECK: aie.trace.combo_event<0> <"LOCK_STALL"> AND_NOT <"DMA_S2MM_0_STALLED">
    aie.trace.combo_event<0> <"LOCK_STALL"> AND_NOT <"DMA_S2MM_0_STALLED">
  }
  
  // CHECK-LABEL: @test_or
  aie.trace @test_or(%tile02) {
    // CHECK: aie.trace.combo_event<0> <"LOCK_STALL"> OR <"DMA_S2MM_0_STALLED">
    aie.trace.combo_event<0> <"LOCK_STALL"> OR <"DMA_S2MM_0_STALLED">
  }
  
  // CHECK-LABEL: @test_or_not
  aie.trace @test_or_not(%tile02) {
    // CHECK: aie.trace.combo_event<0> <"LOCK_STALL"> OR_NOT <"DMA_S2MM_0_STALLED">
    aie.trace.combo_event<0> <"LOCK_STALL"> OR_NOT <"DMA_S2MM_0_STALLED">
  }
  
  // CHECK-LABEL: @test_rising
  aie.trace @test_rising(%tile02) {
    // CHECK: aie.trace.edge_event<0> event = <"LOCK_STALL"> trigger = RISING
    aie.trace.edge_event<0> event=<"LOCK_STALL"> trigger=RISING
  }
  
  // CHECK-LABEL: @test_falling
  aie.trace @test_falling(%tile02) {
    // CHECK: aie.trace.edge_event<0> event = <"LOCK_STALL"> trigger = FALLING
    aie.trace.edge_event<0> event=<"LOCK_STALL"> trigger=FALLING
  }
  
  // CHECK-LABEL: @test_both
  aie.trace @test_both(%tile02) {
    // CHECK: aie.trace.edge_event<0> event = <"LOCK_STALL"> trigger = BOTH
    aie.trace.edge_event<0> event=<"LOCK_STALL"> trigger=BOTH
  }
}
