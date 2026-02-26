// RUN: aie-opt %s -split-input-file -verify-diagnostics

aie.device(npu1_1col) {
  %tile02 = aie.tile(0, 2)
  
  aie.trace @bad_combo_slot(%tile02) {
    // expected-error@+1 {{combo event slot must be 0, 1, or 2, got 5}}
    aie.trace.combo_event<5> <"LOCK_STALL"> AND <"DMA_S2MM_0_STALLED">
  }
}

// -----

aie.device(npu1_1col) {
  %tile02 = aie.tile(0, 2)
  
  aie.trace @duplicate_combo_slot(%tile02) {
    aie.trace.combo_event<0> <"LOCK_STALL"> AND <"DMA_S2MM_0_STALLED">
    // expected-error@+1 {{combo event slot 0 already in use in this trace}}
    aie.trace.combo_event<0> <"INSTR_EVENT_0"> OR <"INSTR_VECTOR">
  }
}

// -----

aie.device(npu1_1col) {
  %tile02 = aie.tile(0, 2)
  
  aie.trace @bad_hierarchical(%tile02) {
    // expected-error@+1 {{combo slot 2 first event must be COMBO_EVENT_0 (hierarchical), got LOCK_STALL}}
    aie.trace.combo_event<2> <"LOCK_STALL"> AND <"DMA_S2MM_0_STALLED">
  }
}

// -----

aie.device(npu1_1col) {
  %tile02 = aie.tile(0, 2)
  
  aie.trace @bad_edge_slot(%tile02) {
    // expected-error@+1 {{edge detection slot must be 0 or 1, got 5}}
    aie.trace.edge_event<5> event=<"LOCK_STALL"> trigger=RISING
  }
}

// -----

aie.device(npu1_1col) {
  %tile02 = aie.tile(0, 2)
  
  aie.trace @duplicate_edge_slot(%tile02) {
    aie.trace.edge_event<0> event=<"LOCK_STALL"> trigger=RISING
    // expected-error@+1 {{edge detection slot 0 already in use in this trace}}
    aie.trace.edge_event<0> event=<"DMA_MM2S_0_FINISHED_BD"> trigger=FALLING
  }
}

// -----

aie.device(npu1_1col) {
  %tile02 = aie.tile(0, 2)
  
  aie.trace @edge_of_edge(%tile02) {
    // expected-error@+1 {{edge detection source should be a regular event, not another EDGE_DETECTION_EVENT}}
    aie.trace.edge_event<0> event=<"EDGE_DETECTION_EVENT_0"> trigger=RISING
  }
}
