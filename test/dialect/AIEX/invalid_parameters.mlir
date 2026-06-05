// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Verify that read_scratchpad_parameter must be inside aie.core.

aiex.scratchpad_parameter @foo : i32
aie.device(npu2) {
  aie.runtime_sequence() {
    // expected-error @+1 {{'aiex.read_scratchpad_parameter' op must be inside an aie.core}}
    %x = aiex.read_scratchpad_parameter @foo : i32
    aie.end
  }
}

// -----

// Verify that read_scratchpad_parameter rejects unknown parameter references.

aie.device(npu2) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    // expected-error @+1 {{'aiex.read_scratchpad_parameter' op references unknown parameter 'nonexistent'}}
    %x = aiex.read_scratchpad_parameter @nonexistent : i32
    aie.end
  }
}

// -----

// Verify that read_scratchpad_parameter rejects f32 result type.

aiex.scratchpad_parameter @foo : f32
aie.device(npu2) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    // expected-error @+1 {{'aiex.read_scratchpad_parameter' op f32 parameters are not supported}}
    %x = aiex.read_scratchpad_parameter @foo : f32
    aie.end
  }
}
