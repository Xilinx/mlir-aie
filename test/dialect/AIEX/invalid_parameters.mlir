// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Verify that read_parameter must be inside aie.core.

aiex.parameter @foo : i32
aie.device(npu2) {
  aie.runtime_sequence() {
    // expected-error @+1 {{'aiex.read_parameter' op must be inside an aie.core}}
    %x = aiex.read_parameter @foo : i32
    aie.end
  }
}

// -----

// Verify that read_parameter rejects unknown parameter references.

aie.device(npu2) {
  %t = aie.tile(0, 2)
  aie.core(%t) {
    // expected-error @+1 {{'aiex.read_parameter' op references unknown parameter 'nonexistent'}}
    %x = aiex.read_parameter @nonexistent : i32
    aie.end
  }
}
