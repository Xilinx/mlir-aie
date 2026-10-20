//===- bad_scratchpad_wait_parameters.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// aiex.wait_scratchpad_parameters belongs in a core that keeps its sync
// preamble, and once a core has one, each of its parameter reads must come
// after a wait, or the read could see a previous dispatch's value.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-lower-scratchpad-parameters %s

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  aie.runtime_sequence(%arg0 : memref<16xi32>) {
    // expected-error @+1 {{'aiex.wait_scratchpad_parameters' op must be inside an aie.core}}
    aiex.wait_scratchpad_parameters
  }
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // expected-error @+1 {{'aiex.read_scratchpad_parameter' op reads @n where no aiex.wait_scratchpad_parameters of its core comes first, so it may read a previous dispatch's value}}
    %early = aiex.read_scratchpad_parameter @n : i32
    scf.for %i = %c0 to %c4 step %c1 {
      aiex.wait_scratchpad_parameters
      %v = aiex.read_scratchpad_parameter @n : i32
    }
    aie.end
  }
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      aiex.wait_scratchpad_parameters
    }
    // expected-error @+1 {{'aiex.read_scratchpad_parameter' op reads @n where no aiex.wait_scratchpad_parameters of its core comes first, so it may read a previous dispatch's value}}
    %late = aiex.read_scratchpad_parameter @n : i32
    aie.end
  }
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t02 = aie.tile(0, 2)
  aie.core(%t02) {
    // expected-error @+1 {{'aiex.wait_scratchpad_parameters' op waits in a core with emit_parameter_sync_preamble = false, which gets no lock to wait on}}
    aiex.wait_scratchpad_parameters
    aie.end
  } {emit_parameter_sync_preamble = false}
}
