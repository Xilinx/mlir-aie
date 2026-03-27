//===- error_run_op.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file --aie-materialize-runtime-sequences %s

// Test that aiex.run ops referencing non-existent runtime sequences, sequences
// with wrong argument counts, or sequences with mismatched argument types are
// rejected by the AIEMaterializeRuntimeSequences pass. These checks are
// performed sequentially in the pass rather than in RunOp::verify() to avoid
// a data race: MLIR's pass manager runs verifiers on sibling DeviceOps
// concurrently, and cross-DeviceOp symbol table lookups from a verifier are
// unsafe under threading.

// -----

// Test: reference to a non-existent runtime sequence name.

module {
  // expected-note@+1 {{This device does not have a 'run_x' runtime sequence}}
  aie.device(npu1) @dev_a {
    aie.runtime_sequence @run_a (%x: memref<1xi32>) {
    }
    aie.runtime_sequence @run_b () {
    }
  }
  aie.device(npu1) {
    aie.runtime_sequence (%x : memref<1xi32>) {
      aiex.configure @dev_a {
        // expected-error@+1 {{No such runtime sequence for device 'dev_a': 'run_x'}}
        aiex.run @run_x (%x) : (memref<1xi32>)
      }
    }
  }
}

// -----

// Test: argument count mismatch — caller passes more args than the sequence expects.

module {
  aie.device(npu1) @dev_a {
    aie.runtime_sequence @run_a () {
    }
  }
  aie.device(npu1) {
    aie.runtime_sequence (%x : memref<1xi32>) {
      aiex.configure @dev_a {
        // expected-error@+1 {{argument count mismatch}}
        aiex.run @run_a (%x) : (memref<1xi32>)
      }
    }
  }
}

// -----

// Test: argument count mismatch — caller passes fewer args than the sequence expects.
// Note: aiex.run with zero args cannot be expressed in the assemblyFormat
// (MLIR parses ': ()' as a function type), so this test uses a 2-arg callee
// with a 1-arg caller.

module {
  aie.device(npu1) @dev_a {
    aie.runtime_sequence @run_a (%x: memref<1xi32>, %y: memref<2xi32>) {
    }
  }
  aie.device(npu1) {
    aie.runtime_sequence (%x : memref<1xi32>) {
      aiex.configure @dev_a {
        // expected-error@+1 {{argument count mismatch}}
        aiex.run @run_a (%x) : (memref<1xi32>)
      }
    }
  }
}

// -----

// Test: argument type mismatch — caller passes wrong element type.

module {
  aie.device(npu2) @callee_dev {
    aie.runtime_sequence @seq_expects_i16 (%arg0: memref<16xi16>) {
      aiex.npu.write32 {address = 100 : ui32, column = 1 : i32, row = 0 : i32, value = 42 : ui32}
    }
  }
  aie.device(npu2) {
    aie.runtime_sequence (%arg0: memref<16xi32>) {
      aiex.configure @callee_dev {
        // expected-error@+1 {{argument 0 type mismatch}}
        aiex.run @seq_expects_i16 (%arg0) : (memref<16xi32>)
      }
    }
  }
}

// -----

// Test: run references a symbol that exists in the callee device but is not a
// runtime sequence (it is a shim_dma_allocation).

module {
  aie.device(npu2) @dev_a {
    %t = aie.tile(2, 0)
    aie.shim_dma_allocation @buf(%t, S2MM, 0)
  }
  aie.device(npu2) {
    aie.runtime_sequence (%x : memref<1xi32>) {
      aiex.configure @dev_a {
        // expected-error@+1 {{'buf' is not a runtime sequence}}
        aiex.run @buf (%x) : (memref<1xi32>)
      }
    }
  }
}

// -----

// Test: argument type mismatch — caller passes wrong memref shape.

module {
  aie.device(npu1) @dev_a {
    aie.runtime_sequence @run_a (%x: memref<4xi32>) {
    }
  }
  aie.device(npu1) {
    aie.runtime_sequence (%x : memref<1xi32>) {
      aiex.configure @dev_a {
        // expected-error@+1 {{argument 0 type mismatch}}
        aiex.run @run_a (%x) : (memref<1xi32>)
      }
    }
  }
}
