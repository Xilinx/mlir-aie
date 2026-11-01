//===- bad_create_scratchpad.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A runtime sequence holds at most one npu.create_scratchpad; the duplicate
// reports, and points at the first.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu2) {
  aie.runtime_sequence() {
    // expected-note@+1 {{previous 'aiex.npu.create_scratchpad' here}}
    aiex.npu.create_scratchpad {size = 8 : ui32}
    // expected-error@+1 {{only one 'aiex.npu.create_scratchpad' is allowed per runtime sequence}}
    aiex.npu.create_scratchpad {size = 8 : ui32}
  }
}

// -----

// The first may sit in a nested region.
aie.device(npu2) {
  aie.runtime_sequence() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1 step %c1 {
      // expected-note@+1 {{previous 'aiex.npu.create_scratchpad' here}}
      aiex.npu.create_scratchpad {size = 8 : ui32}
    }
    // expected-error@+1 {{only one 'aiex.npu.create_scratchpad' is allowed per runtime sequence}}
    aiex.npu.create_scratchpad {size = 8 : ui32}
  }
}

// -----

// One per sequence: two sequences may each hold one.
aie.device(npu2) {
  aie.runtime_sequence @a() {
    aiex.npu.create_scratchpad {size = 8 : ui32}
  }
  aie.runtime_sequence @b() {
    aiex.npu.create_scratchpad {size = 8 : ui32}
  }
}
