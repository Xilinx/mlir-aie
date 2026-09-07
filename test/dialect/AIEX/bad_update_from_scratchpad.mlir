//===- bad_update_from_scratchpad.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// npu.update_from_scratchpad's state_table_idx must index within the
// npu.create_scratchpad in its block, and the diagnostic names the update op.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu2) {
  aie.runtime_sequence() {
    aiex.npu.create_scratchpad {size = 8 : ui32}
    // expected-error@+1 {{state_table_idx 2 is out of bounds for scratchpad of size 8 bytes (2 entries) created by aiex.npu.create_scratchpad.}}
    aiex.npu.update_from_scratchpad {address = 0 : ui32, state_table_idx = 2 : ui8}
  }
}

// -----

// The last in-bounds index.
aie.device(npu2) {
  aie.runtime_sequence() {
    aiex.npu.create_scratchpad {size = 8 : ui32}
    aiex.npu.update_from_scratchpad {address = 0 : ui32, state_table_idx = 1 : ui8}
  }
}

// -----

// With no scratchpad in the block there is nothing to bound the update.
aie.device(npu2) {
  aie.runtime_sequence() {
    aiex.npu.update_from_scratchpad {address = 0 : ui32, state_table_idx = 31 : ui8}
  }
}
