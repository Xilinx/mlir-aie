//===- unplaced_core_sections.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-assign-buffer-addresses %s

// A core on a logical tile with nothing to reserve for its own sections is
// left alone; one that has something to reserve gets a diagnostic.

module @nothing_to_reserve {
  aie.device(npu2_1col) {
    %lc = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%lc) {
      aie.end
    }
  }
}

// -----

module @data_region {
  aie.device(npu2_1col) {
    %lc = aie.logical_tile<CoreTile>(?, ?)
    // expected-error@+1 {{reserves memory for its own sections, but its tile is not placed; run --aie-place-tiles first}}
    aie.core(%lc) {
      aie.end
    } {data_size = 512 : i32}
  }
}
