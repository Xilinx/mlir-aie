//===- set_lock_logical_tile_invalid.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// set_lock skips its range check on an unplaced tile; the lock's own verifier
// refuses an out-of-range id there first.

// RUN: aie-opt --verify-diagnostics %s

module {
  aie.device(npu2) {
    %t = aie.logical_tile<CoreTile>(?, ?)
    // expected-error @+1 {{lock assigned invalid id (maximum is 15)}}
    %lock = aie.lock(%t, 16) {init = 0 : i32}
    aie.runtime_sequence() {
      %v = arith.constant 1 : i32
      aiex.set_lock(%lock, %v)
    }
  }
}
