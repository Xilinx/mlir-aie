//===- set_lock_logical_tile.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A set_lock on a lock whose tile is not placed yet verifies; its lock-ID
// range is checked once aie-place-tiles places the tile.

// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles %s | FileCheck %s --check-prefix=PLACED

// CHECK: aie.logical_tile<CoreTile>
// CHECK: aiex.set_lock
// PLACED: aie.tile
// PLACED: aiex.set_lock

module {
  aie.device(npu2) {
    %t = aie.logical_tile<CoreTile>(?, ?)
    %lock = aie.lock(%t, 0) {init = 0 : i32}
    aie.core(%t) {
      aie.use_lock(%lock, AcquireGreaterEqual, 1)
      aie.end
    }
    aie.runtime_sequence() {
      %v = arith.constant 1 : i32
      aiex.set_lock(%lock, %v)
    }
  }
}
