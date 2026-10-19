//===- set_lock_logical_tile.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A set_lock on a lock whose tile is not placed yet verifies; its range and
// address checks run once aie-place-tiles places the tile, or right away on a
// logical tile with both coordinates pinned.
// set_lock_logical_tile_invalid.mlir shows an out-of-range id is still refused.

// RUN: aie-opt --split-input-file %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s --check-prefix=PLACED

// CHECK: aie.logical_tile<CoreTile>(?, ?)
// CHECK: aiex.set_lock
// PLACED: aie.tile
// PLACED: aiex.set_lock

module {
  aie.device(npu2) {
    %t = aie.logical_tile<CoreTile>(?, ?)
    %lock = aie.lock(%t, 0) {init = 0 : i32}
    aie.core(%t) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      aie.end
    }
    aie.runtime_sequence() {
      %v = arith.constant 1 : i32
      aiex.set_lock(%lock, %v)
    }
  }
}

// -----

// CHECK: aie.logical_tile<CoreTile>(0, 2)
// CHECK: aiex.set_lock
// PLACED: aie.tile(0, 2)
// PLACED: aiex.set_lock

module {
  aie.device(npu2) {
    %t = aie.logical_tile<CoreTile>(0, 2)
    %lock = aie.lock(%t, 15) {init = 0 : i32}
    aie.core(%t) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      aie.end
    }
    aie.runtime_sequence() {
      %v = arith.constant 1 : i32
      aiex.set_lock(%lock, %v)
    }
  }
}
