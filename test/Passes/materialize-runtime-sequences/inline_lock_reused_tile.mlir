//===- inline_lock_reused_tile.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences --split-input-file %s | FileCheck %s

// A callee lock is inlined into a caller device that ALREADY declares the lock's
// tile (the reuse path: the tile is not cloned, the existing one is reused).
// The cloned lock must be inserted AFTER the reused tile so it is dominated by
// its tile operand. Before the fix, the lock was cloned at the device-body start
// -- ahead of the reused tile -- producing "operand #0 does not dominate this
// use". The sequential CHECKs below pin lock-after-tile.

// CHECK-LABEL: aie.device(npu2) {
// CHECK: %[[TILE:.*]] = aie.tile(0, 2)
// CHECK: %[[LOCK:.*]] = aie.lock(%[[TILE]], 0) {{.*}}sym_name = "lk0"
// CHECK: aie.runtime_sequence
// CHECK: aiex.set_lock(%[[LOCK]], 1)

module {
  aie.device(npu2) {
    // The caller device already declares tile(0, 2), so materialization reuses
    // it instead of cloning a fresh copy.
    %tile_0_2 = aie.tile(0, 2)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      aiex.configure @callee {
        aiex.run @sequence(%arg0) : (memref<64xi32>)
      }
    }
  }

  aie.device(npu2) @callee {
    %tile_0_2 = aie.tile(0, 2)
    %lock_0 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "lk0"}
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      aiex.set_lock(%lock_0, 1)
    }
  }
}
