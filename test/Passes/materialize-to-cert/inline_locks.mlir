//===- inline_locks.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences --convert-aie-to-transaction --aie-lower-set-lock --aie-npu-to-cert --split-input-file %s | FileCheck %s

// Test that lock SSA values are correctly inlined into the calling runtime
// sequence, and that multiple locks referencing the same tile are properly
// handled. This specifically tests the fix where argMap must be updated after
// cloning lock operations to ensure that subsequent uses of the lock reference
// the cloned lock rather than the original.

module {
  aie.device(npu2) {
    // The following SSA values should be inlined from the aiex.run call.
    // We should see one tile and multiple locks, each lock correctly
    // referencing the inlined tile.

    // CHECK: %[[TILE:.*]] = aie.tile(0, 2)
    // Locks are cloned in reverse order due to processing, but that's okay
    // CHECK-DAG: %[[LOCK2:.*]] = aie.lock(%[[TILE]], 2) {init = 0 : i32, sym_name = "lock_2"}
    // CHECK-DAG: %[[LOCK1:.*]] = aie.lock(%[[TILE]], 1) {init = 1 : i32, sym_name = "lock_1"}
    // CHECK-DAG: %[[LOCK0:.*]] = aie.lock(%[[TILE]], 0) {init = 0 : i32, sym_name = "lock_0"}

    // The main sequence is lowered to a bare cert.job (no enclosing page yet).
    // CHECK: aiex.cert.job
    aie.runtime_sequence(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
      // CHECK: aiex.cert.load_pdi(1, @callee_device)
      // The set_lock operations should be lowered to cert.write32 operations
      // CHECK-NEXT: aiex.cert.write32
      // CHECK-NEXT: aiex.cert.write32
      // CHECK-NEXT: aiex.cert.write32
      // CHECK-NEXT: aiex.cert.write32
      aiex.configure @callee_device {
        aiex.run @sequence(%arg0, %arg1) : (memref<64xi32>, memref<64xi32>)
      }
    }
    // CHECK: aiex.cert.section @callee_device
    // CHECK: aiex.cert.page
    // CHECK: aiex.cert.job
  }

  // The callee device should not be emitted (absorbed into main device)
  // CHECK-NOT: aie.device(npu2) @callee_device
  aie.device(npu2) @callee_device {
    %tile_0_2 = aie.tile(0, 2)

    %lock_0 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "lock_0"}
    %lock_1 = aie.lock(%tile_0_2, 1) {init = 1 : i32, sym_name = "lock_1"}
    %lock_2 = aie.lock(%tile_0_2, 2) {init = 0 : i32, sym_name = "lock_2"}

    aie.runtime_sequence (%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
      // Multiple set_lock operations with different locks and the same lock used multiple times
      aiex.set_lock(%lock_0, 1)
      aiex.set_lock(%lock_1, 0)
      aiex.set_lock(%lock_2, 1)
      // Use lock_0 again to verify the mapping is maintained
      aiex.set_lock(%lock_0, 2)
    }
  }
}

// -----

// Test with locks on multiple different tiles to ensure each lock correctly
// references its respective tile after inlining.

module {
  aie.device(npu2) {
    // CHECK: %[[TILE02:.*]] = aie.tile(0, 2)
    // CHECK: %[[TILE12:.*]] = aie.tile(1, 2)
    // CHECK-DAG: %[[LOCK12:.*]] = aie.lock(%[[TILE12]], 0)
    // CHECK-DAG: %[[LOCK02:.*]] = aie.lock(%[[TILE02]], 0)

    // The main sequence is lowered to a bare cert.job (no enclosing page yet).
    // CHECK: aiex.cert.job
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // CHECK: aiex.cert.load_pdi(1, @multi_tile_device)
      // The set_lock operations should be lowered to cert.write32 operations
      // CHECK-NEXT: aiex.cert.write32
      // CHECK-NEXT: aiex.cert.write32
      aiex.configure @multi_tile_device {
        aiex.run @sequence(%arg0) : (memref<64xi32>)
      }
    }
    // CHECK: aiex.cert.section @multi_tile_device
    // CHECK: aiex.cert.page
    // CHECK: aiex.cert.job
  }

  // The callee device should not be emitted (absorbed into main device)
  // CHECK-NOT: aie.device(npu2) @multi_tile_device
  aie.device(npu2) @multi_tile_device {
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)

    %lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 0) {init = 0 : i32}

    aie.runtime_sequence (%arg0: memref<64xi32>) {
      aiex.set_lock(%lock_0_2, 1)
      aiex.set_lock(%lock_1_2, 1)
    }
  }
}
