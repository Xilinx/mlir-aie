//===- inline_locks.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences --split-input-file %s | FileCheck %s

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
    
    // CHECK: aie.runtime_sequence
    aie.runtime_sequence(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
      // CHECK: aiex.npu.load_pdi {device_ref = @callee_device}
      // The set_lock operations should reference the inlined locks, not the original ones
      // CHECK-NEXT: aiex.set_lock(%[[LOCK0]], 1)
      // CHECK-NEXT: aiex.set_lock(%[[LOCK1]], 0)
      // CHECK-NEXT: aiex.set_lock(%[[LOCK2]], 1)
      // CHECK-NEXT: aiex.set_lock(%[[LOCK0]], 2)
      aiex.configure @callee_device {
        aiex.run @sequence(%arg0, %arg1) : (memref<64xi32>, memref<64xi32>)
      }
    }
  }
  
  // CHECK: aie.device(npu2) @callee_device
  aie.device(npu2) @callee_device {
    // The original definitions should remain in the callee device
    // CHECK: aie.tile(0, 2)
    // CHECK-DAG: aie.lock({{.*}}, 0)
    // CHECK-DAG: aie.lock({{.*}}, 1)
    // CHECK-DAG: aie.lock({{.*}}, 2)
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
    
    // CHECK: aie.runtime_sequence
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // CHECK: aiex.npu.load_pdi {device_ref = @multi_tile_device}
      // CHECK-NEXT: aiex.set_lock(%[[LOCK02]], 1)
      // CHECK-NEXT: aiex.set_lock(%[[LOCK12]], 1)
      aiex.configure @multi_tile_device {
        aiex.run @sequence(%arg0) : (memref<64xi32>)
      }
    }
  }
  
  // CHECK: aie.device(npu2) @multi_tile_device
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
