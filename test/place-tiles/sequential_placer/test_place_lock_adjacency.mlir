//===- test_place_lock_adjacency.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// Owner pinned at (2, 3); unconstrained consumer acquires its lock. Greedy
// column-major would place the consumer at (0, 2); the lock-affinity check
// rejects that and steers to (2, 2) (north of owner -- first affinity slot
// in greedy sweep order). Same predicate as buffer adjacency, since AIE2
// locks live in L1 alongside buffers.
// CHECK-LABEL: @lock_adjacency_steers_unconstrained_consumer
module @lock_adjacency_steers_unconstrained_consumer {
  aie.device(npu1) {
    // CHECK-DAG: aie.tile(2, 3)
    // CHECK-DAG: aie.tile(2, 2)
    %owner = aie.logical_tile<CoreTile>(2, 3)
    %lock  = aie.lock(%owner, 0) {init = 1 : i32}
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%consumer) {
      aie.use_lock(%lock, AcquireGreaterEqual, 1)
      aie.use_lock(%lock, Release, 1)
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Both endpoints pinned at compatible positions: owner at (0, 2), consumer
// at (0, 3) (consumer's S-affinity slot). Placer is a pass-through.
// CHECK-LABEL: @lock_adjacency_both_pinned_compatible
module @lock_adjacency_both_pinned_compatible {
  aie.device(npu1) {
    // CHECK-DAG: aie.tile(0, 2)
    // CHECK-DAG: aie.tile(0, 3)
    %owner = aie.logical_tile<CoreTile>(0, 2)
    %lock  = aie.lock(%owner, 0) {init = 1 : i32}
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(0, 3)
    aie.core(%consumer) {
      aie.use_lock(%lock, AcquireGreaterEqual, 1)
      aie.use_lock(%lock, Release, 1)
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Self-reference: consumer acquires a lock attached to its own LTO. Not a
// cross-tile constraint; placer behaves as if no adjacency edge existed.
// CHECK-LABEL: @lock_adjacency_self_reference_no_constraint
module @lock_adjacency_self_reference_no_constraint {
  aie.device(npu1) {
    // CHECK-DAG: aie.tile(0, 2)
    %lt = aie.logical_tile<CoreTile>(?, ?)
    %lock = aie.lock(%lt, 0) {init = 1 : i32}
    aie.core(%lt) {
      aie.use_lock(%lock, AcquireGreaterEqual, 1)
      aie.use_lock(%lock, Release, 1)
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Mixed buffer + lock from the same owner: both edges agree, consumer is
// steered to a memory-affinity neighbor of the owner.
// CHECK-LABEL: @lock_and_buffer_agree
module @lock_and_buffer_agree {
  aie.device(npu1) {
    // CHECK-DAG: aie.tile(2, 3)
    // CHECK-DAG: aie.tile(2, 2)
    %owner = aie.logical_tile<CoreTile>(2, 3)
    %buf   = aie.buffer(%owner) : memref<16xi32>
    %lock  = aie.lock(%owner, 0) {init = 1 : i32}
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%consumer) {
      aie.use_lock(%lock, AcquireGreaterEqual, 1)
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.use_lock(%lock, Release, 1)
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// AIE2P (npu2) target-model dispatch: same affinity rules apply.
// CHECK-LABEL: @lock_adjacency_npu2_target_model
module @lock_adjacency_npu2_target_model {
  aie.device(npu2) {
    // CHECK-DAG: aie.tile(2, 3)
    // CHECK-DAG: aie.tile(2, 2)
    %owner = aie.logical_tile<CoreTile>(2, 3)
    %lock  = aie.lock(%owner, 0) {init = 1 : i32}
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%consumer) {
      aie.use_lock(%lock, AcquireGreaterEqual, 1)
      aie.use_lock(%lock, Release, 1)
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}
