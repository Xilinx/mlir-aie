//===- test_place_buffer_adjacency.mlir ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// Owner pinned at (2, 3); unconstrained consumer reads its buffer. Greedy
// column-major would place the consumer at (0, 2); the adjacency check
// rejects that and steers to (2, 2) (north of owner — first affinity slot
// in greedy sweep order).
// CHECK-LABEL: @adjacency_steers_unconstrained_consumer
module @adjacency_steers_unconstrained_consumer {
  aie.device(npu1) {
    // CHECK-DAG: aie.tile(2, 3)
    // CHECK-DAG: aie.tile(2, 2)
    %owner = aie.logical_tile<CoreTile>(2, 3)
    %buf   = aie.buffer(%owner) : memref<16xi32>
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%consumer) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Star: pinned owner at (1, 3) plus three unconstrained consumers. Each
// consumer must be a memory-affinity neighbor of (1, 3): S (1, 2), N (1, 4),
// or W-of-buffer (2, 3). All three slots are filled.
// CHECK-LABEL: @adjacency_star_three_consumers
module @adjacency_star_three_consumers {
  aie.device(npu1) {
    // CHECK-DAG: aie.tile(1, 3)
    // CHECK-DAG: aie.tile(1, 2)
    // CHECK-DAG: aie.tile(1, 4)
    // CHECK-DAG: aie.tile(2, 3)
    %owner = aie.logical_tile<CoreTile>(1, 3)
    %buf   = aie.buffer(%owner) : memref<16xi32>
    aie.core(%owner) { aie.end }
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%c1) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    aie.core(%c2) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    aie.core(%c3) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Both endpoints pinned at compatible positions: owner at (0, 2), consumer
// at (0, 3) (consumer's S-affinity slot). Placer is a pass-through.
// CHECK-LABEL: @adjacency_both_pinned_compatible
module @adjacency_both_pinned_compatible {
  aie.device(npu1) {
    // CHECK-DAG: aie.tile(0, 2)
    // CHECK-DAG: aie.tile(0, 3)
    %owner = aie.logical_tile<CoreTile>(0, 2)
    %buf   = aie.buffer(%owner) : memref<16xi32>
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(0, 3)
    aie.core(%consumer) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Buffer accessed via memref.subview is still traced back to the owning
// LogicalTileOp. Adjacency steers the consumer just as in the direct case.
// CHECK-LABEL: @adjacency_through_view_op
module @adjacency_through_view_op {
  aie.device(npu1) {
    // CHECK-DAG: aie.tile(2, 3)
    // CHECK-DAG: aie.tile(2, 2)
    %owner = aie.logical_tile<CoreTile>(2, 3)
    %buf   = aie.buffer(%owner) : memref<16xi32>
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%consumer) {
      %sub = memref.subview %buf[0][8][1] : memref<16xi32> to memref<8xi32, strided<[1]>>
      %i = arith.constant 0 : index
      %v = memref.load %sub[%i] : memref<8xi32, strided<[1]>>
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Self-reference: consumer reads a buffer attached to its own LTO. Not a
// cross-tile constraint; placer behaves as if no adjacency edge existed.
// CHECK-LABEL: @adjacency_self_reference_no_constraint
module @adjacency_self_reference_no_constraint {
  aie.device(npu1) {
    // CHECK-DAG: aie.tile(0, 2)
    %lt = aie.logical_tile<CoreTile>(?, ?)
    %buf = aie.buffer(%lt) : memref<16xi32>
    aie.core(%lt) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// AIE2P (npu2) target-model dispatch: same affinity rules apply.
// CHECK-LABEL: @adjacency_npu2_target_model
module @adjacency_npu2_target_model {
  aie.device(npu2) {
    // CHECK-DAG: aie.tile(2, 3)
    // CHECK-DAG: aie.tile(2, 2)
    %owner = aie.logical_tile<CoreTile>(2, 3)
    %buf   = aie.buffer(%owner) : memref<16xi32>
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%consumer) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}
