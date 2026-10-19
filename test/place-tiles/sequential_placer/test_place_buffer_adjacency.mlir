//===- test_place_buffer_adjacency.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// -----

// An objectfifo.allocate puts the fifo's objects in the delegate's memory, so
// the unconstrained producer is steered next to the delegate pinned at (2, 3)
// instead of the greedy (0, 2).
// CHECK-LABEL: @adjacency_allocate_delegate
module @adjacency_allocate_delegate {
  aie.device(npu2) {
    // CHECK-DAG: aie.tile(2, 3)
    // CHECK-DAG: aie.tile(2, 2)
    %delegate = aie.logical_tile<CoreTile>(2, 3)
    %producer = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @of (%producer, {%delegate}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.allocate @of (%delegate)
    aie.core(%delegate) { aie.end }
    aie.core(%producer) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// A consumer is placed right after the buffer's owner, before unrelated
// cores take the owner's neighbors: in one column, the owner at (0, 2)
// leaves (0, 3) to its consumer and the other cores go above.
// CHECK-LABEL: @adjacency_consumer_follows_owner
module @adjacency_consumer_follows_owner {
  aie.device(npu2_1col) {
    // CHECK: %[[OWNER:.*]] = aie.tile(0, 2)
    // CHECK: aie.buffer(%[[OWNER]])
    // CHECK: aie.core(%[[OWNER]])
    // CHECK: aie.core(%tile_0_4)
    // CHECK: aie.core(%tile_0_5)
    // CHECK: aie.core(%tile_0_3)
    // CHECK-NEXT: arith.constant
    // CHECK-NEXT: memref.load %buffer_0_2
    %owner = aie.logical_tile<CoreTile>(?, ?)
    %buf = aie.buffer(%owner) : memref<16xi32>
    aie.core(%owner) { aie.end }
    %f0 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%f0) { aie.end }
    %f1 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%f1) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%consumer) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
  }
}

// -----

// An allocate only ties core tiles: a fifo's MemTile end puts no shared-L1
// constraint on the delegate, which takes the greedy (0, 2).
// CHECK-LABEL: @adjacency_allocate_ignores_memtile_end
module @adjacency_allocate_ignores_memtile_end {
  aie.device(npu2) {
    // CHECK-DAG: aie.tile(0, 1)
    // CHECK-DAG: aie.tile(0, 2)
    %mem = aie.logical_tile<MemTile>(0, 1)
    %delegate = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @of (%mem, {%delegate}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.allocate @of (%delegate)
    aie.core(%delegate) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// The owner skips (0, 2): a consumer there would need a free tile beside it,
// and the pinned core at (0, 3) takes the only one. Placing the owner at
// (0, 4) leaves (0, 5) for its consumer.
// CHECK-LABEL: @owner_leaves_room_for_consumer
// CHECK-DAG: aie.tile(0, 5)
// CHECK-DAG: %[[OWNER:.*]] = aie.tile(0, 4)
// CHECK-DAG: aie.buffer(%[[OWNER]])
module @owner_leaves_room_for_consumer {
  aie.device(npu2_1col) {
    %pin = aie.logical_tile<CoreTile>(0, 3)
    aie.core(%pin) { aie.end }
    %owner = aie.logical_tile<CoreTile>(?, ?)
    %buf = aie.buffer(%owner) : memref<16xi32>
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%consumer) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
  }
}

// -----

// Three consumers of one owner each need their own free tile beside it. At
// (0, 2) the owner has only two core neighbours, so it goes to (0, 3).
// CHECK-LABEL: @owner_leaves_room_for_three_consumers
// CHECK-DAG: aie.tile(0, 2)
// CHECK-DAG: aie.tile(0, 4)
// CHECK-DAG: aie.tile(1, 3)
// CHECK-DAG: %[[OWNER:.*]] = aie.tile(0, 3)
// CHECK-DAG: aie.buffer(%[[OWNER]])
module @owner_leaves_room_for_three_consumers {
  aie.device(npu2) {
    %owner = aie.logical_tile<CoreTile>(?, ?)
    %buf = aie.buffer(%owner) : memref<16xi32>
    aie.core(%owner) { aie.end }
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%c0) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%c1) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%c2) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
  }
}

// -----

// In one column, two consumers need the owner's two vertical neighbours, so
// the owner skips (0, 2) and takes (0, 3).
// CHECK-LABEL: @owner_leaves_room_for_two_consumers
// CHECK-DAG: aie.tile(0, 2)
// CHECK-DAG: aie.tile(0, 4)
// CHECK-DAG: %[[OWNER:.*]] = aie.tile(0, 3)
// CHECK-DAG: aie.buffer(%[[OWNER]])
module @owner_leaves_room_for_two_consumers {
  aie.device(npu2_1col) {
    %owner = aie.logical_tile<CoreTile>(?, ?)
    %buf = aie.buffer(%owner) : memref<16xi32>
    aie.core(%owner) { aie.end }
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%c0) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%c1) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
  }
}
