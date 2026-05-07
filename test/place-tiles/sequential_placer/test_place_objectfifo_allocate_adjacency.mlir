//===- test_place_objectfifo_allocate_adjacency.mlir ----------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// `aie.objectfifo.allocate @of (%delegate)` documents that every endpoint of
// `@of` (producer + each consumer) shares L1 with the delegate. Today the
// objectfifo lowering pass silently degrades to a DMA fifo when this
// contract is violated; the placer catches it earlier and steers
// unconstrained endpoints to satisfy it.

// Delegate pinned at (2, 3); producer and consumer are unconstrained. Greedy
// column-major would place them both at column 0; the allocate-adjacency
// check rejects that and steers each endpoint into a memory-affinity
// neighbor of the delegate.
// CHECK-LABEL: @of_allocate_steers_endpoints
module @of_allocate_steers_endpoints {
  aie.device(npu1) {
    // CHECK-DAG: aie.tile(2, 3)
    // CHECK-DAG: aie.tile(2, 2)
    // CHECK-DAG: aie.tile(2, 4)
    %delegate = aie.logical_tile<CoreTile>(2, 3)
    %producer = aie.logical_tile<CoreTile>(?, ?)
    %consumer = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%delegate) { aie.end }
    aie.core(%producer) { aie.end }
    aie.core(%consumer) { aie.end }
    aie.objectfifo @of_steers (%producer, {%consumer}, 2 : i32)
        : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.allocate @of_steers (%delegate)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// All three endpoints pinned at compatible positions: delegate at (0, 3),
// producer at (0, 2) (S of delegate), consumer at (0, 4) (N of delegate).
// Placer is a pass-through.
// CHECK-LABEL: @of_allocate_all_pinned_compatible
module @of_allocate_all_pinned_compatible {
  aie.device(npu1) {
    // CHECK-DAG: aie.tile(0, 2)
    // CHECK-DAG: aie.tile(0, 3)
    // CHECK-DAG: aie.tile(0, 4)
    %delegate = aie.logical_tile<CoreTile>(0, 3)
    %producer = aie.logical_tile<CoreTile>(0, 2)
    %consumer = aie.logical_tile<CoreTile>(0, 4)
    aie.core(%delegate) { aie.end }
    aie.core(%producer) { aie.end }
    aie.core(%consumer) { aie.end }
    aie.objectfifo @of_pinned (%producer, {%consumer}, 2 : i32)
        : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.allocate @of_pinned (%delegate)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Delegate is the producer itself: no cross-tile edge for the producer
// (self-reference); consumer still gets steered into a memory-affinity slot.
// CHECK-LABEL: @of_allocate_self_delegate
module @of_allocate_self_delegate {
  aie.device(npu1) {
    // CHECK-DAG: aie.tile(2, 3)
    // CHECK-DAG: aie.tile(2, 2)
    %producer = aie.logical_tile<CoreTile>(2, 3)
    %consumer = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%producer) { aie.end }
    aie.core(%consumer) { aie.end }
    aie.objectfifo @of_self (%producer, {%consumer}, 2 : i32)
        : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.allocate @of_self (%producer)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// AIE2P (npu2) target-model dispatch: same affinity rules.
// CHECK-LABEL: @of_allocate_npu2_target_model
module @of_allocate_npu2_target_model {
  aie.device(npu2) {
    // CHECK-DAG: aie.tile(2, 3)
    // CHECK-DAG: aie.tile(2, 2)
    // CHECK-DAG: aie.tile(2, 4)
    %delegate = aie.logical_tile<CoreTile>(2, 3)
    %producer = aie.logical_tile<CoreTile>(?, ?)
    %consumer = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%delegate) { aie.end }
    aie.core(%producer) { aie.end }
    aie.core(%consumer) { aie.end }
    aie.objectfifo @of_npu2 (%producer, {%consumer}, 2 : i32)
        : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.allocate @of_npu2 (%delegate)
    // CHECK-NOT: aie.logical_tile
  }
}
