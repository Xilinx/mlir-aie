//===- test_sa_cascade.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify SA placer satisfies cascade_flow adjacency constraints.
// cascade_flow(src, dst) requires either:
//   Horizontal: src at (col, row), dst at (col+1, row)
//   Vertical:   src at (col, row), dst at (col, row-1)

// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=42' %s | FileCheck %s

// Pin src, verify dst lands at a valid cascade position: (4,4) east or (3,3) south
// CHECK-LABEL: @cascade_pinned_src
module @cascade_pinned_src {
  aie.device(npu2) {
    %src = aie.logical_tile<CoreTile>(3, 4)
    %dst = aie.logical_tile<CoreTile>(?, ?)
    aie.cascade_flow(%src, %dst)
    aie.core(%src) { aie.end }
    aie.core(%dst) { aie.end }
    // CHECK-DAG: aie.tile(3, 4)
    // CHECK-DAG: aie.tile({{4, 4|3, 3}})
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Pin dst, verify src lands at a valid cascade position: (2,4) west or (3,5) north
// CHECK-LABEL: @cascade_pinned_dst
module @cascade_pinned_dst {
  aie.device(npu2) {
    %src = aie.logical_tile<CoreTile>(?, ?)
    %dst = aie.logical_tile<CoreTile>(3, 4)
    aie.cascade_flow(%src, %dst)
    aie.core(%src) { aie.end }
    aie.core(%dst) { aie.end }
    // CHECK-DAG: aie.tile(3, 4)
    // CHECK-DAG: aie.tile({{2, 4|3, 5}})
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Cascade pair with ObjectFifo I/O: verify adjacency still holds when
// HPWL cost from fifos competes with cascade penalty
// CHECK-LABEL: @cascade_with_fifo_pinned
module @cascade_with_fifo_pinned {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %src = aie.logical_tile<CoreTile>(2, 3)
    %dst = aie.logical_tile<CoreTile>(?, ?)

    aie.cascade_flow(%src, %dst)

    aie.objectfifo @in(%shim, {%src}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @out(%dst, {%shim}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.core(%src) { aie.end }
    aie.core(%dst) { aie.end }
    // src pinned at (2,3), dst must be at (3,3) east or (2,2) south
    // CHECK-DAG: aie.tile(2, 3)
    // CHECK-DAG: aie.tile({{3, 3|2, 2}})
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Two independent cascade pairs, both with pinned srcs:
// verify both pairs satisfy adjacency independently
// CHECK-LABEL: @two_cascade_pairs_pinned
module @two_cascade_pairs_pinned {
  aie.device(npu2) {
    %s1 = aie.logical_tile<CoreTile>(1, 4)
    %d1 = aie.logical_tile<CoreTile>(?, ?)
    %s2 = aie.logical_tile<CoreTile>(5, 3)
    %d2 = aie.logical_tile<CoreTile>(?, ?)

    aie.cascade_flow(%s1, %d1)
    aie.cascade_flow(%s2, %d2)

    aie.core(%s1) { aie.end }
    aie.core(%d1) { aie.end }
    aie.core(%s2) { aie.end }
    aie.core(%d2) { aie.end }
    // s1=(1,4) -> d1 at (2,4) or (1,3)
    // s2=(5,3) -> d2 at (6,3) or (5,2)
    // CHECK-DAG: aie.tile(1, 4)
    // CHECK-DAG: aie.tile({{2, 4|1, 3}})
    // CHECK-DAG: aie.tile(5, 3)
    // CHECK-DAG: aie.tile({{6, 3|5, 2}})
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}
