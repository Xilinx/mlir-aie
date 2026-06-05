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

// Single cascade pair: must be adjacent
// CHECK-LABEL: @single_cascade_pair
module @single_cascade_pair {
  aie.device(npu2) {
    %src = aie.logical_tile<CoreTile>(?, ?)
    %dst = aie.logical_tile<CoreTile>(?, ?)
    aie.cascade_flow(%src, %dst)
    aie.core(%src) { aie.end }
    aie.core(%dst) { aie.end }
    // CHECK-NOT: aie.logical_tile
    // CHECK: aie.cascade_flow
    aie.end
  }
}

// -----

// Cascade pair with ObjectFifo connections
// CHECK-LABEL: @cascade_with_fifo
module @cascade_with_fifo {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %src = aie.logical_tile<CoreTile>(?, ?)
    %dst = aie.logical_tile<CoreTile>(?, ?)

    aie.cascade_flow(%src, %dst)

    aie.objectfifo @in(%shim, {%src}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @out(%dst, {%shim}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.core(%src) { aie.end }
    aie.core(%dst) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Two independent cascade pairs
// CHECK-LABEL: @two_cascade_pairs
module @two_cascade_pairs {
  aie.device(npu2) {
    %s1 = aie.logical_tile<CoreTile>(?, ?)
    %d1 = aie.logical_tile<CoreTile>(?, ?)
    %s2 = aie.logical_tile<CoreTile>(?, ?)
    %d2 = aie.logical_tile<CoreTile>(?, ?)

    aie.cascade_flow(%s1, %d1)
    aie.cascade_flow(%s2, %d2)

    aie.core(%s1) { aie.end }
    aie.core(%d1) { aie.end }
    aie.core(%s2) { aie.end }
    aie.core(%d2) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Cascade pair coexisting with unconstrained cores
// CHECK-LABEL: @cascade_with_unconstrained
module @cascade_with_unconstrained {
  aie.device(npu2) {
    %src = aie.logical_tile<CoreTile>(?, ?)
    %dst = aie.logical_tile<CoreTile>(?, ?)
    %free1 = aie.logical_tile<CoreTile>(?, ?)
    %free2 = aie.logical_tile<CoreTile>(?, ?)

    aie.cascade_flow(%src, %dst)

    // Pipeline through all cores
    aie.objectfifo @f1(%free1, {%src}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @f2(%dst, {%free2}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.core(%src) { aie.end }
    aie.core(%dst) { aie.end }
    aie.core(%free1) { aie.end }
    aie.core(%free2) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Four cascade pairs (mobilenet-like: bn13 L1, bn13 L3, bn14 L1, bn14 L3)
// CHECK-LABEL: @four_cascade_pairs
module @four_cascade_pairs {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<CoreTile>(?, ?)
    %d1 = aie.logical_tile<CoreTile>(?, ?)
    %s2 = aie.logical_tile<CoreTile>(?, ?)
    %d2 = aie.logical_tile<CoreTile>(?, ?)
    %s3 = aie.logical_tile<CoreTile>(?, ?)
    %d3 = aie.logical_tile<CoreTile>(?, ?)
    %s4 = aie.logical_tile<CoreTile>(?, ?)
    %d4 = aie.logical_tile<CoreTile>(?, ?)
    %mid = aie.logical_tile<CoreTile>(?, ?)

    aie.cascade_flow(%s1, %d1)
    aie.cascade_flow(%s2, %d2)
    aie.cascade_flow(%s3, %d3)
    aie.cascade_flow(%s4, %d4)

    // Chain: shim -> s1/d1 -> mid -> s2/d2 -> ...
    aie.objectfifo @in(%shim, {%s1}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @c1(%d1, {%mid}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @c2(%mid, {%s3, %d3}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @out(%d4, {%shim}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.core(%s1) { aie.end }
    aie.core(%d1) { aie.end }
    aie.core(%s2) { aie.end }
    aie.core(%d2) { aie.end }
    aie.core(%s3) { aie.end }
    aie.core(%d3) { aie.end }
    aie.core(%s4) { aie.end }
    aie.core(%d4) { aie.end }
    aie.core(%mid) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}
