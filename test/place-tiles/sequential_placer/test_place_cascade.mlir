//===- test_place_cascade.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// Cascade chain with no constraints — sequential greedy fill places head at
// (0,2); each subsequent tile in the chain must be south or east of its peer.
// South of (0,2) is row 1 which is not a CoreTile, so the chain extends east
// along row 2.
// CHECK-LABEL: @cascade_chain_unconstrained
module @cascade_chain_unconstrained {
  aie.device(npu1) {
    // CHECK-DAG: %[[A:.*]] = aie.tile(0, 2)
    %a = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[B:.*]] = aie.tile(1, 2)
    %b = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[C:.*]] = aie.tile(2, 2)
    %c = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[D:.*]] = aie.tile(3, 2)
    %d = aie.logical_tile<CoreTile>(?, ?)
    aie.cascade_flow(%a, %b)
    aie.cascade_flow(%b, %c)
    aie.cascade_flow(%c, %d)
    aie.core(%a) { aie.end }
    aie.core(%b) { aie.end }
    aie.core(%c) { aie.end }
    aie.core(%d) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Cascade chain with the head pinned to the top of column 0 — chain extends
// south down the same column.
// CHECK-LABEL: @cascade_chain_vertical
module @cascade_chain_vertical {
  aie.device(npu1) {
    // CHECK-DAG: %[[A:.*]] = aie.tile(0, 5)
    %a = aie.logical_tile<CoreTile>(0, 5)
    // CHECK-DAG: %[[B:.*]] = aie.tile(0, 4)
    %b = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[C:.*]] = aie.tile(0, 3)
    %c = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[D:.*]] = aie.tile(0, 2)
    %d = aie.logical_tile<CoreTile>(?, ?)
    aie.cascade_flow(%a, %b)
    aie.cascade_flow(%b, %c)
    aie.cascade_flow(%c, %d)
    aie.core(%a) { aie.end }
    aie.core(%b) { aie.end }
    aie.core(%c) { aie.end }
    aie.core(%d) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Cascade between two pre-pinned tiles that satisfy the constraint — placer
// is a pass-through, no errors.
// CHECK-LABEL: @cascade_pinned_compatible
module @cascade_pinned_compatible {
  aie.device(npu1) {
    // CHECK-DAG: %[[A:.*]] = aie.tile(2, 4)
    %a = aie.logical_tile<CoreTile>(2, 4)
    // CHECK-DAG: %[[B:.*]] = aie.tile(2, 3)
    %b = aie.logical_tile<CoreTile>(2, 3)
    aie.cascade_flow(%a, %b)
    aie.core(%a) { aie.end }
    aie.core(%b) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}
