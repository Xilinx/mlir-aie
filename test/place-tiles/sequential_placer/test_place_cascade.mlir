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

// Unconstrained chain extends east along row 2 (south of row 2 is non-core).
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

// Pinned head at top of column 0 — chain extends south down the column.
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

// Pre-pinned compatible cascade — placer is a pass-through.
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

// -----

// Pinned peer appears in IR after unconstrained peer; %a's candidates are
// filtered against %b's pin coords (not just placed peers), so %a → (0,4).
// CHECK-LABEL: @cascade_pinned_anchor_after_unconstrained
module @cascade_pinned_anchor_after_unconstrained {
  aie.device(npu1) {
    // CHECK-DAG: %[[A:.*]] = aie.tile(0, 4)
    %a = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[B:.*]] = aie.tile(0, 3)
    %b = aie.logical_tile<CoreTile>(0, 3)
    aie.cascade_flow(%a, %b)
    aie.core(%a) { aie.end }
    aie.core(%b) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Hybrid cascade: TileOp src + LogicalTileOp dst. Dst adapts to TileOp's
// coords via the TileLike interface (would otherwise be silently dropped).
// CHECK-LABEL: @cascade_hybrid_tileop_peer
module @cascade_hybrid_tileop_peer {
  aie.device(npu1) {
    %src = aie.tile(1, 3)
    // CHECK-DAG: %[[D:.*]] = aie.tile(1, 2)
    %dst = aie.logical_tile<CoreTile>(?, ?)
    aie.cascade_flow(%src, %dst)
    aie.core(%src) { aie.end }
    aie.core(%dst) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Mid-chain pin: A→B→C with B pinned at (1,3). Both ends adapt: A west, C south.
// CHECK-LABEL: @cascade_chain_pinned_middle
module @cascade_chain_pinned_middle {
  aie.device(npu1) {
    // CHECK-DAG: %[[A:.*]] = aie.tile(0, 3)
    %a = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[B:.*]] = aie.tile(1, 3)
    %b = aie.logical_tile<CoreTile>(1, 3)
    // CHECK-DAG: %[[C:.*]] = aie.tile(1, 2)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.cascade_flow(%a, %b)
    aie.cascade_flow(%b, %c)
    aie.core(%a) { aie.end }
    aie.core(%b) { aie.end }
    aie.core(%c) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}
