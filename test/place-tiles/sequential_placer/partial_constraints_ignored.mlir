//===- partial_constraints_ignored.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// This test demonstrates that partial constraints (col-only or row-only)
// are ignored by the current placer. A future placer implementation may
// honor partial constraints, at which point this test should be removed.

// Test: Column-only constraint is ignored for CoreTiles
// CHECK-LABEL: @partial_constraint_col_ignored
module @partial_constraint_col_ignored {
  aie.device(npu1) {
    // Tile constrained to column 1, but placed at (0, 2) due to sequential placement
    // CHECK: %[[TILE:.*]] = aie.tile(0, 2)
    %tile = aie.logical_tile<CoreTile>(1, ?)
    // CHECK: aie.core(%[[TILE]])
    aie.core(%tile) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Test: Row-only constraint is ignored for CoreTiles
// CHECK-LABEL: @partial_constraint_row_ignored
module @partial_constraint_row_ignored {
  aie.device(npu1) {
    // Tile constrained to row 3, but placed at (0, 2) due to sequential placement
    // CHECK: %[[TILE:.*]] = aie.tile(0, 2)
    %tile = aie.logical_tile<CoreTile>(?, 3)
    // CHECK: aie.core(%[[TILE]])
    aie.core(%tile) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Test: Column constraint ignored for MemTiles (uses commonCol instead)
// CHECK-LABEL: @partial_constraint_memtile_col
module @partial_constraint_memtile_col {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)
    // MemTile constrained to column 2, but placed at (0, 1) using commonCol
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem = aie.logical_tile<MemTile>(2, ?)
    // CHECK: aie.objectfifo @of1(%[[MEM]], {%[[CORE]]}, 2 : i32)
    aie.objectfifo @of1 (%mem, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // CHECK-NOT: aie.logical_tile
  }
}
