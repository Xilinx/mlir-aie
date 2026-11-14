//===- test_place_cores_per_col_pinned.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="cores-per-col=2" %s | FileCheck %s --check-prefix=TWO
// RUN: aie-opt --split-input-file --aie-place-tiles="cores-per-col=1" %s | FileCheck %s --check-prefix=ONE

// A pinned logical core counts against its column's cores-per-col budget as
// a physical core does, wherever in the column it is pinned, so the floating
// core goes to the next column.

// TWO-LABEL: module @physical_and_pinned
// TWO: aie.core(%tile_0_2)
// TWO: aie.core(%tile_0_4)
// TWO: aie.core(%tile_1_2)
module @physical_and_pinned {
  aie.device(npu2) {
    %p0 = aie.tile(0, 2)
    %l1 = aie.logical_tile<CoreTile>(0, 4)
    %l2 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%p0) { aie.end }
    aie.core(%l1) { aie.end }
    aie.core(%l2) { aie.end }
  }
}

// -----

// ONE-LABEL: module @pinned_high
// ONE: aie.core(%tile_0_4)
// ONE: aie.core(%tile_1_2)
module @pinned_high {
  aie.device(npu2) {
    %l1 = aie.logical_tile<CoreTile>(0, 4)
    %l2 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%l1) { aie.end }
    aie.core(%l2) { aie.end }
  }
}

// -----

// A pinned logical CoreTile that runs no core holds no core of the budget,
// as a physical tile without one does not.
// ONE-LABEL: module @pinned_without_core
// ONE: aie.core(%tile_0_2)
module @pinned_without_core {
  aie.device(npu2_1col) {
    %l1 = aie.logical_tile<CoreTile>(0, 4)
    %b = aie.buffer(%l1) {sym_name = "hold"} : memref<16xi32>
    %l2 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%l2) { aie.end }
  }
}

// -----

// Explicit pins outrank the budget: a core pinned only by column, or only
// by row, takes a tile the budget left out when none within it matches, as a
// fully pinned core already may.
// ONE-LABEL: module @column_pinned
// ONE: aie.core(%tile_0_5)
// ONE: aie.core(%tile_0_2)
module @column_pinned {
  aie.device(npu2) {
    %l0 = aie.logical_tile<CoreTile>(0, 5)
    %l1 = aie.logical_tile<CoreTile>(0, ?)
    aie.core(%l0) { aie.end }
    aie.core(%l1) { aie.end }
  }
}

// -----

// ONE-LABEL: module @row_pinned
// ONE: aie.core(%tile_0_4)
// ONE: aie.core(%tile_0_2)
module @row_pinned {
  aie.device(npu2) {
    %l1 = aie.logical_tile<CoreTile>(?, 4)
    %l2 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%l1) { aie.end }
    aie.core(%l2) { aie.end }
  }
}

// -----

// A pinned core takes a tile past the budget that passes every check before
// it takes, in the last, relaxed pass, the budget's one tile that its floating
// shared-L1 peer needs.
// ONE-LABEL: module @column_pinned_with_floating_peer
// ONE: aie.core(%tile_0_3)
// ONE: aie.core(%tile_0_2)
module @column_pinned_with_floating_peer {
  aie.device(npu2_1col) {
    %c = aie.logical_tile<CoreTile>(0, ?)
    %f = aie.logical_tile<CoreTile>(?, ?)
    %buf = aie.buffer(%c) {sym_name = "shared"} : memref<16xi32>
    aie.core(%c) { aie.end }
    aie.core(%f) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
  }
}
