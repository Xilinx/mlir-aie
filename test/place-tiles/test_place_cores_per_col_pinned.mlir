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
// core goes to the next column. Before, the budget kept the column's first
// rows for floating cores whatever the pinned core took.

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
