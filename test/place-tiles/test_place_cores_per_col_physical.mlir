//===- test_place_cores_per_col_physical.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="cores-per-col=1" %s | FileCheck %s --check-prefix=ONE
// RUN: aie-opt --split-input-file --aie-place-tiles="cores-per-col=2" %s | FileCheck %s --check-prefix=TWO

// A physical aie.tile's own core counts against its column's cores-per-col
// budget, so column 0 holds no more than the budget.

// ONE-LABEL: module @physical_core_low
// ONE: aie.core(%tile_0_2)
// ONE: aie.core(%tile_1_2)
// ONE: aie.core(%tile_2_2)
// ONE: aie.core(%tile_3_2)
// TWO-LABEL: module @physical_core_low
// TWO: aie.core(%tile_0_2)
// TWO: aie.core(%tile_0_3)
// TWO: aie.core(%tile_1_2)
// TWO: aie.core(%tile_1_3)
module @physical_core_low {
  aie.device(npu2) {
    %p = aie.tile(0, 2)
    %l1 = aie.logical_tile<CoreTile>(?, ?)
    %l2 = aie.logical_tile<CoreTile>(?, ?)
    %l3 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%p) { aie.end }
    aie.core(%l1) { aie.end }
    aie.core(%l2) { aie.end }
    aie.core(%l3) { aie.end }
  }
}

// -----

// ONE-LABEL: module @physical_core_high
// ONE: aie.core(%tile_0_5)
// ONE: aie.core(%tile_1_2)
// ONE: aie.core(%tile_2_2)
// ONE: aie.core(%tile_3_2)
// TWO-LABEL: module @physical_core_high
// TWO: aie.core(%tile_0_5)
// TWO: aie.core(%tile_0_2)
// TWO: aie.core(%tile_1_2)
// TWO: aie.core(%tile_1_3)
module @physical_core_high {
  aie.device(npu2) {
    %p = aie.tile(0, 5)
    %l1 = aie.logical_tile<CoreTile>(?, ?)
    %l2 = aie.logical_tile<CoreTile>(?, ?)
    %l3 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%p) { aie.end }
    aie.core(%l1) { aie.end }
    aie.core(%l2) { aie.end }
    aie.core(%l3) { aie.end }
  }
}
