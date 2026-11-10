//===- test_place_cores_per_col_note.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-place-tiles="cores-per-col=2" %s 2>&1 | FileCheck %s

// The cores-per-col note counts the compute tiles left to unpinned cores,
// after the physical and pinned cores in each column take their share of the
// budget. Before, it printed columns x cores-per-col ("leaves 2").

// CHECK: error: no available compute tiles for placement
// CHECK: note: cores-per-col=2 leaves 0 of this device's 4 compute tiles placeable
module @two_physical {
  aie.device(npu2_1col) {
    %p0 = aie.tile(0, 2)
    %p1 = aie.tile(0, 3)
    %l1 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%p0) { aie.end }
    aie.core(%p1) { aie.end }
    aie.core(%l1) { aie.end }
  }
}

// -----

// CHECK: error: no available compute tiles for placement
// CHECK: note: cores-per-col=2 leaves 0 of this device's 4 compute tiles placeable
module @physical_and_pinned {
  aie.device(npu2_1col) {
    %p0 = aie.tile(0, 2)
    %l0 = aie.logical_tile<CoreTile>(0, 5)
    %l1 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%p0) { aie.end }
    aie.core(%l0) { aie.end }
    aie.core(%l1) { aie.end }
  }
}
