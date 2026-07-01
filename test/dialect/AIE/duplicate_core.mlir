//===- duplicate_core.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DeviceOp::verify rejects two cores resolving to the same (col, row): a
// compute tile has exactly one core in hardware. The check fires for physical
// aie.tile coordinates, for pinned aie.logical_tile coordinates (pre-placement),
// and again on the physical tiles produced by --aie-place-tiles. Cores on
// unplaced logical tiles are skipped until placement assigns coordinates.

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s

// Two physical cores on the same tile.
// CHECK: error{{.*}}'aie.core' op tile (0, 2) already has a core; each compute tile can host only one core
module @phys_dup {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.core(%t) { aie.end }
    aie.core(%t) { aie.end }
  }
}

// -----

// Two pinned logical-tile cores at the same coordinate, caught before placement.
// CHECK: error{{.*}}'aie.core' op tile (0, 2) already has a core
module @logical_dup {
  aie.device(npu2) {
    %c1 = aie.logical_tile<CoreTile>(0, 2)
    %c2 = aie.logical_tile<CoreTile>(0, 2)
    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
  }
}

// -----

// Two physical cores referencing two distinct aie.tile ops at the same
// coordinate (e.g. produced by a frontend that did not merge by coordinate).
// CHECK: error{{.*}}'aie.core' op tile (1, 3) already has a core
module @phys_dup_distinct_tiles {
  aie.device(npu2) {
    %t1 = aie.tile(1, 3)
    %t2 = aie.tile(1, 3)
    aie.core(%t1) { aie.end }
    aie.core(%t2) { aie.end }
  }
}
