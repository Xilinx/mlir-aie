//===- test_place_cores_per_col_pinned_diag.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-place-tiles="cores-per-col=1" %s 2>&1 | FileCheck %s
// RUN: not aie-opt --aie-place-tiles %s 2>&1 | FileCheck %s

// Core %c is pinned to column 0 and reads a buffer on (0, 5), but (0, 3) and
// (0, 2), the column's free tiles, are not (0, 5)'s neighbours. A column pin
// outranks cores-per-col, so the error gives the causes of searching past the
// budget, as without one, and does not blame the budget. Before, at
// cores-per-col=1 it dropped the adjacency cause and its note and said the
// budget may be limiting placement.
// CHECK-NOT: cores-per-col
// CHECK: error: no compute tile available matching constraint (0, ?) and shared-L1 buffer adjacency
// CHECK-NOT: cores-per-col
// CHECK: note: shared-L1 buffer owner peer placed at (0, 5)
// CHECK-NOT: cores-per-col

module @diag {
  aie.device(npu2_1col) {
    %a = aie.logical_tile<CoreTile>(0, 5)
    %b = aie.logical_tile<CoreTile>(0, 4)
    %c = aie.logical_tile<CoreTile>(0, ?)
    %buf = aie.buffer(%a) {sym_name = "shared"} : memref<16xi32>
    aie.core(%a) { aie.end }
    aie.core(%b) { aie.end }
    aie.core(%c) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
  }
}
