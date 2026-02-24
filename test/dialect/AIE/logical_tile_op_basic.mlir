//===- logical_tile_op_basic.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file %s | FileCheck %s

// Test LogicalTileOp parsing with all tile types and placement variations

// CHECK-LABEL: @test_logical_tile
// CHECK: %[[CORE_UNPLACED:.*]] = aie.logical_tile<CoreTile>(?, ?)
// CHECK: %[[CORE_COL:.*]] = aie.logical_tile<CoreTile>(0, ?)
// CHECK: %[[CORE_ROW:.*]] = aie.logical_tile<CoreTile>(?, 2)
// CHECK: %[[CORE_FIXED:.*]] = aie.logical_tile<CoreTile>(2, 3)
// CHECK: %[[MEM_UNPLACED:.*]] = aie.logical_tile<MemTile>(?, ?)
// CHECK: %[[MEM_COL:.*]] = aie.logical_tile<MemTile>(1, ?)
// CHECK: %[[MEM_ROW:.*]] = aie.logical_tile<MemTile>(?, 1)
// CHECK: %[[MEM_FIXED:.*]] = aie.logical_tile<MemTile>(1, 1)
// CHECK: %[[SHIM_NOC_UNPLACED:.*]] = aie.logical_tile<ShimNOCTile>(?, ?)
// CHECK: %[[SHIM_NOC_COL:.*]] = aie.logical_tile<ShimNOCTile>(0, ?)
// CHECK: %[[SHIM_NOC_ROW:.*]] = aie.logical_tile<ShimNOCTile>(?, 0)
// CHECK: %[[SHIM_NOC_FIXED:.*]] = aie.logical_tile<ShimNOCTile>(0, 0)
// CHECK: %[[SHIM_PL_UNPLACED:.*]] = aie.logical_tile<ShimPLTile>(?, ?)
// CHECK: %[[SHIM_PL_COL:.*]] = aie.logical_tile<ShimPLTile>(0, ?)
// CHECK: %[[SHIM_PL_ROW:.*]] = aie.logical_tile<ShimPLTile>(?, 0)
// CHECK: %[[SHIM_PL_FIXED:.*]] = aie.logical_tile<ShimPLTile>(0, 0)
module @test_logical_tile {
  aie.device(xcve2802) {
    %core_unplaced = aie.logical_tile<CoreTile>(?, ?)
    %core_col = aie.logical_tile<CoreTile>(0, ?)
    %core_row = aie.logical_tile<CoreTile>(?, 2)
    %core_fixed = aie.logical_tile<CoreTile>(2, 3)

    %mem_unplaced = aie.logical_tile<MemTile>(?, ?)
    %mem_col = aie.logical_tile<MemTile>(1, ?)
    %mem_row = aie.logical_tile<MemTile>(?, 1)
    %mem_fixed = aie.logical_tile<MemTile>(1, 1)

    %shim_noc_unplaced = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_noc_col = aie.logical_tile<ShimNOCTile>(3, ?)
    %shim_noc_row = aie.logical_tile<ShimNOCTile>(?, 0)
    %shim_noc_fixed = aie.logical_tile<ShimNOCTile>(2, 0)

    %shim_pl_unplaced = aie.logical_tile<ShimPLTile>(?, ?)
    %shim_pl_col = aie.logical_tile<ShimPLTile>(0, ?)
    %shim_pl_row = aie.logical_tile<ShimPLTile>(?, 0)
    %shim_pl_fixed = aie.logical_tile<ShimPLTile>(0, 0)
    aie.end
  }
}

// -----

// CHECK-LABEL: @test_ssa_names
// CHECK: %logical_core = aie.logical_tile<CoreTile>(?, ?)
// CHECK: %logical_core_0 = aie.logical_tile<CoreTile>(?, ?)
// CHECK: %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
// CHECK: %logical_shim_pl = aie.logical_tile<ShimPLTile>(?, ?)
// CHECK: %logical_mem = aie.logical_tile<MemTile>(?, ?)
module @test_ssa_names {
  aie.device(xcve2802) {
    %t0 = aie.logical_tile<CoreTile>(?, ?)
    %t1 = aie.logical_tile<CoreTile>(?, ?)
    %t2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %t3 = aie.logical_tile<ShimPLTile>(?, ?)
    %t4 = aie.logical_tile<MemTile>(?, ?)
    aie.end
  }
}
