//===- test_sa_merge.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify SA placer merges MemTile and ShimNOCTile logical tiles that land
// in the same column into a single physical tile.

// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=42' %s | FileCheck %s

// Two MemTiles feeding the same pinned core should merge into one physical tile
// CHECK-LABEL: @memtile_merge_same_column
module @memtile_merge_same_column {
  aie.device(npu2) {
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)
    // Pin core to col 3 so MemTiles are drawn to same column
    %core = aie.logical_tile<CoreTile>(3, 3)

    aie.objectfifo @of1(%mem1, {%core}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @of2(%mem2, {%core}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.core(%core) { aie.end }

    // Both MemTiles should merge to the same physical tile in col 3
    // CHECK-COUNT-1: aie.tile(3, 1)
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Two ShimNOCTiles feeding the same column should merge
// CHECK-LABEL: @shimtile_merge_same_column
module @shimtile_merge_same_column {
  aie.device(npu2) {
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    // Pin core and mem to col 2 to draw shims to same column
    %mem = aie.logical_tile<MemTile>(2, ?)
    %core = aie.logical_tile<CoreTile>(2, 3)

    aie.objectfifo @in1(%shim1, {%mem}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @in2(%shim2, {%mem}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @to_core(%mem, {%core}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.core(%core) { aie.end }

    // Both ShimNOCTiles should merge to one physical tile
    // CHECK-COUNT-1: aie.tile(2, 0)
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Two MemTiles serving cores in different columns should NOT merge
// CHECK-LABEL: @memtile_separate_columns
module @memtile_separate_columns {
  aie.device(npu2) {
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)
    // Pin cores to different columns
    %core1 = aie.logical_tile<CoreTile>(2, 3)
    %core2 = aie.logical_tile<CoreTile>(5, 3)

    aie.objectfifo @of1(%mem1, {%core1}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @of2(%mem2, {%core2}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.core(%core1) { aie.end }
    aie.core(%core2) { aie.end }

    // Two separate MemTiles at row 1 in different columns
    // CHECK-DAG: aie.tile({{[0-7]}}, 1)
    // CHECK-DAG: aie.tile({{[0-7]}}, 1)
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}
