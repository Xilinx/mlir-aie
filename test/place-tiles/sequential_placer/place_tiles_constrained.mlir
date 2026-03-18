//===- place_tiles_constrained.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// Test: Fully constrained placement
// CHECK-LABEL: @fully_constrained
module @fully_constrained {
  aie.device(npu1) {
    // Tile explicitly placed at (2, 3)
    // CHECK: %[[TILE:.*]] = aie.tile(2, 3)
    %tile = aie.logical_tile<CoreTile>(2, 3)

    // CHECK: aie.core(%[[TILE]])
    aie.core(%tile) {
      aie.end
    }

    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Test: Mixed constrained and unconstrained
// CHECK-LABEL: @mixed_constraints
module @mixed_constraints {
  aie.device(npu1) {
    // Fully constrained to (1, 2)
    // CHECK-DAG: %[[C1:.*]] = aie.tile(1, 2)
    %c1 = aie.logical_tile<CoreTile>(1, 2)

    // First available should be (0, 2)
    // CHECK-DAG: %[[C2:.*]] = aie.tile(0, 2)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    // Second available should be (2, 2) (skipping already used (1, 2))
    // CHECK-DAG: %[[C3:.*]] = aie.tile(2, 2)
    %c3 = aie.logical_tile<CoreTile>(?, ?)

    // CHECK: aie.core(%[[C1]])
    aie.core(%c1) { aie.end }
    // CHECK: aie.core(%[[C2]])
    aie.core(%c2) { aie.end }
    // CHECK: aie.core(%[[C3]])
    aie.core(%c3) { aie.end }

    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Test: Constrained MemTile and ShimTile
// CHECK-LABEL: @constrained_memtile_shimtile
module @constrained_memtile_shimtile {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(1, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)

    // MemTile fully constrained to (1, 1)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(1, 1)
    %mem = aie.logical_tile<MemTile>(1, 1)

    // ShimTile fully constrained to (0, 0)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)

    // CHECK: aie.objectfifo @of1(%[[SHIM]], {%[[MEM]]}, 2 : i32)
    aie.objectfifo @of1 (%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    // CHECK: aie.objectfifo @of2(%[[MEM]], {%[[CORE]]}, 2 : i32)
    aie.objectfifo @of2 (%mem, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    // CHECK-NOT: aie.logical_tile
  }
}
