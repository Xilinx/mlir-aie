//===- test_place_basic.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// CHECK-LABEL: @simple_worker
module @simple_worker {
  aie.device(npu1) {
    // CHECK: %[[TILE:.*]] = aie.tile(0, 2)
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    // CHECK: aie.core(%[[TILE]])
    aie.core(%logical_core) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// CHECK-LABEL: @allocation_scheme_copied
module @allocation_scheme_copied {
  aie.device(npu1) {
    // CHECK: %[[TILE:.*]] = aie.tile(0, 2) {allocation_scheme = "bank_aware"}
    %logical_core = aie.logical_tile<CoreTile>(?, ?) {allocation_scheme = "bank_aware"}
    // CHECK: aie.core(%[[TILE]])
    aie.core(%logical_core) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// MemTile not in ObjectFifo uses fallback placement
// CHECK-LABEL: @standalone_memtile
module @standalone_memtile {
  aie.device(npu1) {
    // CHECK: %[[MEM:.*]] = aie.tile(0, 1)
    %mem = aie.logical_tile<MemTile>(?, ?)
    // CHECK: aie.buffer(%[[MEM]])
    %buf = aie.buffer(%mem) : memref<16xi32>
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// CHECK-LABEL: @fully_constrained
module @fully_constrained {
  aie.device(npu1) {
    // CHECK: %[[TILE:.*]] = aie.tile(2, 3)
    %tile = aie.logical_tile<CoreTile>(2, 3)
    // CHECK: aie.core(%[[TILE]])
    aie.core(%tile) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// CHECK-LABEL: @mixed_constraints
module @mixed_constraints {
  aie.device(npu1) {
    // Fully constrained to (1, 2)
    // CHECK-DAG: %[[C1:.*]] = aie.tile(1, 2)
    %c1 = aie.logical_tile<CoreTile>(1, 2)
    // Sequential placement: (0, 2), (0, 3)
    // CHECK-DAG: %[[C2:.*]] = aie.tile(0, 2)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[C3:.*]] = aie.tile(0, 3)
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

// CHECK-LABEL: @constrained_mem_shim
module @constrained_mem_shim {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(1, 1)
    %mem = aie.logical_tile<MemTile>(1, 1)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)

    // CHECK: aie.objectfifo @of1(%[[SHIM]], {%[[MEM]]}, 2 : i32)
    aie.objectfifo @of1 (%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // CHECK: aie.objectfifo @of2(%[[MEM]], {%[[CORE]]}, 2 : i32)
    aie.objectfifo @of2 (%mem, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Partial constraints are currently ignored by sequential placer

// Column-only constraint ignored
// CHECK-LABEL: @partial_col_ignored
module @partial_col_ignored {
  aie.device(npu1) {
    // Constrained to column 1, but placed at (0, 2)
    // CHECK: %[[TILE:.*]] = aie.tile(0, 2)
    %tile = aie.logical_tile<CoreTile>(1, ?)
    // CHECK: aie.core(%[[TILE]])
    aie.core(%tile) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Row-only constraint ignored
// CHECK-LABEL: @partial_row_ignored
module @partial_row_ignored {
  aie.device(npu1) {
    // Constrained to row 3, but placed at (0, 2)
    // CHECK: %[[TILE:.*]] = aie.tile(0, 2)
    %tile = aie.logical_tile<CoreTile>(?, 3)
    // CHECK: aie.core(%[[TILE]])
    aie.core(%tile) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// MemTile column constraint ignored (uses commonCol)
// CHECK-LABEL: @partial_memtile_col
module @partial_memtile_col {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(0, 2)
    %core = aie.logical_tile<CoreTile>(?, ?)
    // Constrained to column 2, but placed at (0, 1)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem = aie.logical_tile<MemTile>(2, ?)
    // CHECK: aie.objectfifo @of1(%[[MEM]], {%[[CORE]]}, 2 : i32)
    aie.objectfifo @of1 (%mem, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // CHECK-NOT: aie.logical_tile
  }
}
