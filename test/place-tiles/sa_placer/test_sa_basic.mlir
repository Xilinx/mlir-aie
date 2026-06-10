//===- test_sa_basic.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Basic SA placer tests: structural correctness and constraint satisfaction.

// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=42' %s | FileCheck %s

// Single unconstrained core tile placed somewhere valid
// CHECK-LABEL: @single_core
module @single_core {
  aie.device(npu2) {
    // CHECK: aie.tile({{[0-7]}}, {{[2-5]}})
    %c = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-NOT: aie.logical_tile
    aie.core(%c) { aie.end }
    aie.end
  }
}

// -----

// Two unconstrained cores get distinct physical tiles
// CHECK-LABEL: @two_cores
module @two_cores {
  aie.device(npu2) {
    // CHECK-DAG: %[[T1:.*]] = aie.tile({{[0-7]}}, {{[2-5]}})
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: %[[T2:.*]] = aie.tile({{[0-7]}}, {{[2-5]}})
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Fully constrained core placed at exact coordinates
// CHECK-LABEL: @fully_constrained
module @fully_constrained {
  aie.device(npu2) {
    // CHECK: aie.tile(3, 4)
    %c = aie.logical_tile<CoreTile>(3, 4)
    aie.core(%c) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Column-constrained core: fixed column, free row
// CHECK-LABEL: @column_constrained
module @column_constrained {
  aie.device(npu2) {
    // CHECK: aie.tile(5, {{[2-5]}})
    %c = aie.logical_tile<CoreTile>(5, ?)
    aie.core(%c) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Row-constrained core: free column, fixed row
// CHECK-LABEL: @row_constrained
module @row_constrained {
  aie.device(npu2) {
    // CHECK: aie.tile({{[0-7]}}, 3)
    %c = aie.logical_tile<CoreTile>(?, 3)
    aie.core(%c) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Mixed: one constrained, two unconstrained
// CHECK-LABEL: @mixed_constraints
module @mixed_constraints {
  aie.device(npu2) {
    // CHECK-DAG: aie.tile(2, 3)
    %c1 = aie.logical_tile<CoreTile>(2, 3)
    // CHECK-DAG: aie.tile({{[0-7]}}, {{[2-5]}})
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-DAG: aie.tile({{[0-7]}}, {{[2-5]}})
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    aie.core(%c3) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}

// -----

// Attribute preservation through placement
// CHECK-LABEL: @attribute_preserved
module @attribute_preserved {
  aie.device(npu2) {
    // CHECK: aie.tile({{.*}}) {allocation_scheme = "bank_aware"}
    %c = aie.logical_tile<CoreTile>(?, ?) {allocation_scheme = "bank_aware"}
    aie.core(%c) { aie.end }
    // CHECK-NOT: aie.logical_tile
    aie.end
  }
}
