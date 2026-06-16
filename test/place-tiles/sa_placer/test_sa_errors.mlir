//===- test_sa_errors.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify SA placer emits proper errors for invalid designs.

// RUN: not aie-opt --aie-place-tiles='placer=sa_placer sa-seed=42' %s 2>&1 | FileCheck %s

// Device overflow: 33 CoreTiles on npu2 (only 32 available = 8 cols x 4 rows)
// CHECK: error: no available physical tile for placement
module @device_overflow_cores {
  aie.device(npu2) {
    %c00 = aie.logical_tile<CoreTile>(?, ?)
    %c01 = aie.logical_tile<CoreTile>(?, ?)
    %c02 = aie.logical_tile<CoreTile>(?, ?)
    %c03 = aie.logical_tile<CoreTile>(?, ?)
    %c04 = aie.logical_tile<CoreTile>(?, ?)
    %c05 = aie.logical_tile<CoreTile>(?, ?)
    %c06 = aie.logical_tile<CoreTile>(?, ?)
    %c07 = aie.logical_tile<CoreTile>(?, ?)
    %c08 = aie.logical_tile<CoreTile>(?, ?)
    %c09 = aie.logical_tile<CoreTile>(?, ?)
    %c10 = aie.logical_tile<CoreTile>(?, ?)
    %c11 = aie.logical_tile<CoreTile>(?, ?)
    %c12 = aie.logical_tile<CoreTile>(?, ?)
    %c13 = aie.logical_tile<CoreTile>(?, ?)
    %c14 = aie.logical_tile<CoreTile>(?, ?)
    %c15 = aie.logical_tile<CoreTile>(?, ?)
    %c16 = aie.logical_tile<CoreTile>(?, ?)
    %c17 = aie.logical_tile<CoreTile>(?, ?)
    %c18 = aie.logical_tile<CoreTile>(?, ?)
    %c19 = aie.logical_tile<CoreTile>(?, ?)
    %c20 = aie.logical_tile<CoreTile>(?, ?)
    %c21 = aie.logical_tile<CoreTile>(?, ?)
    %c22 = aie.logical_tile<CoreTile>(?, ?)
    %c23 = aie.logical_tile<CoreTile>(?, ?)
    %c24 = aie.logical_tile<CoreTile>(?, ?)
    %c25 = aie.logical_tile<CoreTile>(?, ?)
    %c26 = aie.logical_tile<CoreTile>(?, ?)
    %c27 = aie.logical_tile<CoreTile>(?, ?)
    %c28 = aie.logical_tile<CoreTile>(?, ?)
    %c29 = aie.logical_tile<CoreTile>(?, ?)
    %c30 = aie.logical_tile<CoreTile>(?, ?)
    %c31 = aie.logical_tile<CoreTile>(?, ?)
    // This 33rd tile exceeds the 32 available positions
    %c32 = aie.logical_tile<CoreTile>(?, ?)

    aie.core(%c00) { aie.end }
    aie.core(%c01) { aie.end }
    aie.core(%c02) { aie.end }
    aie.core(%c03) { aie.end }
    aie.core(%c04) { aie.end }
    aie.core(%c05) { aie.end }
    aie.core(%c06) { aie.end }
    aie.core(%c07) { aie.end }
    aie.core(%c08) { aie.end }
    aie.core(%c09) { aie.end }
    aie.core(%c10) { aie.end }
    aie.core(%c11) { aie.end }
    aie.core(%c12) { aie.end }
    aie.core(%c13) { aie.end }
    aie.core(%c14) { aie.end }
    aie.core(%c15) { aie.end }
    aie.core(%c16) { aie.end }
    aie.core(%c17) { aie.end }
    aie.core(%c18) { aie.end }
    aie.core(%c19) { aie.end }
    aie.core(%c20) { aie.end }
    aie.core(%c21) { aie.end }
    aie.core(%c22) { aie.end }
    aie.core(%c23) { aie.end }
    aie.core(%c24) { aie.end }
    aie.core(%c25) { aie.end }
    aie.core(%c26) { aie.end }
    aie.core(%c27) { aie.end }
    aie.core(%c28) { aie.end }
    aie.core(%c29) { aie.end }
    aie.core(%c30) { aie.end }
    aie.core(%c31) { aie.end }
    aie.core(%c32) { aie.end }
  }
}
