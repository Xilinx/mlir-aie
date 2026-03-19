//===- test_place_options.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file --aie-place-tiles='cores-per-col=2' %s | FileCheck %s
// RUN: aie-opt -split-input-file --aie-place-tiles %s | FileCheck %s --check-prefix=NO-LIMIT

// cores-per-col=2 spreads 4 cores across 2 columns
// CHECK-LABEL: @cores_per_col_limit
// NO-LIMIT-LABEL: @cores_per_col_limit
module @cores_per_col_limit {
  aie.device(npu1) {
    // With limit: columns 0 and 1
    // CHECK-DAG: %[[C0:.*]] = aie.tile(0, 2)
    // CHECK-DAG: %[[C1:.*]] = aie.tile(0, 3)
    // CHECK-DAG: %[[C2:.*]] = aie.tile(1, 2)
    // CHECK-DAG: %[[C3:.*]] = aie.tile(1, 3)

    // Without limit: all in column 0
    // NO-LIMIT-DAG: %[[C0:.*]] = aie.tile(0, 2)
    // NO-LIMIT-DAG: %[[C1:.*]] = aie.tile(0, 3)
    // NO-LIMIT-DAG: %[[C2:.*]] = aie.tile(0, 4)
    // NO-LIMIT-DAG: %[[C3:.*]] = aie.tile(0, 5)

    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK-NOT: aie.logical_tile
    // NO-LIMIT-NOT: aie.logical_tile
  }
}
