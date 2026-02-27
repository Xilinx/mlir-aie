//===- place_tiles_simple.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles %s | FileCheck %s

// Test simple worker placement
// CHECK-LABEL: @simple_worker
module @simple_worker {
  aie.device(npu1) {
    // CHECK: %[[TILE:.*]] = aie.tile(0, 2)
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    // CHECK: aie.core(%[[TILE]])
    aie.core(%logical_core) {
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}
