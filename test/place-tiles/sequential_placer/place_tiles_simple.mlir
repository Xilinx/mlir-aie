//===- place_tiles_simple.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

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

// -----

// Test: allocation_scheme attribute is copied to physical tile
// CHECK-LABEL: @allocation_scheme_copied
module @allocation_scheme_copied {
  aie.device(npu1) {
    // CHECK: %[[TILE:.*]] = aie.tile(0, 2) {allocation_scheme = "bank_aware"}
    %logical_core = aie.logical_tile<CoreTile>(?, ?) {allocation_scheme = "bank_aware"}
    // CHECK: aie.core(%[[TILE]])
    aie.core(%logical_core) {
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}
