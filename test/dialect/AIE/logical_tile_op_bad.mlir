//===- logical_tile_op_bad.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file %s 2>&1 | FileCheck %s

// CHECK: error{{.*}}'aie.logical_tile' op column index (100) must be less than the number of columns in the device
module @test_logical_tile_bad_col {
  aie.device(npu2) {
    %tile = aie.logical_tile<CoreTile>(100, ?)
    aie.end
  }
}

// -----

// CHECK: error{{.*}}'aie.logical_tile' op row index (100) must be less than the number of rows in the device
module @test_logical_tile_bad_row {
  aie.device(npu2) {
    %tile = aie.logical_tile<CoreTile>(?, 100)
    aie.end
  }
}

// -----

// CHECK: error{{.*}}'aie.logical_tile' op declared logical tile type does not match the tile type at coordinates (0, 0)
module @test_logical_tile_type_mismatch_fixed {
  aie.device(npu2) {
    %tile = aie.logical_tile<CoreTile>(0, 0)
    aie.end
  }
}

// -----

// CHECK: error{{.*}}'aie.logical_tile' op Shim tiles cannot have an allocation scheme
module @test_logical_tile_shim_allocation_scheme {
  aie.device(npu2) {
    %tile = aie.logical_tile<ShimNOCTile>(0, 0) {allocation_scheme = "basic-sequential"}
    aie.end
  }
}
