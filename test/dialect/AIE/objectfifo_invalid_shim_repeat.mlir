//===- objectfifo_invalid_shim_repeat.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s

// CHECK: `repeat_count` unavailable for shim tiles

module @objectfifo_invalid_shim_repeat {
 aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of_0 (%tile20, {%tile13}, 2 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
 }
}
