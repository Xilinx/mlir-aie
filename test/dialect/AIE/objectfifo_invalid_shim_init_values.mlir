//===- objectfifo_invalid_shim_init_values.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s

// CHECK: `init_values` unavailable for shim tiles

module @objectfifo_invalid_shim_init_values {
 aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of_0 (%tile20, {%tile13}, 2 : i32) : !aie.objectfifo<memref<2x2xi32>> = [dense<[[0, 1], [2, 3]]> : memref<2x2xi32>, 
                                                                                             dense<[[4, 5], [6, 7]]> : memref<2x2xi32>]
 }
}
