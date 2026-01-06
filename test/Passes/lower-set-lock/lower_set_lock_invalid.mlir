//===- lower_set_lock_invalid.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file --aie-lower-set-lock %s 2>&1 | FileCheck %s

module @test_invalid_lock_value {
  aie.device(npu2) {
    %tile22 = aie.tile(2, 2)
    %lock22_0 = aie.lock(%tile22, 0)  {init = 1 : i32}
    %lock22_15 = aie.lock(%tile22, 15)  {init = 0 : i32}

    aie.runtime_sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
        // Set some runtime parameters before starting execution

        // CHECK: Lock value exceeds the maximum value
        aiex.set_lock(%lock22_0, 1024)
    }
  }
}

// -----

module @test_invalid_lock_value {
  aie.device(xcvc1902) {
    %tile22 = aie.tile(2, 2)
    %lock22_0 = aie.lock(%tile22, 0)  {init = 1 : i32}
    %lock22_15 = aie.lock(%tile22, 15)  {init = 0 : i32}

    aie.runtime_sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
        // Set some runtime parameters before starting execution

        // CHECK: SetLockOp is not supported on AIE1.
        aiex.set_lock(%lock22_0, 1)
    }
  }
}
