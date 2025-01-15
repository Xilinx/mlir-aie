//===- objectfifo_invalid_shared_mem.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s

// CHECK: `via_shared_mem` can only be used in 1-to-1 object FIFOs

module @objectfifo_invalid_shared_mem {
 aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile13 = aie.tile(1, 3)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of_0 (%tile20, {%tile13, %tile23}, 2 : i32) {via_shared_mem = 1 : i32} : !aie.objectfifo<memref<16xi32>>
 }
}
