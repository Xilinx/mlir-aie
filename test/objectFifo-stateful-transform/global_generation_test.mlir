//===- global_generation_test.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

module @globals {
 aie.device(npu1) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of0 (%tile12, {%tile13}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of1 (%tile12, {%tile13, %tile33}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
 }
}
