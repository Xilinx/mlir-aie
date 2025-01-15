//===- objectfifo_insufficient_depths.mlir ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s

// CHECK: 'aie.objectfifo' op does not have enough depths specified for producer and for each consumer.

module @objectfifo_insufficient_depths {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of_0 (%tile12, {%tile13, %tile23}, [2, 2]) : !aie.objectfifo<memref<16xi32>>
 }
}
