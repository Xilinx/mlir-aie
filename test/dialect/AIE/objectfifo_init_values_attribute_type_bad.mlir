//===- objectfifo_init_values_attribute_type_bad.mlir -----------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s

// CHECK: custom op 'aie.objectfifo' initial value should be an elements attribute

module @objectfifo_init_values_attribute_type_bad {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile12, {%tile23}, 1 : i32) : !aie.objectfifo<memref<4xi32>> = [[0, 1, 2, 3]]
 }
}
