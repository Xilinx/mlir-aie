//===- init_values_bad.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// CHECK:   error: inferred shape of elements literal ({{\[}}2, 3]) does not match type ({{\[}}2, 2])

module @init {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile12, {%tile23}, 2 : i32) : !aie.objectfifo<memref<2x2xi32>> = [dense<[[4, 5], [6, 7]]> : memref<2x2xi32>, 
                                                                                            dense<[[0, 1, 2], [3, 4, 5]]> : memref<2x2xi32>]
 }
}
