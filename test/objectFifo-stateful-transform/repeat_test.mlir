//===- repeat_test.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @repeat {

module @repeat {
    aie.device(xcve2802) {
        %tile10 = aie.tile(1, 1)
        %tile12 = aie.tile(1, 2)

        aie.objectfifo @of0 (%tile10, {%tile12}, 1 : i32) : !aie.objectfifo<memref<64x256xi32>>

        aie.objectfifo @of1 (%tile12 toStream [<size = 8, stride = 32>, <size = 32, stride = 256>, <size = 32, stride = 1>], {%tile13}, 1 : i32) : !aie.objectfifo<memref<32x32xi32>>

        aie.objectfifo @of2 (%tile12 toStream [<size = 8, stride = 32>, <size = 32, stride = 256>, <size = 32, stride = 1>], {%tile33}, 1 : i32) : !aie.objectfifo<memref<32x32xi32>>

        aie.objectfifo.link [@of0] -> [@of1, @of2] ()
    }
}
