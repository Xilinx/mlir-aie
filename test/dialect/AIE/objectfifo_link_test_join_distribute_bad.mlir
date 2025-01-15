//===- objectfifo_link_test_join_distribute_bad.mlir ------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: June 28th 2023
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s

// CHECK:   error: ObjectFifoLinkOp does not support 'join' and 'distribute' at the same time

module @objectfifo_link_test_join_distribute_bad {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)
        %tile33 = aie.tile(3, 3)

        aie.objectfifo @link1 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
        aie.objectfifo @link2 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
        aie.objectfifo @link3 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

        aie.objectfifo @link4 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
        aie.objectfifo @link5 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
        aie.objectfifo @link6 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

        aie.objectfifo.link [@link1, @link2, @link3] -> [@link4, @link5, @link6] ([0, 16, 36][0, 16, 36])
    }
}
