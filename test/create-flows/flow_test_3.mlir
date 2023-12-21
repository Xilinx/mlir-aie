//===- flow_test_3.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s
//CHECK: %[[t01:.*]] = aie.tile(0, 1)
//CHECK: %[[t02:.*]] = aie.tile(0, 2)
//CHECK: %[[t03:.*]] = aie.tile(0, 3)
//CHECK: %[[t04:.*]] = aie.tile(0, 4)
//CHECK: %[[t11:.*]] = aie.tile(1, 1)
//CHECK: %[[t12:.*]] = aie.tile(1, 2)
//CHECK: %[[t13:.*]] = aie.tile(1, 3)
//CHECK: %[[t14:.*]] = aie.tile(1, 4)
//CHECK: %[[t20:.*]] = aie.tile(2, 0)
//CHECK: %[[t21:.*]] = aie.tile(2, 1)
//CHECK: %[[t22:.*]] = aie.tile(2, 2)
//CHECK: %[[t23:.*]] = aie.tile(2, 3)
//CHECK: %[[t24:.*]] = aie.tile(2, 4)
//CHECK: %[[t30:.*]] = aie.tile(3, 0)
//CHECK: %[[t71:.*]] = aie.tile(7, 1)
//CHECK: %[[t72:.*]] = aie.tile(7, 2)
//CHECK: %[[t73:.*]] = aie.tile(7, 3)
//CHECK: %[[t74:.*]] = aie.tile(7, 4)
//CHECK: %[[t81:.*]] = aie.tile(8, 1)
//CHECK: %[[t82:.*]] = aie.tile(8, 2)
//CHECK: %[[t83:.*]] = aie.tile(8, 3)
//CHECK: %[[t84:.*]] = aie.tile(8, 4)

//CHECK: aie.flow(%[[t01]], Core : 0, %[[t83]], Core : 0)
//CHECK: aie.flow(%[[t01]], Core : 1, %[[t72]], Core : 1)
//CHECK: aie.flow(%[[t02]], Core : 1, %[[t24]], Core : 1)
//CHECK: aie.flow(%[[t03]], Core : 0, %[[t71]], Core : 0)
//CHECK: aie.flow(%[[t11]], Core : 0, %[[t24]], Core : 0)
//CHECK: aie.flow(%[[t14]], Core : 0, %[[t01]], Core : 0)
//CHECK: aie.flow(%[[t20]], DMA : 0, %[[t03]], DMA : 0)
//CHECK: aie.flow(%[[t20]], DMA : 1, %[[t83]], DMA : 1)
//CHECK: aie.flow(%[[t21]], Core : 0, %[[t73]], Core : 0)
//CHECK: aie.flow(%[[t24]], Core : 1, %[[t71]], Core : 1)
//CHECK: aie.flow(%[[t24]], DMA : 0, %[[t20]], DMA : 0)
//CHECK: aie.flow(%[[t30]], DMA : 0, %[[t14]], DMA : 0)
//CHECK: aie.flow(%[[t71]], Core : 0, %[[t84]], Core : 0)
//CHECK: aie.flow(%[[t71]], Core : 1, %[[t84]], Core : 1)
//CHECK: aie.flow(%[[t72]], Core : 1, %[[t02]], Core : 1)
//CHECK: aie.flow(%[[t73]], Core : 0, %[[t82]], Core : 0)
//CHECK: aie.flow(%[[t82]], DMA : 0, %[[t30]], DMA : 0)
//CHECK: aie.flow(%[[t83]], Core : 0, %[[t21]], Core : 0)
//CHECK: aie.flow(%[[t83]], Core : 1, %[[t01]], Core : 1)
//CHECK: aie.flow(%[[t84]], Core : 0, %[[t11]], Core : 0)
//CHECK: aie.flow(%[[t84]], DMA : 1, %[[t20]], DMA : 1)

module {
    aie.device(xcvc1902) {
        %t01 = aie.tile(0, 1)
        %t02 = aie.tile(0, 2)
        %t03 = aie.tile(0, 3)
        %t04 = aie.tile(0, 4)
        %t11 = aie.tile(1, 1)
        %t12 = aie.tile(1, 2)
        %t13 = aie.tile(1, 3)
        %t14 = aie.tile(1, 4)
        %t20 = aie.tile(2, 0)
        %t21 = aie.tile(2, 1)
        %t22 = aie.tile(2, 2)
        %t23 = aie.tile(2, 3)
        %t24 = aie.tile(2, 4)
        %t30 = aie.tile(3, 0)
        %t71 = aie.tile(7, 1)
        %t72 = aie.tile(7, 2)
        %t73 = aie.tile(7, 3)
        %t74 = aie.tile(7, 4)
        %t81 = aie.tile(8, 1)
        %t82 = aie.tile(8, 2)
        %t83 = aie.tile(8, 3)
        %t84 = aie.tile(8, 4)

        //TASK 1
        aie.flow(%t20, DMA : 0, %t03, DMA : 0)
        aie.flow(%t03, Core : 0, %t71, Core : 0)
        aie.flow(%t71, Core : 0, %t84, Core : 0)
        aie.flow(%t84, Core : 0, %t11, Core : 0)
        aie.flow(%t11, Core : 0, %t24, Core : 0)
        aie.flow(%t24, DMA : 0, %t20, DMA : 0)

        //TASK 2
        aie.flow(%t30, DMA : 0, %t14, DMA : 0)
        aie.flow(%t14, Core : 0, %t01, Core : 0)
        aie.flow(%t01, Core : 0, %t83, Core : 0)
        aie.flow(%t83, Core : 0, %t21, Core : 0)
        aie.flow(%t21, Core : 0, %t73, Core : 0)
        aie.flow(%t73, Core : 0, %t82, Core : 0)
        aie.flow(%t82, DMA : 0, %t30, DMA : 0)

        //TASK 3
        aie.flow(%t20, DMA : 1, %t83, DMA : 1)
        aie.flow(%t83, Core : 1, %t01, Core : 1)
        aie.flow(%t01, Core : 1, %t72, Core : 1)
        aie.flow(%t72, Core : 1, %t02, Core : 1)
        aie.flow(%t02, Core : 1, %t24, Core : 1)
        aie.flow(%t24, Core : 1, %t71, Core : 1)
        aie.flow(%t71, Core : 1, %t84, Core : 1)
        aie.flow(%t84, DMA : 1, %t20, DMA : 1)
    }
}   