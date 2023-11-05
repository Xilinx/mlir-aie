//===- many_flows2.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s
// CHECK: %[[T02:.*]] = AIE.tile(0, 2)
// CHECK: %[[T03:.*]] = AIE.tile(0, 3)
// CHECK: %[[T11:.*]] = AIE.tile(1, 1)
// CHECK: %[[T13:.*]] = AIE.tile(1, 3)
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK: %[[T22:.*]] = AIE.tile(2, 2)
// CHECK: %[[T30:.*]] = AIE.tile(3, 0)
// CHECK: %[[T31:.*]] = AIE.tile(3, 1)
// CHECK: %[[T60:.*]] = AIE.tile(6, 0)
// CHECK: %[[T70:.*]] = AIE.tile(7, 0)
// CHECK: %[[T73:.*]] = AIE.tile(7, 3)
//
// CHECK: AIE.flow(%[[T02]], DMA : 0, %[[T60]], DMA : 0)
// CHECK: AIE.flow(%[[T03]], Core : 0, %[[T02]], Core : 1)
// CHECK: AIE.flow(%[[T03]], Core : 1, %[[T02]], Core : 0)
// CHECK: AIE.flow(%[[T03]], DMA : 0, %[[T30]], DMA : 0)
// CHECK: AIE.flow(%[[T03]], DMA : 1, %[[T70]], DMA : 1)
// CHECK: AIE.flow(%[[T13]], Core : 1, %[[T31]], Core : 1)
// CHECK: AIE.flow(%[[T22]], Core : 0, %[[T13]], Core : 0)
// CHECK: AIE.flow(%[[T22]], DMA : 0, %[[T20]], DMA : 0)
// CHECK: AIE.flow(%[[T31]], DMA : 0, %[[T20]], DMA : 1)
// CHECK: AIE.flow(%[[T31]], DMA : 1, %[[T30]], DMA : 1)
// CHECK: AIE.flow(%[[T73]], Core : 0, %[[T31]], Core : 0)
// CHECK: AIE.flow(%[[T73]], Core : 1, %[[T22]], Core : 1)
// CHECK: AIE.flow(%[[T73]], DMA : 0, %[[T60]], DMA : 1)
// CHECK: AIE.flow(%[[T73]], DMA : 1, %[[T70]], DMA : 0)

module {
    AIE.device(xcvc1902) {
        %t02 = AIE.tile(0, 2)
        %t03 = AIE.tile(0, 3)
        %t11 = AIE.tile(1, 1)
        %t13 = AIE.tile(1, 3)
        %t20 = AIE.tile(2, 0)
        %t22 = AIE.tile(2, 2)
        %t30 = AIE.tile(3, 0)
        %t31 = AIE.tile(3, 1)
        %t60 = AIE.tile(6, 0)
        %t70 = AIE.tile(7, 0)
        %t73 = AIE.tile(7, 3)

        AIE.flow(%t03, DMA : 0, %t30, DMA : 0)
        AIE.flow(%t03, DMA : 1, %t70, DMA : 1)
        AIE.flow(%t02, DMA : 0, %t60, DMA : 0)
        AIE.flow(%t22, DMA : 0, %t20, DMA : 0)

        AIE.flow(%t22, Core : 0, %t13, Core : 0)
        AIE.flow(%t03, Core : 1, %t02, Core : 0)
        AIE.flow(%t73, Core : 0, %t31, Core : 0)
        AIE.flow(%t73, Core : 1, %t22, Core : 1)

        AIE.flow(%t73, DMA : 0, %t60, DMA : 1)
        AIE.flow(%t73, DMA : 1, %t70, DMA : 0)
        AIE.flow(%t31, DMA : 0, %t20, DMA : 1)
        AIE.flow(%t31, DMA : 1, %t30, DMA : 1)

        AIE.flow(%t03, Core : 0, %t02, Core : 1)
        AIE.flow(%t13, Core : 1, %t31, Core : 1)
    }
}
