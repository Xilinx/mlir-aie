//===- flow_test_3.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-flows --aie-find-flows %s | FileCheck %s
//CHECK: %[[t01:.*]] = AIE.tile(0, 1)
//CHECK: %[[t02:.*]] = AIE.tile(0, 2)
//CHECK: %[[t03:.*]] = AIE.tile(0, 3)
//CHECK: %[[t04:.*]] = AIE.tile(0, 4)
//CHECK: %[[t11:.*]] = AIE.tile(1, 1)
//CHECK: %[[t12:.*]] = AIE.tile(1, 2)
//CHECK: %[[t13:.*]] = AIE.tile(1, 3)
//CHECK: %[[t14:.*]] = AIE.tile(1, 4)
//CHECK: %[[t20:.*]] = AIE.tile(2, 0)
//CHECK: %[[t21:.*]] = AIE.tile(2, 1)
//CHECK: %[[t22:.*]] = AIE.tile(2, 2)
//CHECK: %[[t23:.*]] = AIE.tile(2, 3)
//CHECK: %[[t24:.*]] = AIE.tile(2, 4)
//CHECK: %[[t30:.*]] = AIE.tile(3, 0)
//CHECK: %[[t71:.*]] = AIE.tile(7, 1)
//CHECK: %[[t72:.*]] = AIE.tile(7, 2)
//CHECK: %[[t73:.*]] = AIE.tile(7, 3)
//CHECK: %[[t74:.*]] = AIE.tile(7, 4)
//CHECK: %[[t81:.*]] = AIE.tile(8, 1)
//CHECK: %[[t82:.*]] = AIE.tile(8, 2)
//CHECK: %[[t83:.*]] = AIE.tile(8, 3)
//CHECK: %[[t84:.*]] = AIE.tile(8, 4)

//CHECK: AIE.flow(%[[t01]], Core : 0, %[[t83]], Core : 0)
//CHECK: AIE.flow(%[[t01]], Core : 1, %[[t72]], Core : 1)
//CHECK: AIE.flow(%[[t02]], Core : 1, %[[t24]], Core : 1)
//CHECK: AIE.flow(%[[t03]], Core : 0, %[[t71]], Core : 0)
//CHECK: AIE.flow(%[[t11]], Core : 0, %[[t24]], Core : 0)
//CHECK: AIE.flow(%[[t14]], Core : 0, %[[t01]], Core : 0)
//CHECK: AIE.flow(%[[t20]], DMA : 0, %[[t03]], DMA : 0)
//CHECK: AIE.flow(%[[t20]], DMA : 1, %[[t83]], DMA : 1)
//CHECK: AIE.flow(%[[t21]], Core : 0, %[[t73]], Core : 0)
//CHECK: AIE.flow(%[[t24]], Core : 1, %[[t71]], Core : 1)
//CHECK: AIE.flow(%[[t24]], DMA : 0, %[[t20]], DMA : 0)
//CHECK: AIE.flow(%[[t30]], DMA : 0, %[[t14]], DMA : 0)
//CHECK: AIE.flow(%[[t71]], Core : 0, %[[t84]], Core : 0)
//CHECK: AIE.flow(%[[t71]], Core : 1, %[[t84]], Core : 1)
//CHECK: AIE.flow(%[[t72]], Core : 1, %[[t02]], Core : 1)
//CHECK: AIE.flow(%[[t73]], Core : 0, %[[t82]], Core : 0)
//CHECK: AIE.flow(%[[t82]], DMA : 0, %[[t30]], DMA : 0)
//CHECK: AIE.flow(%[[t83]], Core : 0, %[[t21]], Core : 0)
//CHECK: AIE.flow(%[[t83]], Core : 1, %[[t01]], Core : 1)
//CHECK: AIE.flow(%[[t84]], Core : 0, %[[t11]], Core : 0)
//CHECK: AIE.flow(%[[t84]], DMA : 1, %[[t20]], DMA : 1)

module {
%t01 = AIE.tile(0, 1)
%t02 = AIE.tile(0, 2)
%t03 = AIE.tile(0, 3)
%t04 = AIE.tile(0, 4)
%t11 = AIE.tile(1, 1)
%t12 = AIE.tile(1, 2)
%t13 = AIE.tile(1, 3)
%t14 = AIE.tile(1, 4)
%t20 = AIE.tile(2, 0)
%t21 = AIE.tile(2, 1)
%t22 = AIE.tile(2, 2)
%t23 = AIE.tile(2, 3)
%t24 = AIE.tile(2, 4)
%t30 = AIE.tile(3, 0)
%t71 = AIE.tile(7, 1)
%t72 = AIE.tile(7, 2)
%t73 = AIE.tile(7, 3)
%t74 = AIE.tile(7, 4)
%t81 = AIE.tile(8, 1)
%t82 = AIE.tile(8, 2)
%t83 = AIE.tile(8, 3)
%t84 = AIE.tile(8, 4)

//TASK 1
AIE.flow(%t20, DMA : 0, %t03, DMA : 0)
AIE.flow(%t03, Core : 0, %t71, Core : 0)
AIE.flow(%t71, Core : 0, %t84, Core : 0)
AIE.flow(%t84, Core : 0, %t11, Core : 0)
AIE.flow(%t11, Core : 0, %t24, Core : 0)
AIE.flow(%t24, DMA : 0, %t20, DMA : 0)

//TASK 2
AIE.flow(%t30, DMA : 0, %t14, DMA : 0)
AIE.flow(%t14, Core : 0, %t01, Core : 0)
AIE.flow(%t01, Core : 0, %t83, Core : 0)
AIE.flow(%t83, Core : 0, %t21, Core : 0)
AIE.flow(%t21, Core : 0, %t73, Core : 0)
AIE.flow(%t73, Core : 0, %t82, Core : 0)
AIE.flow(%t82, DMA : 0, %t30, DMA : 0)

//TASK 3
AIE.flow(%t20, DMA : 1, %t83, DMA : 1)
AIE.flow(%t83, Core : 1, %t01, Core : 1)
AIE.flow(%t01, Core : 1, %t72, Core : 1)
AIE.flow(%t72, Core : 1, %t02, Core : 1)
AIE.flow(%t02, Core : 1, %t24, Core : 1)
AIE.flow(%t24, Core : 1, %t71, Core : 1)
AIE.flow(%t71, Core : 1, %t84, Core : 1)
AIE.flow(%t84, DMA : 1, %t20, DMA : 1)

}