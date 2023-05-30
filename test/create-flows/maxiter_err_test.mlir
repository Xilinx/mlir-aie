//===- maxiter_err_test.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows --aie-find-flows %s |& FileCheck %s
// CHECK: error: Unable to find a legal routing

module {
    AIE.device(xcvc1902) {
        %t01 = AIE.tile(0, 1)
        %t11 = AIE.tile(1, 1)
        %t21 = AIE.tile(2, 1)
        %t31 = AIE.tile(3, 1)
        %t41 = AIE.tile(4, 1)
        %t51 = AIE.tile(5, 1)
        %t61 = AIE.tile(6, 1)
        %t71 = AIE.tile(7, 1)
        %t81 = AIE.tile(8, 1)
        %t02 = AIE.tile(0, 2)
        %t12 = AIE.tile(1, 2)
        %t22 = AIE.tile(2, 2)
        %t32 = AIE.tile(3, 2)
        %t42 = AIE.tile(4, 2)
        %t52 = AIE.tile(5, 2)
        %t62 = AIE.tile(6, 2)
        %t72 = AIE.tile(7, 2)
        %t82 = AIE.tile(8, 2)
        %t03 = AIE.tile(0, 3)
        %t13 = AIE.tile(1, 3)
        %t23 = AIE.tile(2, 3)
        %t33 = AIE.tile(3, 3)
        %t43 = AIE.tile(4, 3)
        %t53 = AIE.tile(5, 3)
        %t63 = AIE.tile(6, 3)
        %t73 = AIE.tile(7, 3)
        %t83 = AIE.tile(8, 3)
        %t04 = AIE.tile(0, 4)
        %t14 = AIE.tile(1, 4)
        %t24 = AIE.tile(2, 4)
        %t34 = AIE.tile(3, 4)
        %t44 = AIE.tile(4, 4)
        %t54 = AIE.tile(5, 4)
        %t64 = AIE.tile(6, 4)
        %t74 = AIE.tile(7, 4)
        %t84 = AIE.tile(8, 4)
        %t20 = AIE.tile(2, 0)
        %t60 = AIE.tile(6, 0)

        AIE.flow(%t01, DMA : 0, %t51, DMA : 0)
        AIE.flow(%t11, DMA : 0, %t61, DMA : 0)
        AIE.flow(%t21, DMA : 0, %t71, DMA : 0)
        AIE.flow(%t31, DMA : 0, %t81, DMA : 0)
        AIE.flow(%t41, DMA : 0, %t81, DMA : 1)

        AIE.flow(%t02, DMA : 0, %t52, DMA : 0)
        AIE.flow(%t12, DMA : 0, %t62, DMA : 0)
        AIE.flow(%t22, DMA : 0, %t72, DMA : 0)
        AIE.flow(%t32, DMA : 0, %t82, DMA : 0)
        AIE.flow(%t42, DMA : 0, %t82, DMA : 1)

        AIE.flow(%t03, DMA : 0, %t53, DMA : 0)
        AIE.flow(%t13, DMA : 0, %t63, DMA : 0)
        AIE.flow(%t23, DMA : 0, %t73, DMA : 0)
        AIE.flow(%t33, DMA : 0, %t83, DMA : 0)
        AIE.flow(%t43, DMA : 0, %t83, DMA : 1)

        AIE.flow(%t04, DMA : 0, %t54, DMA : 0)
        AIE.flow(%t14, DMA : 0, %t64, DMA : 0)
        AIE.flow(%t24, DMA : 0, %t74, DMA : 0)
        AIE.flow(%t34, DMA : 0, %t84, DMA : 0)
        AIE.flow(%t44, DMA : 0, %t84, DMA : 1)

        AIE.flow(%t20, DMA : 0, %t60, DMA : 0)
    }
} 
