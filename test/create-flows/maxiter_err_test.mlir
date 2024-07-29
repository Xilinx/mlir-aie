//===- maxiter_err_test.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// This test is known to timeout after the refactor of the pathfinder
// REQUIRES: zhewen

// RUN: not aie-opt --aie-create-pathfinder-flows --aie-find-flows %s 2>&1 | FileCheck %s
// CHECK: error: Unable to find a legal routing

module {
    aie.device(xcvc1902) {
        %t01 = aie.tile(0, 1)
        %t11 = aie.tile(1, 1)
        %t21 = aie.tile(2, 1)
        %t31 = aie.tile(3, 1)
        %t41 = aie.tile(4, 1)
        %t51 = aie.tile(5, 1)
        %t61 = aie.tile(6, 1)
        %t71 = aie.tile(7, 1)
        %t81 = aie.tile(8, 1)
        %t02 = aie.tile(0, 2)
        %t12 = aie.tile(1, 2)
        %t22 = aie.tile(2, 2)
        %t32 = aie.tile(3, 2)
        %t42 = aie.tile(4, 2)
        %t52 = aie.tile(5, 2)
        %t62 = aie.tile(6, 2)
        %t72 = aie.tile(7, 2)
        %t82 = aie.tile(8, 2)
        %t03 = aie.tile(0, 3)
        %t13 = aie.tile(1, 3)
        %t23 = aie.tile(2, 3)
        %t33 = aie.tile(3, 3)
        %t43 = aie.tile(4, 3)
        %t53 = aie.tile(5, 3)
        %t63 = aie.tile(6, 3)
        %t73 = aie.tile(7, 3)
        %t83 = aie.tile(8, 3)
        %t04 = aie.tile(0, 4)
        %t14 = aie.tile(1, 4)
        %t24 = aie.tile(2, 4)
        %t34 = aie.tile(3, 4)
        %t44 = aie.tile(4, 4)
        %t54 = aie.tile(5, 4)
        %t64 = aie.tile(6, 4)
        %t74 = aie.tile(7, 4)
        %t84 = aie.tile(8, 4)
        %t20 = aie.tile(2, 0)
        %t60 = aie.tile(6, 0)

        aie.flow(%t01, DMA : 0, %t51, DMA : 0)
        aie.flow(%t11, DMA : 0, %t61, DMA : 0)
        aie.flow(%t21, DMA : 0, %t71, DMA : 0)
        aie.flow(%t31, DMA : 0, %t81, DMA : 0)
        aie.flow(%t41, DMA : 0, %t81, DMA : 1)

        aie.flow(%t02, DMA : 0, %t52, DMA : 0)
        aie.flow(%t12, DMA : 0, %t62, DMA : 0)
        aie.flow(%t22, DMA : 0, %t72, DMA : 0)
        aie.flow(%t32, DMA : 0, %t82, DMA : 0)
        aie.flow(%t42, DMA : 0, %t82, DMA : 1)

        aie.flow(%t03, DMA : 0, %t53, DMA : 0)
        aie.flow(%t13, DMA : 0, %t63, DMA : 0)
        aie.flow(%t23, DMA : 0, %t73, DMA : 0)
        aie.flow(%t33, DMA : 0, %t83, DMA : 0)
        aie.flow(%t43, DMA : 0, %t83, DMA : 1)

        aie.flow(%t04, DMA : 0, %t54, DMA : 0)
        aie.flow(%t14, DMA : 0, %t64, DMA : 0)
        aie.flow(%t24, DMA : 0, %t74, DMA : 0)
        aie.flow(%t34, DMA : 0, %t84, DMA : 0)
        aie.flow(%t44, DMA : 0, %t84, DMA : 1)

        aie.flow(%t20, DMA : 0, %t60, DMA : 0)
    }
} 
