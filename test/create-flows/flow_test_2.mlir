//===- flow_test_2.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s
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
//CHECK: %[[t31:.*]] = AIE.tile(3, 1)
//CHECK: %[[t32:.*]] = AIE.tile(3, 2)
//CHECK: %[[t33:.*]] = AIE.tile(3, 3)
//CHECK: %[[t34:.*]] = AIE.tile(3, 4)

//CHECK-DAG: AIE.flow(%[[t01]], Core : 0, %[[t12]], Core : 0)
//CHECK-DAG: AIE.flow(%[[t02]], DMA : 0, %[[t20]], DMA : 0)
//CHECK-DAG: AIE.flow(%[[t04]], Core : 0, %[[t13]], Core : 0)
//CHECK-DAG: AIE.flow(%[[t11]], Core : 0, %[[t01]], Core : 0)
//CHECK-DAG: AIE.flow(%[[t12]], Core : 0, %[[t02]], Core : 0)
//CHECK-DAG: AIE.flow(%[[t13]], DMA : 0, %[[t20]], DMA : 1)
//CHECK-DAG: AIE.flow(%[[t14]], Core : 0, %[[t04]], Core : 0)
//CHECK-DAG: AIE.flow(%[[t20]], DMA : 0, %[[t11]], DMA : 0)
//CHECK-DAG: AIE.flow(%[[t20]], DMA : 1, %[[t14]], DMA : 0)
//CHECK-DAG: AIE.flow(%[[t21]], Core : 0, %[[t33]], Core : 0)
//CHECK-DAG: AIE.flow(%[[t22]], Core : 0, %[[t34]], Core : 0)
//CHECK-DAG: AIE.flow(%[[t23]], Core : 1, %[[t34]], Core : 1)
//CHECK-DAG: AIE.flow(%[[t23]], DMA : 0, %[[t30]], DMA : 0)
//CHECK-DAG: AIE.flow(%[[t24]], Core : 0, %[[t23]], Core : 0)
//CHECK-DAG: AIE.flow(%[[t24]], Core : 1, %[[t33]], Core : 1)
//CHECK-DAG: AIE.flow(%[[t30]], DMA : 0, %[[t21]], DMA : 0)
//CHECK-DAG: AIE.flow(%[[t30]], DMA : 1, %[[t31]], DMA : 1)
//CHECK-DAG: AIE.flow(%[[t31]], Core : 1, %[[t23]], Core : 1)
//CHECK-DAG: AIE.flow(%[[t32]], DMA : 1, %[[t30]], DMA : 1)
//CHECK-DAG: AIE.flow(%[[t33]], Core : 0, %[[t22]], Core : 0)
//CHECK-DAG: AIE.flow(%[[t33]], Core : 1, %[[t32]], Core : 1)
//CHECK-DAG: AIE.flow(%[[t34]], Core : 0, %[[t24]], Core : 0)
//CHECK-DAG: AIE.flow(%[[t34]], Core : 1, %[[t24]], Core : 1)

module {
    AIE.device(xcvc1902) {
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
        %t31 = AIE.tile(3, 1)
        %t32 = AIE.tile(3, 2)
        %t33 = AIE.tile(3, 3)
        %t34 = AIE.tile(3, 4)

        //TASK 1
        AIE.flow(%t20, DMA : 0, %t11, DMA : 0)
        AIE.flow(%t11, Core : 0, %t01, Core : 0)
        AIE.flow(%t01, Core : 0, %t12, Core : 0)
        AIE.flow(%t12, Core : 0, %t02, Core : 0)
        AIE.flow(%t02, DMA : 0, %t20, DMA : 0)

        //TASK 2
        AIE.flow(%t20, DMA : 1, %t14, DMA : 0)
        AIE.flow(%t14, Core : 0, %t04, Core : 0)
        AIE.flow(%t04, Core : 0, %t13, Core : 0)
        AIE.flow(%t13, DMA : 0, %t20, DMA : 1)

        //TASK 3
        AIE.flow(%t30, DMA : 0, %t21, DMA : 0)
        AIE.flow(%t21, Core : 0, %t33, Core : 0)
        AIE.flow(%t33, Core : 0, %t22, Core : 0)
        AIE.flow(%t22, Core : 0, %t34, Core : 0)
        AIE.flow(%t34, Core : 0, %t24, Core : 0)
        AIE.flow(%t24, Core : 0, %t23, Core : 0)
        AIE.flow(%t23, DMA : 0, %t30, DMA : 0)

        //TASK 4
        AIE.flow(%t30, DMA : 1, %t31, DMA : 1)
        AIE.flow(%t31, Core : 1, %t23, Core : 1)
        AIE.flow(%t23, Core : 1, %t34, Core : 1)
        AIE.flow(%t34, Core : 1, %t24, Core : 1)
        AIE.flow(%t24, Core : 1, %t33, Core : 1)
        AIE.flow(%t33, Core : 1, %t32, Core : 1)
        AIE.flow(%t32, DMA : 1, %t30, DMA : 1)
    }
}