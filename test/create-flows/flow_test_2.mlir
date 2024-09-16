//===- flow_test_2.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s -o %t.opt
// RUN: FileCheck %s --check-prefix=CHECK1 < %t.opt
// RUN: aie-translate --aie-flows-to-json %t.opt | FileCheck %s --check-prefix=CHECK2

// CHECK1: %[[t01:.*]] = aie.tile(0, 1)
// CHECK1: %[[t02:.*]] = aie.tile(0, 2)
// CHECK1: %[[t03:.*]] = aie.tile(0, 3)
// CHECK1: %[[t04:.*]] = aie.tile(0, 4)
// CHECK1: %[[t11:.*]] = aie.tile(1, 1)
// CHECK1: %[[t12:.*]] = aie.tile(1, 2)
// CHECK1: %[[t13:.*]] = aie.tile(1, 3)
// CHECK1: %[[t14:.*]] = aie.tile(1, 4)
// CHECK1: %[[t20:.*]] = aie.tile(2, 0)
// CHECK1: %[[t21:.*]] = aie.tile(2, 1)
// CHECK1: %[[t22:.*]] = aie.tile(2, 2)
// CHECK1: %[[t23:.*]] = aie.tile(2, 3)
// CHECK1: %[[t24:.*]] = aie.tile(2, 4)
// CHECK1: %[[t30:.*]] = aie.tile(3, 0)
// CHECK1: %[[t31:.*]] = aie.tile(3, 1)
// CHECK1: %[[t32:.*]] = aie.tile(3, 2)
// CHECK1: %[[t33:.*]] = aie.tile(3, 3)
// CHECK1: %[[t34:.*]] = aie.tile(3, 4)

// CHECK1: aie.flow(%[[t01]], Core : 0, %[[t12]], Core : 0)
// CHECK1: aie.flow(%[[t02]], DMA : 0, %[[t20]], DMA : 0)
// CHECK1: aie.flow(%[[t04]], Core : 0, %[[t13]], Core : 0)
// CHECK1: aie.flow(%[[t11]], Core : 0, %[[t01]], Core : 0)
// CHECK1: aie.flow(%[[t12]], Core : 0, %[[t02]], Core : 0)
// CHECK1: aie.flow(%[[t13]], DMA : 0, %[[t20]], DMA : 1)
// CHECK1: aie.flow(%[[t14]], Core : 0, %[[t04]], Core : 0)
// CHECK1: aie.flow(%[[t20]], DMA : 0, %[[t11]], DMA : 0)
// CHECK1: aie.flow(%[[t20]], DMA : 1, %[[t14]], DMA : 0)
// CHECK1: aie.flow(%[[t21]], Core : 0, %[[t33]], Core : 0)
// CHECK1: aie.flow(%[[t22]], Core : 0, %[[t34]], Core : 0)
// CHECK1: aie.flow(%[[t23]], Core : 1, %[[t34]], Core : 1)
// CHECK1: aie.flow(%[[t23]], DMA : 0, %[[t30]], DMA : 0)
// CHECK1: aie.flow(%[[t24]], Core : 0, %[[t23]], Core : 0)
// CHECK1: aie.flow(%[[t24]], Core : 1, %[[t33]], Core : 1)
// CHECK1: aie.flow(%[[t30]], DMA : 0, %[[t21]], DMA : 0)
// CHECK1: aie.flow(%[[t30]], DMA : 1, %[[t31]], DMA : 1)
// CHECK1: aie.flow(%[[t31]], Core : 1, %[[t23]], Core : 1)
// CHECK1: aie.flow(%[[t32]], DMA : 1, %[[t30]], DMA : 1)
// CHECK1: aie.flow(%[[t33]], Core : 0, %[[t22]], Core : 0)
// CHECK1: aie.flow(%[[t33]], Core : 1, %[[t32]], Core : 1)
// CHECK1: aie.flow(%[[t34]], Core : 0, %[[t24]], Core : 0)
// CHECK1: aie.flow(%[[t34]], Core : 1, %[[t24]], Core : 1)

// CHECK2: "total_path_length": 50

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
        %t31 = aie.tile(3, 1)
        %t32 = aie.tile(3, 2)
        %t33 = aie.tile(3, 3)
        %t34 = aie.tile(3, 4)

        //TASK 1
        aie.flow(%t20, DMA : 0, %t11, DMA : 0)
        aie.flow(%t11, Core : 0, %t01, Core : 0)
        aie.flow(%t01, Core : 0, %t12, Core : 0)
        aie.flow(%t12, Core : 0, %t02, Core : 0)
        aie.flow(%t02, DMA : 0, %t20, DMA : 0)

        //TASK 2
        aie.flow(%t20, DMA : 1, %t14, DMA : 0)
        aie.flow(%t14, Core : 0, %t04, Core : 0)
        aie.flow(%t04, Core : 0, %t13, Core : 0)
        aie.flow(%t13, DMA : 0, %t20, DMA : 1)

        //TASK 3
        aie.flow(%t30, DMA : 0, %t21, DMA : 0)
        aie.flow(%t21, Core : 0, %t33, Core : 0)
        aie.flow(%t33, Core : 0, %t22, Core : 0)
        aie.flow(%t22, Core : 0, %t34, Core : 0)
        aie.flow(%t34, Core : 0, %t24, Core : 0)
        aie.flow(%t24, Core : 0, %t23, Core : 0)
        aie.flow(%t23, DMA : 0, %t30, DMA : 0)

        //TASK 4
        aie.flow(%t30, DMA : 1, %t31, DMA : 1)
        aie.flow(%t31, Core : 1, %t23, Core : 1)
        aie.flow(%t23, Core : 1, %t34, Core : 1)
        aie.flow(%t34, Core : 1, %t24, Core : 1)
        aie.flow(%t24, Core : 1, %t33, Core : 1)
        aie.flow(%t33, Core : 1, %t32, Core : 1)
        aie.flow(%t32, DMA : 1, %t30, DMA : 1)
    }
}