//===- flow_test_1.mlir ----------------------------------------*- MLIR -*-===//
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

// CHECK1: %[[t20:.*]] = aie.tile(2, 0)
// CHECK1: %[[t30:.*]] = aie.tile(3, 0)
// CHECK1: %[[t34:.*]] = aie.tile(3, 4)
// CHECK1: %[[t43:.*]] = aie.tile(4, 3)
// CHECK1: %[[t44:.*]] = aie.tile(4, 4)
// CHECK1: %[[t54:.*]] = aie.tile(5, 4)
// CHECK1: %[[t60:.*]] = aie.tile(6, 0)
// CHECK1: %[[t63:.*]] = aie.tile(6, 3)
// CHECK1: %[[t70:.*]] = aie.tile(7, 0)
// CHECK1: %[[t72:.*]] = aie.tile(7, 2)
// CHECK1: %[[t83:.*]] = aie.tile(8, 3)
// CHECK1: %[[t84:.*]] = aie.tile(8, 4)

// CHECK1: aie.flow(%[[t20]], DMA : 0, %[[t63]], DMA : 0)
// CHECK1: aie.flow(%[[t20]], DMA : 1, %[[t83]], DMA : 0)
// CHECK1: aie.flow(%[[t30]], DMA : 0, %[[t72]], DMA : 0)
// CHECK1: aie.flow(%[[t30]], DMA : 1, %[[t54]], DMA : 0)

// CHECK1: aie.flow(%[[t34]], Core : 0, %[[t63]], Core : 1)
// CHECK1: aie.flow(%[[t34]], DMA : 1, %[[t70]], DMA : 0)
// CHECK1: aie.flow(%[[t43]], Core : 0, %[[t84]], Core : 1)
// CHECK1: aie.flow(%[[t43]], DMA : 1, %[[t60]], DMA : 1)

// CHECK1: aie.flow(%[[t44]], Core : 0, %[[t54]], Core : 1)
// CHECK1: aie.flow(%[[t44]], DMA : 1, %[[t60]], DMA : 0)
// CHECK1: aie.flow(%[[t54]], Core : 0, %[[t43]], Core : 1)
// CHECK1: aie.flow(%[[t54]], DMA : 1, %[[t30]], DMA : 1)

// CHECK1: aie.flow(%[[t60]], DMA : 0, %[[t44]], DMA : 0)
// CHECK1: aie.flow(%[[t60]], DMA : 1, %[[t43]], DMA : 0)
// CHECK1: aie.flow(%[[t63]], Core : 0, %[[t34]], Core : 1)
// CHECK1: aie.flow(%[[t63]], DMA : 1, %[[t20]], DMA : 1)

// CHECK1: aie.flow(%[[t70]], DMA : 0, %[[t34]], DMA : 0)
// CHECK1: aie.flow(%[[t70]], DMA : 1, %[[t84]], DMA : 0)
// CHECK1: aie.flow(%[[t72]], Core : 0, %[[t83]], Core : 1)
// CHECK1: aie.flow(%[[t72]], DMA : 1, %[[t30]], DMA : 0)

// CHECK1: aie.flow(%[[t83]], Core : 0, %[[t44]], Core : 1)
// CHECK1: aie.flow(%[[t83]], DMA : 1, %[[t20]], DMA : 0)
// CHECK1: aie.flow(%[[t84]], Core : 0, %[[t72]], Core : 1)
// CHECK1: aie.flow(%[[t84]], DMA : 1, %[[t70]], DMA : 1)

// CHECK2: "total_path_length": 130

module {
    aie.device(xcvc1902) {
        %t20 = aie.tile(2, 0)
        %t30 = aie.tile(3, 0)
        %t34 = aie.tile(3, 4)
        %t43 = aie.tile(4, 3)
        %t44 = aie.tile(4, 4)
        %t54 = aie.tile(5, 4)
        %t60 = aie.tile(6, 0)
        %t63 = aie.tile(6, 3)
        %t70 = aie.tile(7, 0)
        %t72 = aie.tile(7, 2)
        %t83 = aie.tile(8, 3)
        %t84 = aie.tile(8, 4)

        aie.flow(%t20, DMA : 0, %t63, DMA : 0)
        aie.flow(%t20, DMA : 1, %t83, DMA : 0)
        aie.flow(%t30, DMA : 0, %t72, DMA : 0)
        aie.flow(%t30, DMA : 1, %t54, DMA : 0)

        aie.flow(%t34, Core : 0, %t63, Core : 1)
        aie.flow(%t34, DMA : 1, %t70, DMA : 0)
        aie.flow(%t43, Core : 0, %t84, Core : 1)
        aie.flow(%t43, DMA : 1, %t60, DMA : 1)

        aie.flow(%t44, Core : 0, %t54, Core : 1)
        aie.flow(%t44, DMA : 1, %t60, DMA : 0)
        aie.flow(%t54, Core : 0, %t43, Core : 1)
        aie.flow(%t54, DMA : 1, %t30, DMA : 1)

        aie.flow(%t60, DMA : 0, %t44, DMA : 0)
        aie.flow(%t60, DMA : 1, %t43, DMA : 0)
        aie.flow(%t63, Core : 0, %t34, Core : 1)
        aie.flow(%t63, DMA : 1, %t20, DMA : 1)

        aie.flow(%t70, DMA : 0, %t34, DMA : 0)
        aie.flow(%t70, DMA : 1, %t84, DMA : 0)
        aie.flow(%t72, Core : 0, %t83, Core : 1)
        aie.flow(%t72, DMA : 1, %t30, DMA : 0)

        aie.flow(%t83, Core : 0, %t44, Core : 1)
        aie.flow(%t83, DMA : 1, %t20, DMA : 0)
        aie.flow(%t84, Core : 0, %t72, Core : 1)
        aie.flow(%t84, DMA : 1, %t70, DMA : 1)
    }
}