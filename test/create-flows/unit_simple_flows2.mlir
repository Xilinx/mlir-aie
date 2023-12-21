//===- simple_flows2.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_2:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_3:.*]] = aie.switchbox(%[[VAL_1]]) {
// CHECK:             aie.connect<North : 0, Core : 1>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_4:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_7:.*]] = aie.switchbox(%[[VAL_6]]) {
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[VAL_2]] : Core, %[[VAL_8:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_2]] : DMA, %[[VAL_8]] : DMA)
// CHECK:           aie.wire(%[[VAL_8]] : East, %[[VAL_9:.*]] : West)
// CHECK:           aie.wire(%[[VAL_6]] : Core, %[[VAL_9]] : Core)
// CHECK:           aie.wire(%[[VAL_6]] : DMA, %[[VAL_9]] : DMA)
// CHECK:           aie.wire(%[[VAL_1]] : Core, %[[VAL_10:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_1]] : DMA, %[[VAL_10]] : DMA)
// CHECK:           aie.wire(%[[VAL_9]] : North, %[[VAL_10]] : South)
// CHECK:           aie.wire(%[[VAL_0]] : Core, %[[VAL_11:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_0]] : DMA, %[[VAL_11]] : DMA)
// CHECK:           aie.wire(%[[VAL_10]] : North, %[[VAL_11]] : South)
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %t23 = aie.tile(2, 3)
    %t22 = aie.tile(2, 2)
    %t11 = aie.tile(1, 1)
    aie.flow(%t23, Core : 0, %t22, Core : 1)
    aie.flow(%t22, Core : 0, %t11, Core : 0)
  }
}
