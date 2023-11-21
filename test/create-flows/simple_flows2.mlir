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

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(2, 3)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(1, 1)
// CHECK:           %[[VAL_3:.*]] = AIE.switchbox(%[[VAL_1]]) {
// CHECK:             AIE.connect<North : 0, Core : 1>
// CHECK:             AIE.connect<Core : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_4:.*]] = AIE.switchbox(%[[VAL_0]]) {
// CHECK:             AIE.connect<Core : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = AIE.switchbox(%[[VAL_2]]) {
// CHECK:             AIE.connect<East : 0, Core : 0>
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_7:.*]] = AIE.switchbox(%[[VAL_6]]) {
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_2]] : Core, %[[VAL_5]] : Core)
// CHECK:           AIE.wire(%[[VAL_2]] : DMA, %[[VAL_5]] : DMA)
// CHECK:           AIE.wire(%[[VAL_5]] : East, %[[VAL_7]] : West)
// CHECK:           AIE.wire(%[[VAL_6]] : Core, %[[VAL_7]] : Core)
// CHECK:           AIE.wire(%[[VAL_6]] : DMA, %[[VAL_7]] : DMA)
// CHECK:           AIE.wire(%[[VAL_1]] : Core, %[[VAL_3]] : Core)
// CHECK:           AIE.wire(%[[VAL_1]] : DMA, %[[VAL_3]] : DMA)
// CHECK:           AIE.wire(%[[VAL_7]] : North, %[[VAL_3]] : South)
// CHECK:           AIE.wire(%[[VAL_0]] : Core, %[[VAL_4]] : Core)
// CHECK:           AIE.wire(%[[VAL_0]] : DMA, %[[VAL_4]] : DMA)
// CHECK:           AIE.wire(%[[VAL_3]] : North, %[[VAL_4]] : South)
// CHECK:         }


module {
  AIE.device(xcvc1902) {
    %t23 = AIE.tile(2, 3)
    %t22 = AIE.tile(2, 2)
    %t11 = AIE.tile(1, 1)
    AIE.flow(%t23, Core : 0, %t22, Core : 1)
    AIE.flow(%t22, Core : 0, %t11, Core : 0)
  }
}
