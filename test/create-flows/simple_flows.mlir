//===- simple_flows.mlir ---------------------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_2:.*]] = AIE.switchbox(%[[VAL_1]]) {
// CHECK:             AIE.connect<North : 0, Core : 1>
// CHECK:             AIE.connect<Core : 0, Core : 0>
// CHECK:             AIE.connect<Core : 1, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_3:.*]] = AIE.switchbox(%[[VAL_0]]) {
// CHECK:             AIE.connect<Core : 0, South : 0>
// CHECK:             AIE.connect<South : 0, Core : 1>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_1]] : Core, %[[VAL_2]] : Core)
// CHECK:           AIE.wire(%[[VAL_1]] : DMA, %[[VAL_2]] : DMA)
// CHECK:           AIE.wire(%[[VAL_0]] : Core, %[[VAL_3]] : Core)
// CHECK:           AIE.wire(%[[VAL_0]] : DMA, %[[VAL_3]] : DMA)
// CHECK:           AIE.wire(%[[VAL_2]] : North, %[[VAL_3]] : South)
// CHECK:         }


module {
  AIE.device(xcvc1902) {
    %t23 = AIE.tile(2, 3)
    %t22 = AIE.tile(2, 2)
    AIE.flow(%t23, Core : 0, %t22, Core : 1)
    AIE.flow(%t22, Core : 0, %t22, Core : 0)
    AIE.flow(%t22, Core : 1, %t23, Core : 1)
  }
}
