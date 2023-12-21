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

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_2:.*]] = aie.switchbox(%[[VAL_1]]) {
// CHECK:             aie.connect<North : 0, Core : 1>
// CHECK:             aie.connect<Core : 0, Core : 0>
// CHECK:             aie.connect<Core : 1, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_3:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:             aie.connect<South : 0, Core : 1>
// CHECK:           }
// CHECK:           aie.wire(%[[VAL_1]] : Core, %[[VAL_4:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_1]] : DMA, %[[VAL_4]] : DMA)
// CHECK:           aie.wire(%[[VAL_0]] : Core, %[[VAL_5:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_0]] : DMA, %[[VAL_5]] : DMA)
// CHECK:           aie.wire(%[[VAL_4]] : North, %[[VAL_5]] : South)
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %t23 = aie.tile(2, 3)
    %t22 = aie.tile(2, 2)
    aie.flow(%t23, Core : 0, %t22, Core : 1)
    aie.flow(%t22, Core : 0, %t22, Core : 0)
    aie.flow(%t22, Core : 1, %t23, Core : 1)
  }
}
