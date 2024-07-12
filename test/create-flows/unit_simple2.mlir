//===- simple2.mlir --------------------------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_1:.*]] = aie.tile(3, 2)
// CHECK:           %[[VAL_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_3:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_4:.*]] = aie.switchbox(%[[VAL_1]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             aie.connect<Core : 1, South : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[VAL_2]] : Core, %[[VAL_6:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_2]] : DMA, %[[VAL_6]] : DMA)
// CHECK:           aie.wire(%[[VAL_0]] : Core, %[[VAL_7:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_0]] : DMA, %[[VAL_7]] : DMA)
// CHECK:           aie.wire(%[[VAL_6]] : North, %[[VAL_7]] : South)
// CHECK:           aie.wire(%[[VAL_6]] : East, %[[VAL_8:.*]] : West)
// CHECK:           aie.wire(%[[VAL_1]] : Core, %[[VAL_8]] : Core)
// CHECK:           aie.wire(%[[VAL_1]] : DMA, %[[VAL_8]] : DMA)
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %0 = aie.tile(2, 3)
    %1 = aie.tile(3, 2)
    aie.flow(%0, Core : 1, %1, DMA : 0)
  }
}
