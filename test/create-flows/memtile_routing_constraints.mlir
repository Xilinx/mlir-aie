//===- memtile_routing_constraints.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcve2802) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(2, 4)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(2, 3)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_4:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_5:.*]] = AIE.switchbox(%[[VAL_3]]) {
// CHECK:             AIE.connect<North : 0, DMA : 0>
// CHECK:             AIE.connect<North : 1, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = AIE.switchbox(%[[VAL_2]]) {
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = AIE.switchbox(%[[VAL_4]]) {
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = AIE.shimmux(%[[VAL_4]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = AIE.switchbox(%[[VAL_1]]) {
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_8]] : North, %[[VAL_7]] : South)
// CHECK:           AIE.wire(%[[VAL_4]] : DMA, %[[VAL_8]] : DMA)
// CHECK:           AIE.wire(%[[VAL_3]] : Core, %[[VAL_5]] : Core)
// CHECK:           AIE.wire(%[[VAL_3]] : DMA, %[[VAL_5]] : DMA)
// CHECK:           AIE.wire(%[[VAL_7]] : North, %[[VAL_5]] : South)
// CHECK:           AIE.wire(%[[VAL_2]] : Core, %[[VAL_6]] : Core)
// CHECK:           AIE.wire(%[[VAL_2]] : DMA, %[[VAL_6]] : DMA)
// CHECK:           AIE.wire(%[[VAL_5]] : North, %[[VAL_6]] : South)
// CHECK:           AIE.wire(%[[VAL_1]] : Core, %[[VAL_9]] : Core)
// CHECK:           AIE.wire(%[[VAL_1]] : DMA, %[[VAL_9]] : DMA)
// CHECK:           AIE.wire(%[[VAL_6]] : North, %[[VAL_9]] : South)
// CHECK:         }



module {
    AIE.device(xcve2802) {
        %t04 = AIE.tile(2, 4)
        %t03 = AIE.tile(2, 3)
        %t02 = AIE.tile(2, 2)
        %t01 = AIE.tile(2, 1)
        %t00 = AIE.tile(2, 0)

        AIE.flow(%t02, DMA : 0, %t01, DMA : 0)
        AIE.flow(%t03, DMA : 0, %t00, DMA : 0)
    }
}
