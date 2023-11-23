//===- memtile.mlir --------------------------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_0:.*]] = AIE.tile(0, 4)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(0, 3)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(0, 2)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(0, 1)
// CHECK:           %[[VAL_4:.*]] = AIE.switchbox(%[[VAL_3]]) {
// CHECK:             AIE.connect<DMA : 0, North : 0>
// CHECK:             AIE.connect<DMA : 1, North : 1>
// CHECK:             AIE.connect<North : 0, DMA : 0>
// CHECK:             AIE.connect<North : 1, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = AIE.switchbox(%[[VAL_2]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<South : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<DMA : 1, South : 1>
// CHECK:             AIE.connect<DMA : 2, North : 0>
// CHECK:             AIE.connect<DMA : 3, North : 1>
// CHECK:             AIE.connect<North : 0, DMA : 2>
// CHECK:             AIE.connect<North : 1, DMA : 3>
// CHECK:             AIE.connect<DMA : 4, North : 2>
// CHECK:             AIE.connect<DMA : 5, North : 3>
// CHECK:             AIE.connect<North : 2, DMA : 4>
// CHECK:             AIE.connect<North : 3, DMA : 5>
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = AIE.switchbox(%[[VAL_1]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<South : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<DMA : 1, South : 1>
// CHECK:             AIE.connect<South : 2, North : 0>
// CHECK:             AIE.connect<South : 3, North : 1>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:             AIE.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = AIE.switchbox(%[[VAL_0]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<South : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<DMA : 1, South : 1>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_3]] : Core, %[[VAL_8:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_3]] : DMA, %[[VAL_8]] : DMA)
// CHECK:           AIE.wire(%[[VAL_2]] : Core, %[[VAL_9:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_2]] : DMA, %[[VAL_9]] : DMA)
// CHECK:           AIE.wire(%[[VAL_8]] : North, %[[VAL_9]] : South)
// CHECK:           AIE.wire(%[[VAL_1]] : Core, %[[VAL_10:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_1]] : DMA, %[[VAL_10]] : DMA)
// CHECK:           AIE.wire(%[[VAL_9]] : North, %[[VAL_10]] : South)
// CHECK:           AIE.wire(%[[VAL_0]] : Core, %[[VAL_11:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_0]] : DMA, %[[VAL_11]] : DMA)
// CHECK:           AIE.wire(%[[VAL_10]] : North, %[[VAL_11]] : South)
// CHECK:         }

module {
    AIE.device(xcve2802) {
        %t04 = AIE.tile(0, 4)
        %t03 = AIE.tile(0, 3)
        %t02 = AIE.tile(0, 2)
        %t01 = AIE.tile(0, 1)

        AIE.flow(%t01, DMA : 0, %t02, DMA : 0)
        AIE.flow(%t01, DMA : 1, %t02, DMA : 1)
        AIE.flow(%t02, DMA : 0, %t01, DMA : 0)
        AIE.flow(%t02, DMA : 1, %t01, DMA : 1)

        AIE.flow(%t02, DMA : 2, %t03, DMA : 0)
        AIE.flow(%t02, DMA : 3, %t03, DMA : 1)
        AIE.flow(%t03, DMA : 0, %t02, DMA : 2)
        AIE.flow(%t03, DMA : 1, %t02, DMA : 3)

        AIE.flow(%t02, DMA : 4, %t04, DMA : 0)
        AIE.flow(%t02, DMA : 5, %t04, DMA : 1)
        AIE.flow(%t04, DMA : 0, %t02, DMA : 4)
        AIE.flow(%t04, DMA : 1, %t02, DMA : 5)
    }
}

