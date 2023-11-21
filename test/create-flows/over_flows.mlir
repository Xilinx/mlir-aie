//===- over_flows.mlir -----------------------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_0:.*]] = AIE.tile(0, 3)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(0, 2)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(0, 0)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(1, 3)
// CHECK:           %[[VAL_4:.*]] = AIE.tile(1, 1)
// CHECK:           %[[VAL_5:.*]] = AIE.tile(1, 0)
// CHECK:           %[[VAL_6:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_7:.*]] = AIE.tile(3, 0)
// CHECK:           %[[VAL_8:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_9:.*]] = AIE.tile(3, 1)
// CHECK:           %[[VAL_10:.*]] = AIE.tile(6, 0)
// CHECK:           %[[VAL_11:.*]] = AIE.tile(7, 0)
// CHECK:           %[[VAL_12:.*]] = AIE.tile(7, 1)
// CHECK:           %[[VAL_13:.*]] = AIE.tile(7, 2)
// CHECK:           %[[VAL_14:.*]] = AIE.tile(7, 3)
// CHECK:           %[[VAL_15:.*]] = AIE.tile(8, 0)
// CHECK:           %[[VAL_16:.*]] = AIE.tile(8, 2)
// CHECK:           %[[VAL_17:.*]] = AIE.tile(8, 3)
// CHECK:           %[[VAL_18:.*]] = AIE.switchbox(%[[VAL_6]]) {
// CHECK:             AIE.connect<East : 0, South : 2>
// CHECK:             AIE.connect<East : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_19:.*]] = AIE.shimmux(%[[VAL_6]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = AIE.switchbox(%[[VAL_7]]) {
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<North : 1, West : 1>
// CHECK:             AIE.connect<East : 0, South : 2>
// CHECK:             AIE.connect<East : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = AIE.switchbox(%[[VAL_9]]) {
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:             AIE.connect<East : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = AIE.tile(4, 1)
// CHECK:           %[[VAL_23:.*]] = AIE.switchbox(%[[VAL_22]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, South : 0>
// CHECK:             AIE.connect<East : 3, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = AIE.tile(5, 1)
// CHECK:           %[[VAL_25:.*]] = AIE.switchbox(%[[VAL_24]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = AIE.tile(6, 1)
// CHECK:           %[[VAL_27:.*]] = AIE.switchbox(%[[VAL_26]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<North : 0, West : 2>
// CHECK:             AIE.connect<North : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = AIE.switchbox(%[[VAL_12]]) {
// CHECK:             AIE.connect<DMA : 0, West : 0>
// CHECK:             AIE.connect<DMA : 1, West : 1>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<North : 1, South : 1>
// CHECK:             AIE.connect<North : 2, South : 2>
// CHECK:             AIE.connect<North : 3, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = AIE.switchbox(%[[VAL_10]]) {
// CHECK:             AIE.connect<East : 0, South : 2>
// CHECK:             AIE.connect<East : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = AIE.shimmux(%[[VAL_10]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = AIE.switchbox(%[[VAL_11]]) {
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<North : 1, West : 1>
// CHECK:             AIE.connect<North : 2, South : 2>
// CHECK:             AIE.connect<North : 3, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = AIE.switchbox(%[[VAL_13]]) {
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<DMA : 1, South : 1>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:             AIE.connect<North : 1, South : 3>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = AIE.shimmux(%[[VAL_11]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = AIE.switchbox(%[[VAL_14]]) {
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<DMA : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = AIE.shimmux(%[[VAL_7]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = AIE.tile(4, 0)
// CHECK:           %[[VAL_37:.*]] = AIE.switchbox(%[[VAL_36]]) {
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<North : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_38:.*]] = AIE.tile(6, 2)
// CHECK:           %[[VAL_39:.*]] = AIE.switchbox(%[[VAL_38]]) {
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:             AIE.connect<East : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_40:.*]] = AIE.switchbox(%[[VAL_16]]) {
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<North : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = AIE.switchbox(%[[VAL_17]]) {
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<DMA : 1, South : 1>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_19]] : North, %[[VAL_18]] : South)
// CHECK:           AIE.wire(%[[VAL_6]] : DMA, %[[VAL_19]] : DMA)
// CHECK:           AIE.wire(%[[VAL_18]] : East, %[[VAL_20]] : West)
// CHECK:           AIE.wire(%[[VAL_35]] : North, %[[VAL_20]] : South)
// CHECK:           AIE.wire(%[[VAL_7]] : DMA, %[[VAL_35]] : DMA)
// CHECK:           AIE.wire(%[[VAL_9]] : Core, %[[VAL_21]] : Core)
// CHECK:           AIE.wire(%[[VAL_9]] : DMA, %[[VAL_21]] : DMA)
// CHECK:           AIE.wire(%[[VAL_20]] : North, %[[VAL_21]] : South)
// CHECK:           AIE.wire(%[[VAL_20]] : East, %[[VAL_37]] : West)
// CHECK:           AIE.wire(%[[VAL_21]] : East, %[[VAL_23]] : West)
// CHECK:           AIE.wire(%[[VAL_22]] : Core, %[[VAL_23]] : Core)
// CHECK:           AIE.wire(%[[VAL_22]] : DMA, %[[VAL_23]] : DMA)
// CHECK:           AIE.wire(%[[VAL_37]] : North, %[[VAL_23]] : South)
// CHECK:           AIE.wire(%[[VAL_23]] : East, %[[VAL_25]] : West)
// CHECK:           AIE.wire(%[[VAL_24]] : Core, %[[VAL_25]] : Core)
// CHECK:           AIE.wire(%[[VAL_24]] : DMA, %[[VAL_25]] : DMA)
// CHECK:           AIE.wire(%[[VAL_30]] : North, %[[VAL_29]] : South)
// CHECK:           AIE.wire(%[[VAL_10]] : DMA, %[[VAL_30]] : DMA)
// CHECK:           AIE.wire(%[[VAL_25]] : East, %[[VAL_27]] : West)
// CHECK:           AIE.wire(%[[VAL_26]] : Core, %[[VAL_27]] : Core)
// CHECK:           AIE.wire(%[[VAL_26]] : DMA, %[[VAL_27]] : DMA)
// CHECK:           AIE.wire(%[[VAL_29]] : North, %[[VAL_27]] : South)
// CHECK:           AIE.wire(%[[VAL_38]] : Core, %[[VAL_39]] : Core)
// CHECK:           AIE.wire(%[[VAL_38]] : DMA, %[[VAL_39]] : DMA)
// CHECK:           AIE.wire(%[[VAL_27]] : North, %[[VAL_39]] : South)
// CHECK:           AIE.wire(%[[VAL_29]] : East, %[[VAL_31]] : West)
// CHECK:           AIE.wire(%[[VAL_33]] : North, %[[VAL_31]] : South)
// CHECK:           AIE.wire(%[[VAL_11]] : DMA, %[[VAL_33]] : DMA)
// CHECK:           AIE.wire(%[[VAL_27]] : East, %[[VAL_28]] : West)
// CHECK:           AIE.wire(%[[VAL_12]] : Core, %[[VAL_28]] : Core)
// CHECK:           AIE.wire(%[[VAL_12]] : DMA, %[[VAL_28]] : DMA)
// CHECK:           AIE.wire(%[[VAL_31]] : North, %[[VAL_28]] : South)
// CHECK:           AIE.wire(%[[VAL_39]] : East, %[[VAL_32]] : West)
// CHECK:           AIE.wire(%[[VAL_13]] : Core, %[[VAL_32]] : Core)
// CHECK:           AIE.wire(%[[VAL_13]] : DMA, %[[VAL_32]] : DMA)
// CHECK:           AIE.wire(%[[VAL_28]] : North, %[[VAL_32]] : South)
// CHECK:           AIE.wire(%[[VAL_14]] : Core, %[[VAL_34]] : Core)
// CHECK:           AIE.wire(%[[VAL_14]] : DMA, %[[VAL_34]] : DMA)
// CHECK:           AIE.wire(%[[VAL_32]] : North, %[[VAL_34]] : South)
// CHECK:           AIE.wire(%[[VAL_32]] : East, %[[VAL_40]] : West)
// CHECK:           AIE.wire(%[[VAL_16]] : Core, %[[VAL_40]] : Core)
// CHECK:           AIE.wire(%[[VAL_16]] : DMA, %[[VAL_40]] : DMA)
// CHECK:           AIE.wire(%[[VAL_34]] : East, %[[VAL_41]] : West)
// CHECK:           AIE.wire(%[[VAL_17]] : Core, %[[VAL_41]] : Core)
// CHECK:           AIE.wire(%[[VAL_17]] : DMA, %[[VAL_41]] : DMA)
// CHECK:           AIE.wire(%[[VAL_40]] : North, %[[VAL_41]] : South)
// CHECK:         }

//

module {
    AIE.device(xcvc1902) {
        %t03 = AIE.tile(0, 3)
        %t02 = AIE.tile(0, 2)
        %t00 = AIE.tile(0, 0)
        %t13 = AIE.tile(1, 3)
        %t11 = AIE.tile(1, 1)
        %t10 = AIE.tile(1, 0)
        %t20 = AIE.tile(2, 0)
        %t30 = AIE.tile(3, 0)
        %t22 = AIE.tile(2, 2)
        %t31 = AIE.tile(3, 1)
        %t60 = AIE.tile(6, 0)
        %t70 = AIE.tile(7, 0)
        %t71 = AIE.tile(7, 1)
        %t72 = AIE.tile(7, 2)
        %t73 = AIE.tile(7, 3)
        %t80 = AIE.tile(8, 0)
        %t82 = AIE.tile(8, 2)
        %t83 = AIE.tile(8, 3)

        AIE.flow(%t71, DMA : 0, %t20, DMA : 0)
        AIE.flow(%t71, DMA : 1, %t20, DMA : 1)
        AIE.flow(%t72, DMA : 0, %t60, DMA : 0)
        AIE.flow(%t72, DMA : 1, %t60, DMA : 1)
        AIE.flow(%t73, DMA : 0, %t70, DMA : 0)
        AIE.flow(%t73, DMA : 1, %t70, DMA : 1)
        AIE.flow(%t83, DMA : 0, %t30, DMA : 0)
        AIE.flow(%t83, DMA : 1, %t30, DMA : 1)
    }
}

