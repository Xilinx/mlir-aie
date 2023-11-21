//===- broadcast.mlir ------------------------------------------*- MLIR -*-===//
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
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 3, East : 0>
// CHECK:             AIE.connect<East : 0, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_19:.*]] = AIE.shimmux(%[[VAL_6]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = AIE.switchbox(%[[VAL_7]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = AIE.tile(4, 0)
// CHECK:           %[[VAL_22:.*]] = AIE.switchbox(%[[VAL_21]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = AIE.tile(5, 0)
// CHECK:           %[[VAL_24:.*]] = AIE.switchbox(%[[VAL_23]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = AIE.switchbox(%[[VAL_10]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<South : 3, West : 0>
// CHECK:             AIE.connect<South : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = AIE.switchbox(%[[VAL_11]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_28:.*]] = AIE.switchbox(%[[VAL_27]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = AIE.switchbox(%[[VAL_9]]) {
// CHECK:             AIE.connect<West : 0, DMA : 0>
// CHECK:             AIE.connect<South : 0, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = AIE.tile(6, 1)
// CHECK:           %[[VAL_31:.*]] = AIE.switchbox(%[[VAL_30]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = AIE.switchbox(%[[VAL_12]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = AIE.switchbox(%[[VAL_8]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<East : 0, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = AIE.tile(6, 2)
// CHECK:           %[[VAL_35:.*]] = AIE.switchbox(%[[VAL_34]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = AIE.switchbox(%[[VAL_13]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = AIE.switchbox(%[[VAL_16]]) {
// CHECK:             AIE.connect<West : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_38:.*]] = AIE.switchbox(%[[VAL_3]]) {
// CHECK:             AIE.connect<East : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_39:.*]] = AIE.tile(2, 3)
// CHECK:           %[[VAL_40:.*]] = AIE.switchbox(%[[VAL_39]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = AIE.shimmux(%[[VAL_10]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_42:.*]] = AIE.tile(0, 1)
// CHECK:           %[[VAL_43:.*]] = AIE.switchbox(%[[VAL_42]]) {
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_44:.*]] = AIE.switchbox(%[[VAL_4]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_45:.*]] = AIE.tile(4, 1)
// CHECK:           %[[VAL_46:.*]] = AIE.switchbox(%[[VAL_45]]) {
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_47:.*]] = AIE.tile(5, 1)
// CHECK:           %[[VAL_48:.*]] = AIE.switchbox(%[[VAL_47]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_49:.*]] = AIE.switchbox(%[[VAL_1]]) {
// CHECK:             AIE.connect<South : 0, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_50:.*]] = AIE.tile(3, 2)
// CHECK:           %[[VAL_51:.*]] = AIE.switchbox(%[[VAL_50]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_52:.*]] = AIE.tile(4, 2)
// CHECK:           %[[VAL_53:.*]] = AIE.switchbox(%[[VAL_52]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = AIE.switchbox(%[[VAL_14]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_55:.*]] = AIE.switchbox(%[[VAL_17]]) {
// CHECK:             AIE.connect<West : 0, DMA : 1>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_42]] : Core, %[[VAL_43]] : Core)
// CHECK:           AIE.wire(%[[VAL_42]] : DMA, %[[VAL_43]] : DMA)
// CHECK:           AIE.wire(%[[VAL_1]] : Core, %[[VAL_49]] : Core)
// CHECK:           AIE.wire(%[[VAL_1]] : DMA, %[[VAL_49]] : DMA)
// CHECK:           AIE.wire(%[[VAL_43]] : North, %[[VAL_49]] : South)
// CHECK:           AIE.wire(%[[VAL_43]] : East, %[[VAL_44]] : West)
// CHECK:           AIE.wire(%[[VAL_4]] : Core, %[[VAL_44]] : Core)
// CHECK:           AIE.wire(%[[VAL_4]] : DMA, %[[VAL_44]] : DMA)
// CHECK:           AIE.wire(%[[VAL_3]] : Core, %[[VAL_38]] : Core)
// CHECK:           AIE.wire(%[[VAL_3]] : DMA, %[[VAL_38]] : DMA)
// CHECK:           AIE.wire(%[[VAL_19]] : North, %[[VAL_18]] : South)
// CHECK:           AIE.wire(%[[VAL_6]] : DMA, %[[VAL_19]] : DMA)
// CHECK:           AIE.wire(%[[VAL_44]] : East, %[[VAL_28]] : West)
// CHECK:           AIE.wire(%[[VAL_27]] : Core, %[[VAL_28]] : Core)
// CHECK:           AIE.wire(%[[VAL_27]] : DMA, %[[VAL_28]] : DMA)
// CHECK:           AIE.wire(%[[VAL_18]] : North, %[[VAL_28]] : South)
// CHECK:           AIE.wire(%[[VAL_8]] : Core, %[[VAL_33]] : Core)
// CHECK:           AIE.wire(%[[VAL_8]] : DMA, %[[VAL_33]] : DMA)
// CHECK:           AIE.wire(%[[VAL_28]] : North, %[[VAL_33]] : South)
// CHECK:           AIE.wire(%[[VAL_38]] : East, %[[VAL_40]] : West)
// CHECK:           AIE.wire(%[[VAL_39]] : Core, %[[VAL_40]] : Core)
// CHECK:           AIE.wire(%[[VAL_39]] : DMA, %[[VAL_40]] : DMA)
// CHECK:           AIE.wire(%[[VAL_33]] : North, %[[VAL_40]] : South)
// CHECK:           AIE.wire(%[[VAL_18]] : East, %[[VAL_20]] : West)
// CHECK:           AIE.wire(%[[VAL_28]] : East, %[[VAL_29]] : West)
// CHECK:           AIE.wire(%[[VAL_9]] : Core, %[[VAL_29]] : Core)
// CHECK:           AIE.wire(%[[VAL_9]] : DMA, %[[VAL_29]] : DMA)
// CHECK:           AIE.wire(%[[VAL_20]] : North, %[[VAL_29]] : South)
// CHECK:           AIE.wire(%[[VAL_33]] : East, %[[VAL_51]] : West)
// CHECK:           AIE.wire(%[[VAL_50]] : Core, %[[VAL_51]] : Core)
// CHECK:           AIE.wire(%[[VAL_50]] : DMA, %[[VAL_51]] : DMA)
// CHECK:           AIE.wire(%[[VAL_29]] : North, %[[VAL_51]] : South)
// CHECK:           AIE.wire(%[[VAL_20]] : East, %[[VAL_22]] : West)
// CHECK:           AIE.wire(%[[VAL_29]] : East, %[[VAL_46]] : West)
// CHECK:           AIE.wire(%[[VAL_45]] : Core, %[[VAL_46]] : Core)
// CHECK:           AIE.wire(%[[VAL_45]] : DMA, %[[VAL_46]] : DMA)
// CHECK:           AIE.wire(%[[VAL_22]] : North, %[[VAL_46]] : South)
// CHECK:           AIE.wire(%[[VAL_51]] : East, %[[VAL_53]] : West)
// CHECK:           AIE.wire(%[[VAL_52]] : Core, %[[VAL_53]] : Core)
// CHECK:           AIE.wire(%[[VAL_52]] : DMA, %[[VAL_53]] : DMA)
// CHECK:           AIE.wire(%[[VAL_46]] : North, %[[VAL_53]] : South)
// CHECK:           AIE.wire(%[[VAL_22]] : East, %[[VAL_24]] : West)
// CHECK:           AIE.wire(%[[VAL_46]] : East, %[[VAL_48]] : West)
// CHECK:           AIE.wire(%[[VAL_47]] : Core, %[[VAL_48]] : Core)
// CHECK:           AIE.wire(%[[VAL_47]] : DMA, %[[VAL_48]] : DMA)
// CHECK:           AIE.wire(%[[VAL_24]] : North, %[[VAL_48]] : South)
// CHECK:           AIE.wire(%[[VAL_24]] : East, %[[VAL_25]] : West)
// CHECK:           AIE.wire(%[[VAL_41]] : North, %[[VAL_25]] : South)
// CHECK:           AIE.wire(%[[VAL_10]] : DMA, %[[VAL_41]] : DMA)
// CHECK:           AIE.wire(%[[VAL_48]] : East, %[[VAL_31]] : West)
// CHECK:           AIE.wire(%[[VAL_30]] : Core, %[[VAL_31]] : Core)
// CHECK:           AIE.wire(%[[VAL_30]] : DMA, %[[VAL_31]] : DMA)
// CHECK:           AIE.wire(%[[VAL_25]] : North, %[[VAL_31]] : South)
// CHECK:           AIE.wire(%[[VAL_34]] : Core, %[[VAL_35]] : Core)
// CHECK:           AIE.wire(%[[VAL_34]] : DMA, %[[VAL_35]] : DMA)
// CHECK:           AIE.wire(%[[VAL_31]] : North, %[[VAL_35]] : South)
// CHECK:           AIE.wire(%[[VAL_25]] : East, %[[VAL_26]] : West)
// CHECK:           AIE.wire(%[[VAL_31]] : East, %[[VAL_32]] : West)
// CHECK:           AIE.wire(%[[VAL_12]] : Core, %[[VAL_32]] : Core)
// CHECK:           AIE.wire(%[[VAL_12]] : DMA, %[[VAL_32]] : DMA)
// CHECK:           AIE.wire(%[[VAL_26]] : North, %[[VAL_32]] : South)
// CHECK:           AIE.wire(%[[VAL_35]] : East, %[[VAL_36]] : West)
// CHECK:           AIE.wire(%[[VAL_13]] : Core, %[[VAL_36]] : Core)
// CHECK:           AIE.wire(%[[VAL_13]] : DMA, %[[VAL_36]] : DMA)
// CHECK:           AIE.wire(%[[VAL_32]] : North, %[[VAL_36]] : South)
// CHECK:           AIE.wire(%[[VAL_14]] : Core, %[[VAL_54]] : Core)
// CHECK:           AIE.wire(%[[VAL_14]] : DMA, %[[VAL_54]] : DMA)
// CHECK:           AIE.wire(%[[VAL_36]] : North, %[[VAL_54]] : South)
// CHECK:           AIE.wire(%[[VAL_36]] : East, %[[VAL_37]] : West)
// CHECK:           AIE.wire(%[[VAL_16]] : Core, %[[VAL_37]] : Core)
// CHECK:           AIE.wire(%[[VAL_16]] : DMA, %[[VAL_37]] : DMA)
// CHECK:           AIE.wire(%[[VAL_54]] : East, %[[VAL_55]] : West)
// CHECK:           AIE.wire(%[[VAL_17]] : Core, %[[VAL_55]] : Core)
// CHECK:           AIE.wire(%[[VAL_17]] : DMA, %[[VAL_55]] : DMA)
// CHECK:           AIE.wire(%[[VAL_37]] : North, %[[VAL_55]] : South)
// CHECK:         }

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

        AIE.flow(%t20, DMA : 0, %t13, DMA : 0)
        AIE.flow(%t20, DMA : 0, %t31, DMA : 0)
        AIE.flow(%t20, DMA : 0, %t71, DMA : 0)
        AIE.flow(%t20, DMA : 0, %t82, DMA : 0)

        AIE.flow(%t60, DMA : 0, %t02, DMA : 1)
        AIE.flow(%t60, DMA : 0, %t83, DMA : 1)
        AIE.flow(%t60, DMA : 0, %t22, DMA : 1)
        AIE.flow(%t60, DMA : 0, %t31, DMA : 1)
    }
}

