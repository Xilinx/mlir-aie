//===- many_flows2.mlir ----------------------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_0:.*]] = AIE.tile(0, 2)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(0, 3)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(1, 1)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(1, 3)
// CHECK:           %[[VAL_4:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_5:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_6:.*]] = AIE.tile(3, 0)
// CHECK:           %[[VAL_7:.*]] = AIE.tile(3, 1)
// CHECK:           %[[VAL_8:.*]] = AIE.tile(6, 0)
// CHECK:           %[[VAL_9:.*]] = AIE.tile(7, 0)
// CHECK:           %[[VAL_10:.*]] = AIE.tile(7, 3)
// CHECK:           %[[VAL_11:.*]] = AIE.switchbox(%[[VAL_4]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<West : 0, East : 1>
// CHECK:             AIE.connect<North : 1, South : 2>
// CHECK:             AIE.connect<East : 0, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = AIE.switchbox(%[[VAL_6]]) {
// CHECK:             AIE.connect<West : 0, South : 2>
// CHECK:             AIE.connect<West : 1, East : 0>
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = AIE.shimmux(%[[VAL_6]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_15:.*]] = AIE.switchbox(%[[VAL_14]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<North : 1, East : 0>
// CHECK:             AIE.connect<North : 2, South : 1>
// CHECK:             AIE.connect<North : 3, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = AIE.switchbox(%[[VAL_0]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<North : 1, East : 1>
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<North : 2, Core : 0>
// CHECK:             AIE.connect<North : 3, Core : 1>
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_18:.*]] = AIE.switchbox(%[[VAL_17]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<North : 0, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_19:.*]] = AIE.switchbox(%[[VAL_5]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<West : 1, South : 1>
// CHECK:             AIE.connect<DMA : 0, South : 2>
// CHECK:             AIE.connect<Core : 0, North : 0>
// CHECK:             AIE.connect<North : 0, Core : 1>
// CHECK:             AIE.connect<West : 2, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = AIE.switchbox(%[[VAL_1]]) {
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<DMA : 1, South : 1>
// CHECK:             AIE.connect<Core : 1, South : 2>
// CHECK:             AIE.connect<Core : 0, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = AIE.switchbox(%[[VAL_8]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<West : 0, South : 2>
// CHECK:             AIE.connect<East : 0, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = AIE.switchbox(%[[VAL_9]]) {
// CHECK:             AIE.connect<West : 0, South : 3>
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<North : 1, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = AIE.shimmux(%[[VAL_9]]) {
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = AIE.switchbox(%[[VAL_7]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 0, Core : 0>
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<DMA : 1, South : 1>
// CHECK:             AIE.connect<West : 1, Core : 1>
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = AIE.tile(4, 1)
// CHECK:           %[[VAL_26:.*]] = AIE.switchbox(%[[VAL_25]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = AIE.tile(5, 1)
// CHECK:           %[[VAL_28:.*]] = AIE.switchbox(%[[VAL_27]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = AIE.tile(6, 1)
// CHECK:           %[[VAL_30:.*]] = AIE.switchbox(%[[VAL_29]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = AIE.tile(1, 0)
// CHECK:           %[[VAL_32:.*]] = AIE.switchbox(%[[VAL_31]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = AIE.tile(4, 0)
// CHECK:           %[[VAL_34:.*]] = AIE.switchbox(%[[VAL_33]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = AIE.tile(5, 0)
// CHECK:           %[[VAL_36:.*]] = AIE.switchbox(%[[VAL_35]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = AIE.shimmux(%[[VAL_8]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_38:.*]] = AIE.tile(0, 1)
// CHECK:           %[[VAL_39:.*]] = AIE.switchbox(%[[VAL_38]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_40:.*]] = AIE.switchbox(%[[VAL_2]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = AIE.shimmux(%[[VAL_4]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_42:.*]] = AIE.switchbox(%[[VAL_3]]) {
// CHECK:             AIE.connect<East : 0, Core : 0>
// CHECK:             AIE.connect<Core : 1, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_43:.*]] = AIE.tile(2, 3)
// CHECK:           %[[VAL_44:.*]] = AIE.switchbox(%[[VAL_43]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_45:.*]] = AIE.tile(5, 2)
// CHECK:           %[[VAL_46:.*]] = AIE.switchbox(%[[VAL_45]]) {
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_47:.*]] = AIE.tile(6, 2)
// CHECK:           %[[VAL_48:.*]] = AIE.switchbox(%[[VAL_47]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_49:.*]] = AIE.tile(7, 2)
// CHECK:           %[[VAL_50:.*]] = AIE.switchbox(%[[VAL_49]]) {
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<North : 1, South : 0>
// CHECK:             AIE.connect<North : 2, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_51:.*]] = AIE.switchbox(%[[VAL_10]]) {
// CHECK:             AIE.connect<Core : 0, South : 0>
// CHECK:             AIE.connect<Core : 1, West : 0>
// CHECK:             AIE.connect<DMA : 0, South : 1>
// CHECK:             AIE.connect<DMA : 1, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_52:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_53:.*]] = AIE.switchbox(%[[VAL_52]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = AIE.tile(4, 3)
// CHECK:           %[[VAL_55:.*]] = AIE.switchbox(%[[VAL_54]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_56:.*]] = AIE.tile(5, 3)
// CHECK:           %[[VAL_57:.*]] = AIE.switchbox(%[[VAL_56]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_58:.*]] = AIE.tile(6, 3)
// CHECK:           %[[VAL_59:.*]] = AIE.switchbox(%[[VAL_58]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_60:.*]] = AIE.tile(7, 1)
// CHECK:           %[[VAL_61:.*]] = AIE.switchbox(%[[VAL_60]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_38]] : Core, %[[VAL_39]] : Core)
// CHECK:           AIE.wire(%[[VAL_38]] : DMA, %[[VAL_39]] : DMA)
// CHECK:           AIE.wire(%[[VAL_0]] : Core, %[[VAL_16]] : Core)
// CHECK:           AIE.wire(%[[VAL_0]] : DMA, %[[VAL_16]] : DMA)
// CHECK:           AIE.wire(%[[VAL_39]] : North, %[[VAL_16]] : South)
// CHECK:           AIE.wire(%[[VAL_1]] : Core, %[[VAL_20]] : Core)
// CHECK:           AIE.wire(%[[VAL_1]] : DMA, %[[VAL_20]] : DMA)
// CHECK:           AIE.wire(%[[VAL_16]] : North, %[[VAL_20]] : South)
// CHECK:           AIE.wire(%[[VAL_39]] : East, %[[VAL_40]] : West)
// CHECK:           AIE.wire(%[[VAL_2]] : Core, %[[VAL_40]] : Core)
// CHECK:           AIE.wire(%[[VAL_2]] : DMA, %[[VAL_40]] : DMA)
// CHECK:           AIE.wire(%[[VAL_32]] : North, %[[VAL_40]] : South)
// CHECK:           AIE.wire(%[[VAL_16]] : East, %[[VAL_18]] : West)
// CHECK:           AIE.wire(%[[VAL_17]] : Core, %[[VAL_18]] : Core)
// CHECK:           AIE.wire(%[[VAL_17]] : DMA, %[[VAL_18]] : DMA)
// CHECK:           AIE.wire(%[[VAL_40]] : North, %[[VAL_18]] : South)
// CHECK:           AIE.wire(%[[VAL_20]] : East, %[[VAL_42]] : West)
// CHECK:           AIE.wire(%[[VAL_3]] : Core, %[[VAL_42]] : Core)
// CHECK:           AIE.wire(%[[VAL_3]] : DMA, %[[VAL_42]] : DMA)
// CHECK:           AIE.wire(%[[VAL_18]] : North, %[[VAL_42]] : South)
// CHECK:           AIE.wire(%[[VAL_32]] : East, %[[VAL_11]] : West)
// CHECK:           AIE.wire(%[[VAL_41]] : North, %[[VAL_11]] : South)
// CHECK:           AIE.wire(%[[VAL_4]] : DMA, %[[VAL_41]] : DMA)
// CHECK:           AIE.wire(%[[VAL_40]] : East, %[[VAL_15]] : West)
// CHECK:           AIE.wire(%[[VAL_14]] : Core, %[[VAL_15]] : Core)
// CHECK:           AIE.wire(%[[VAL_14]] : DMA, %[[VAL_15]] : DMA)
// CHECK:           AIE.wire(%[[VAL_11]] : North, %[[VAL_15]] : South)
// CHECK:           AIE.wire(%[[VAL_18]] : East, %[[VAL_19]] : West)
// CHECK:           AIE.wire(%[[VAL_5]] : Core, %[[VAL_19]] : Core)
// CHECK:           AIE.wire(%[[VAL_5]] : DMA, %[[VAL_19]] : DMA)
// CHECK:           AIE.wire(%[[VAL_15]] : North, %[[VAL_19]] : South)
// CHECK:           AIE.wire(%[[VAL_42]] : East, %[[VAL_44]] : West)
// CHECK:           AIE.wire(%[[VAL_43]] : Core, %[[VAL_44]] : Core)
// CHECK:           AIE.wire(%[[VAL_43]] : DMA, %[[VAL_44]] : DMA)
// CHECK:           AIE.wire(%[[VAL_19]] : North, %[[VAL_44]] : South)
// CHECK:           AIE.wire(%[[VAL_11]] : East, %[[VAL_12]] : West)
// CHECK:           AIE.wire(%[[VAL_13]] : North, %[[VAL_12]] : South)
// CHECK:           AIE.wire(%[[VAL_6]] : DMA, %[[VAL_13]] : DMA)
// CHECK:           AIE.wire(%[[VAL_15]] : East, %[[VAL_24]] : West)
// CHECK:           AIE.wire(%[[VAL_7]] : Core, %[[VAL_24]] : Core)
// CHECK:           AIE.wire(%[[VAL_7]] : DMA, %[[VAL_24]] : DMA)
// CHECK:           AIE.wire(%[[VAL_12]] : North, %[[VAL_24]] : South)
// CHECK:           AIE.wire(%[[VAL_44]] : East, %[[VAL_53]] : West)
// CHECK:           AIE.wire(%[[VAL_52]] : Core, %[[VAL_53]] : Core)
// CHECK:           AIE.wire(%[[VAL_52]] : DMA, %[[VAL_53]] : DMA)
// CHECK:           AIE.wire(%[[VAL_12]] : East, %[[VAL_34]] : West)
// CHECK:           AIE.wire(%[[VAL_24]] : East, %[[VAL_26]] : West)
// CHECK:           AIE.wire(%[[VAL_25]] : Core, %[[VAL_26]] : Core)
// CHECK:           AIE.wire(%[[VAL_25]] : DMA, %[[VAL_26]] : DMA)
// CHECK:           AIE.wire(%[[VAL_34]] : North, %[[VAL_26]] : South)
// CHECK:           AIE.wire(%[[VAL_53]] : East, %[[VAL_55]] : West)
// CHECK:           AIE.wire(%[[VAL_54]] : Core, %[[VAL_55]] : Core)
// CHECK:           AIE.wire(%[[VAL_54]] : DMA, %[[VAL_55]] : DMA)
// CHECK:           AIE.wire(%[[VAL_34]] : East, %[[VAL_36]] : West)
// CHECK:           AIE.wire(%[[VAL_26]] : East, %[[VAL_28]] : West)
// CHECK:           AIE.wire(%[[VAL_27]] : Core, %[[VAL_28]] : Core)
// CHECK:           AIE.wire(%[[VAL_27]] : DMA, %[[VAL_28]] : DMA)
// CHECK:           AIE.wire(%[[VAL_36]] : North, %[[VAL_28]] : South)
// CHECK:           AIE.wire(%[[VAL_45]] : Core, %[[VAL_46]] : Core)
// CHECK:           AIE.wire(%[[VAL_45]] : DMA, %[[VAL_46]] : DMA)
// CHECK:           AIE.wire(%[[VAL_28]] : North, %[[VAL_46]] : South)
// CHECK:           AIE.wire(%[[VAL_55]] : East, %[[VAL_57]] : West)
// CHECK:           AIE.wire(%[[VAL_56]] : Core, %[[VAL_57]] : Core)
// CHECK:           AIE.wire(%[[VAL_56]] : DMA, %[[VAL_57]] : DMA)
// CHECK:           AIE.wire(%[[VAL_46]] : North, %[[VAL_57]] : South)
// CHECK:           AIE.wire(%[[VAL_36]] : East, %[[VAL_21]] : West)
// CHECK:           AIE.wire(%[[VAL_37]] : North, %[[VAL_21]] : South)
// CHECK:           AIE.wire(%[[VAL_8]] : DMA, %[[VAL_37]] : DMA)
// CHECK:           AIE.wire(%[[VAL_28]] : East, %[[VAL_30]] : West)
// CHECK:           AIE.wire(%[[VAL_29]] : Core, %[[VAL_30]] : Core)
// CHECK:           AIE.wire(%[[VAL_29]] : DMA, %[[VAL_30]] : DMA)
// CHECK:           AIE.wire(%[[VAL_21]] : North, %[[VAL_30]] : South)
// CHECK:           AIE.wire(%[[VAL_46]] : East, %[[VAL_48]] : West)
// CHECK:           AIE.wire(%[[VAL_47]] : Core, %[[VAL_48]] : Core)
// CHECK:           AIE.wire(%[[VAL_47]] : DMA, %[[VAL_48]] : DMA)
// CHECK:           AIE.wire(%[[VAL_30]] : North, %[[VAL_48]] : South)
// CHECK:           AIE.wire(%[[VAL_57]] : East, %[[VAL_59]] : West)
// CHECK:           AIE.wire(%[[VAL_58]] : Core, %[[VAL_59]] : Core)
// CHECK:           AIE.wire(%[[VAL_58]] : DMA, %[[VAL_59]] : DMA)
// CHECK:           AIE.wire(%[[VAL_48]] : North, %[[VAL_59]] : South)
// CHECK:           AIE.wire(%[[VAL_21]] : East, %[[VAL_22]] : West)
// CHECK:           AIE.wire(%[[VAL_23]] : North, %[[VAL_22]] : South)
// CHECK:           AIE.wire(%[[VAL_9]] : DMA, %[[VAL_23]] : DMA)
// CHECK:           AIE.wire(%[[VAL_30]] : East, %[[VAL_61]] : West)
// CHECK:           AIE.wire(%[[VAL_60]] : Core, %[[VAL_61]] : Core)
// CHECK:           AIE.wire(%[[VAL_60]] : DMA, %[[VAL_61]] : DMA)
// CHECK:           AIE.wire(%[[VAL_22]] : North, %[[VAL_61]] : South)
// CHECK:           AIE.wire(%[[VAL_48]] : East, %[[VAL_50]] : West)
// CHECK:           AIE.wire(%[[VAL_49]] : Core, %[[VAL_50]] : Core)
// CHECK:           AIE.wire(%[[VAL_49]] : DMA, %[[VAL_50]] : DMA)
// CHECK:           AIE.wire(%[[VAL_61]] : North, %[[VAL_50]] : South)
// CHECK:           AIE.wire(%[[VAL_59]] : East, %[[VAL_51]] : West)
// CHECK:           AIE.wire(%[[VAL_10]] : Core, %[[VAL_51]] : Core)
// CHECK:           AIE.wire(%[[VAL_10]] : DMA, %[[VAL_51]] : DMA)
// CHECK:           AIE.wire(%[[VAL_50]] : North, %[[VAL_51]] : South)
// CHECK:         }

//

module {
    AIE.device(xcvc1902) {
        %t02 = AIE.tile(0, 2)
        %t03 = AIE.tile(0, 3)
        %t11 = AIE.tile(1, 1)
        %t13 = AIE.tile(1, 3)
        %t20 = AIE.tile(2, 0)
        %t22 = AIE.tile(2, 2)
        %t30 = AIE.tile(3, 0)
        %t31 = AIE.tile(3, 1)
        %t60 = AIE.tile(6, 0)
        %t70 = AIE.tile(7, 0)
        %t73 = AIE.tile(7, 3)

        AIE.flow(%t03, DMA : 0, %t30, DMA : 0)
        AIE.flow(%t03, DMA : 1, %t70, DMA : 1)
        AIE.flow(%t02, DMA : 0, %t60, DMA : 0)
        AIE.flow(%t22, DMA : 0, %t20, DMA : 0)

        AIE.flow(%t22, Core : 0, %t13, Core : 0)
        AIE.flow(%t03, Core : 1, %t02, Core : 0)
        AIE.flow(%t73, Core : 0, %t31, Core : 0)
        AIE.flow(%t73, Core : 1, %t22, Core : 1)

        AIE.flow(%t73, DMA : 0, %t60, DMA : 1)
        AIE.flow(%t73, DMA : 1, %t70, DMA : 0)
        AIE.flow(%t31, DMA : 0, %t20, DMA : 1)
        AIE.flow(%t31, DMA : 1, %t30, DMA : 1)

        AIE.flow(%t03, Core : 0, %t02, Core : 1)
        AIE.flow(%t13, Core : 1, %t31, Core : 1)
    }
}
