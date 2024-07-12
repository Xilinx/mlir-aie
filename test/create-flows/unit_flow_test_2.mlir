//===- flow_test_2.mlir ----------------------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_2:.*]] = aie.tile(0, 3)
// CHECK:           %[[VAL_3:.*]] = aie.tile(0, 4)
// CHECK:           %[[VAL_4:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_5:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_6:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_7:.*]] = aie.tile(1, 4)
// CHECK:           %[[VAL_8:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_9:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_10:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_11:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_12:.*]] = aie.tile(2, 4)
// CHECK:           %[[VAL_13:.*]] = aie.tile(3, 0)
// CHECK:           %[[VAL_14:.*]] = aie.tile(3, 1)
// CHECK:           %[[VAL_15:.*]] = aie.tile(3, 2)
// CHECK:           %[[VAL_16:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_17:.*]] = aie.tile(3, 4)
// CHECK:           %[[VAL_18:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_19:.*]] = aie.switchbox(%[[VAL_18]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = aie.switchbox(%[[VAL_8]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<South : 7, North : 0>
// CHECK:             aie.connect<West : 0, South : 3>
// CHECK:             aie.connect<East : 0, North : 1>
// CHECK:             aie.connect<North : 1, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = aie.shim_mux(%[[VAL_8]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = aie.switchbox(%[[VAL_4]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = aie.switchbox(%[[VAL_1]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = aie.switchbox(%[[VAL_5]]) {
// CHECK:             aie.connect<West : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:             aie.connect<West : 1, East : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = aie.switchbox(%[[VAL_9]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, DMA : 0>
// CHECK:             aie.connect<Core : 0, North : 1>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = aie.switchbox(%[[VAL_10]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, East : 0>
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, North : 1>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = aie.switchbox(%[[VAL_11]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, East : 0>
// CHECK:             aie.connect<North : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<East : 0, Core : 1>
// CHECK:             aie.connect<Core : 1, North : 1>
// CHECK:             aie.connect<North : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = aie.switchbox(%[[VAL_7]]) {
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = aie.switchbox(%[[VAL_12]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:             aie.connect<South : 1, East : 0>
// CHECK:             aie.connect<East : 1, Core : 1>
// CHECK:             aie.connect<Core : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = aie.switchbox(%[[VAL_3]]) {
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = aie.switchbox(%[[VAL_6]]) {
// CHECK:             aie.connect<West : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = aie.switchbox(%[[VAL_13]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<West : 0, South : 2>
// CHECK:             aie.connect<South : 7, North : 0>
// CHECK:             aie.connect<North : 0, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = aie.shim_mux(%[[VAL_13]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = aie.switchbox(%[[VAL_15]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<South : 0, North : 1>
// CHECK:             aie.connect<North : 1, Core : 1>
// CHECK:             aie.connect<DMA : 1, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = aie.switchbox(%[[VAL_16]]) {
// CHECK:             aie.connect<South : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<South : 1, West : 0>
// CHECK:             aie.connect<West : 1, Core : 1>
// CHECK:             aie.connect<Core : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_38:.*]] = aie.switchbox(%[[VAL_17]]) {
// CHECK:             aie.connect<South : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:             aie.connect<West : 0, Core : 1>
// CHECK:             aie.connect<Core : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_39:.*]] = aie.switchbox(%[[VAL_14]]) {
// CHECK:             aie.connect<South : 0, DMA : 1>
// CHECK:             aie.connect<Core : 1, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[VAL_0]] : Core, %[[VAL_40:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_0]] : DMA, %[[VAL_40]] : DMA)
// CHECK:           aie.wire(%[[VAL_1]] : Core, %[[VAL_41:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_1]] : DMA, %[[VAL_41]] : DMA)
// CHECK:           aie.wire(%[[VAL_40]] : North, %[[VAL_41]] : South)
// CHECK:           aie.wire(%[[VAL_2]] : Core, %[[VAL_42:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_2]] : DMA, %[[VAL_42]] : DMA)
// CHECK:           aie.wire(%[[VAL_41]] : North, %[[VAL_42]] : South)
// CHECK:           aie.wire(%[[VAL_3]] : Core, %[[VAL_43:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_3]] : DMA, %[[VAL_43]] : DMA)
// CHECK:           aie.wire(%[[VAL_42]] : North, %[[VAL_43]] : South)
// CHECK:           aie.wire(%[[VAL_40]] : East, %[[VAL_44:.*]] : West)
// CHECK:           aie.wire(%[[VAL_4]] : Core, %[[VAL_44]] : Core)
// CHECK:           aie.wire(%[[VAL_4]] : DMA, %[[VAL_44]] : DMA)
// CHECK:           aie.wire(%[[VAL_45:.*]] : North, %[[VAL_44]] : South)
// CHECK:           aie.wire(%[[VAL_41]] : East, %[[VAL_46:.*]] : West)
// CHECK:           aie.wire(%[[VAL_5]] : Core, %[[VAL_46]] : Core)
// CHECK:           aie.wire(%[[VAL_5]] : DMA, %[[VAL_46]] : DMA)
// CHECK:           aie.wire(%[[VAL_44]] : North, %[[VAL_46]] : South)
// CHECK:           aie.wire(%[[VAL_42]] : East, %[[VAL_47:.*]] : West)
// CHECK:           aie.wire(%[[VAL_6]] : Core, %[[VAL_47]] : Core)
// CHECK:           aie.wire(%[[VAL_6]] : DMA, %[[VAL_47]] : DMA)
// CHECK:           aie.wire(%[[VAL_46]] : North, %[[VAL_47]] : South)
// CHECK:           aie.wire(%[[VAL_43]] : East, %[[VAL_48:.*]] : West)
// CHECK:           aie.wire(%[[VAL_7]] : Core, %[[VAL_48]] : Core)
// CHECK:           aie.wire(%[[VAL_7]] : DMA, %[[VAL_48]] : DMA)
// CHECK:           aie.wire(%[[VAL_47]] : North, %[[VAL_48]] : South)
// CHECK:           aie.wire(%[[VAL_45]] : East, %[[VAL_49:.*]] : West)
// CHECK:           aie.wire(%[[VAL_50:.*]] : North, %[[VAL_49]] : South)
// CHECK:           aie.wire(%[[VAL_8]] : DMA, %[[VAL_50]] : DMA)
// CHECK:           aie.wire(%[[VAL_44]] : East, %[[VAL_51:.*]] : West)
// CHECK:           aie.wire(%[[VAL_9]] : Core, %[[VAL_51]] : Core)
// CHECK:           aie.wire(%[[VAL_9]] : DMA, %[[VAL_51]] : DMA)
// CHECK:           aie.wire(%[[VAL_49]] : North, %[[VAL_51]] : South)
// CHECK:           aie.wire(%[[VAL_46]] : East, %[[VAL_52:.*]] : West)
// CHECK:           aie.wire(%[[VAL_10]] : Core, %[[VAL_52]] : Core)
// CHECK:           aie.wire(%[[VAL_10]] : DMA, %[[VAL_52]] : DMA)
// CHECK:           aie.wire(%[[VAL_51]] : North, %[[VAL_52]] : South)
// CHECK:           aie.wire(%[[VAL_47]] : East, %[[VAL_53:.*]] : West)
// CHECK:           aie.wire(%[[VAL_11]] : Core, %[[VAL_53]] : Core)
// CHECK:           aie.wire(%[[VAL_11]] : DMA, %[[VAL_53]] : DMA)
// CHECK:           aie.wire(%[[VAL_52]] : North, %[[VAL_53]] : South)
// CHECK:           aie.wire(%[[VAL_48]] : East, %[[VAL_54:.*]] : West)
// CHECK:           aie.wire(%[[VAL_12]] : Core, %[[VAL_54]] : Core)
// CHECK:           aie.wire(%[[VAL_12]] : DMA, %[[VAL_54]] : DMA)
// CHECK:           aie.wire(%[[VAL_53]] : North, %[[VAL_54]] : South)
// CHECK:           aie.wire(%[[VAL_49]] : East, %[[VAL_55:.*]] : West)
// CHECK:           aie.wire(%[[VAL_56:.*]] : North, %[[VAL_55]] : South)
// CHECK:           aie.wire(%[[VAL_13]] : DMA, %[[VAL_56]] : DMA)
// CHECK:           aie.wire(%[[VAL_51]] : East, %[[VAL_57:.*]] : West)
// CHECK:           aie.wire(%[[VAL_14]] : Core, %[[VAL_57]] : Core)
// CHECK:           aie.wire(%[[VAL_14]] : DMA, %[[VAL_57]] : DMA)
// CHECK:           aie.wire(%[[VAL_55]] : North, %[[VAL_57]] : South)
// CHECK:           aie.wire(%[[VAL_52]] : East, %[[VAL_58:.*]] : West)
// CHECK:           aie.wire(%[[VAL_15]] : Core, %[[VAL_58]] : Core)
// CHECK:           aie.wire(%[[VAL_15]] : DMA, %[[VAL_58]] : DMA)
// CHECK:           aie.wire(%[[VAL_57]] : North, %[[VAL_58]] : South)
// CHECK:           aie.wire(%[[VAL_53]] : East, %[[VAL_59:.*]] : West)
// CHECK:           aie.wire(%[[VAL_16]] : Core, %[[VAL_59]] : Core)
// CHECK:           aie.wire(%[[VAL_16]] : DMA, %[[VAL_59]] : DMA)
// CHECK:           aie.wire(%[[VAL_58]] : North, %[[VAL_59]] : South)
// CHECK:           aie.wire(%[[VAL_54]] : East, %[[VAL_60:.*]] : West)
// CHECK:           aie.wire(%[[VAL_17]] : Core, %[[VAL_60]] : Core)
// CHECK:           aie.wire(%[[VAL_17]] : DMA, %[[VAL_60]] : DMA)
// CHECK:           aie.wire(%[[VAL_59]] : North, %[[VAL_60]] : South)
// CHECK:         }

module {
    aie.device(xcvc1902) {
        %t01 = aie.tile(0, 1)
        %t02 = aie.tile(0, 2)
        %t03 = aie.tile(0, 3)
        %t04 = aie.tile(0, 4)
        %t11 = aie.tile(1, 1)
        %t12 = aie.tile(1, 2)
        %t13 = aie.tile(1, 3)
        %t14 = aie.tile(1, 4)
        %t20 = aie.tile(2, 0)
        %t21 = aie.tile(2, 1)
        %t22 = aie.tile(2, 2)
        %t23 = aie.tile(2, 3)
        %t24 = aie.tile(2, 4)
        %t30 = aie.tile(3, 0)
        %t31 = aie.tile(3, 1)
        %t32 = aie.tile(3, 2)
        %t33 = aie.tile(3, 3)
        %t34 = aie.tile(3, 4)

        //TASK 1
        aie.flow(%t20, DMA : 0, %t11, DMA : 0)
        aie.flow(%t11, Core : 0, %t01, Core : 0)
        aie.flow(%t01, Core : 0, %t12, Core : 0)
        aie.flow(%t12, Core : 0, %t02, Core : 0)
        aie.flow(%t02, DMA : 0, %t20, DMA : 0)

        //TASK 2
        aie.flow(%t20, DMA : 1, %t14, DMA : 0)
        aie.flow(%t14, Core : 0, %t04, Core : 0)
        aie.flow(%t04, Core : 0, %t13, Core : 0)
        aie.flow(%t13, DMA : 0, %t20, DMA : 1)

        //TASK 3
        aie.flow(%t30, DMA : 0, %t21, DMA : 0)
        aie.flow(%t21, Core : 0, %t33, Core : 0)
        aie.flow(%t33, Core : 0, %t22, Core : 0)
        aie.flow(%t22, Core : 0, %t34, Core : 0)
        aie.flow(%t34, Core : 0, %t24, Core : 0)
        aie.flow(%t24, Core : 0, %t23, Core : 0)
        aie.flow(%t23, DMA : 0, %t30, DMA : 0)

        //TASK 4
        aie.flow(%t30, DMA : 1, %t31, DMA : 1)
        aie.flow(%t31, Core : 1, %t23, Core : 1)
        aie.flow(%t23, Core : 1, %t34, Core : 1)
        aie.flow(%t34, Core : 1, %t24, Core : 1)
        aie.flow(%t24, Core : 1, %t33, Core : 1)
        aie.flow(%t33, Core : 1, %t32, Core : 1)
        aie.flow(%t32, DMA : 1, %t30, DMA : 1)
    }
}
