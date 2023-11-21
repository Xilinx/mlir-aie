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

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(0, 1)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(0, 2)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(0, 3)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(0, 4)
// CHECK:           %[[VAL_4:.*]] = AIE.tile(1, 1)
// CHECK:           %[[VAL_5:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_6:.*]] = AIE.tile(1, 3)
// CHECK:           %[[VAL_7:.*]] = AIE.tile(1, 4)
// CHECK:           %[[VAL_8:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_9:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_10:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_11:.*]] = AIE.tile(2, 3)
// CHECK:           %[[VAL_12:.*]] = AIE.tile(2, 4)
// CHECK:           %[[VAL_13:.*]] = AIE.tile(3, 0)
// CHECK:           %[[VAL_14:.*]] = AIE.tile(3, 1)
// CHECK:           %[[VAL_15:.*]] = AIE.tile(3, 2)
// CHECK:           %[[VAL_16:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_17:.*]] = AIE.tile(3, 4)
// CHECK:           %[[VAL_18:.*]] = AIE.tile(1, 0)
// CHECK:           %[[VAL_19:.*]] = AIE.switchbox(%[[VAL_18]]) {
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = AIE.switchbox(%[[VAL_8]]) {
// CHECK:             AIE.connect<South : 3, West : 0>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:             AIE.connect<South : 7, North : 0>
// CHECK:             AIE.connect<West : 0, South : 3>
// CHECK:             AIE.connect<East : 0, North : 1>
// CHECK:             AIE.connect<North : 1, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = AIE.shimmux(%[[VAL_8]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = AIE.switchbox(%[[VAL_4]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<Core : 0, West : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = AIE.switchbox(%[[VAL_0]]) {
// CHECK:             AIE.connect<East : 0, Core : 0>
// CHECK:             AIE.connect<Core : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = AIE.switchbox(%[[VAL_1]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<East : 0, Core : 0>
// CHECK:             AIE.connect<DMA : 0, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = AIE.switchbox(%[[VAL_5]]) {
// CHECK:             AIE.connect<West : 0, Core : 0>
// CHECK:             AIE.connect<Core : 0, West : 0>
// CHECK:             AIE.connect<West : 1, East : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = AIE.switchbox(%[[VAL_9]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, DMA : 0>
// CHECK:             AIE.connect<Core : 0, North : 1>
// CHECK:             AIE.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = AIE.switchbox(%[[VAL_10]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, East : 0>
// CHECK:             AIE.connect<East : 0, Core : 0>
// CHECK:             AIE.connect<Core : 0, North : 1>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = AIE.switchbox(%[[VAL_11]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, East : 0>
// CHECK:             AIE.connect<North : 0, Core : 0>
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<East : 0, Core : 1>
// CHECK:             AIE.connect<Core : 1, North : 1>
// CHECK:             AIE.connect<North : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = AIE.switchbox(%[[VAL_7]]) {
// CHECK:             AIE.connect<East : 0, DMA : 0>
// CHECK:             AIE.connect<Core : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = AIE.switchbox(%[[VAL_12]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<East : 0, Core : 0>
// CHECK:             AIE.connect<Core : 0, South : 0>
// CHECK:             AIE.connect<South : 1, East : 0>
// CHECK:             AIE.connect<East : 1, Core : 1>
// CHECK:             AIE.connect<Core : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = AIE.switchbox(%[[VAL_3]]) {
// CHECK:             AIE.connect<East : 0, Core : 0>
// CHECK:             AIE.connect<Core : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = AIE.switchbox(%[[VAL_2]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = AIE.switchbox(%[[VAL_6]]) {
// CHECK:             AIE.connect<West : 0, Core : 0>
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = AIE.switchbox(%[[VAL_13]]) {
// CHECK:             AIE.connect<South : 3, West : 0>
// CHECK:             AIE.connect<West : 0, South : 2>
// CHECK:             AIE.connect<South : 7, North : 0>
// CHECK:             AIE.connect<North : 0, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = AIE.shimmux(%[[VAL_13]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = AIE.switchbox(%[[VAL_15]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<South : 0, North : 1>
// CHECK:             AIE.connect<North : 1, Core : 1>
// CHECK:             AIE.connect<DMA : 1, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = AIE.switchbox(%[[VAL_16]]) {
// CHECK:             AIE.connect<South : 0, Core : 0>
// CHECK:             AIE.connect<Core : 0, South : 0>
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<South : 1, West : 0>
// CHECK:             AIE.connect<West : 1, Core : 1>
// CHECK:             AIE.connect<Core : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_38:.*]] = AIE.switchbox(%[[VAL_17]]) {
// CHECK:             AIE.connect<South : 0, Core : 0>
// CHECK:             AIE.connect<Core : 0, West : 0>
// CHECK:             AIE.connect<West : 0, Core : 1>
// CHECK:             AIE.connect<Core : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_39:.*]] = AIE.switchbox(%[[VAL_14]]) {
// CHECK:             AIE.connect<South : 0, DMA : 1>
// CHECK:             AIE.connect<Core : 1, North : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_0]] : Core, %[[VAL_23]] : Core)
// CHECK:           AIE.wire(%[[VAL_0]] : DMA, %[[VAL_23]] : DMA)
// CHECK:           AIE.wire(%[[VAL_1]] : Core, %[[VAL_24]] : Core)
// CHECK:           AIE.wire(%[[VAL_1]] : DMA, %[[VAL_24]] : DMA)
// CHECK:           AIE.wire(%[[VAL_23]] : North, %[[VAL_24]] : South)
// CHECK:           AIE.wire(%[[VAL_2]] : Core, %[[VAL_32]] : Core)
// CHECK:           AIE.wire(%[[VAL_2]] : DMA, %[[VAL_32]] : DMA)
// CHECK:           AIE.wire(%[[VAL_24]] : North, %[[VAL_32]] : South)
// CHECK:           AIE.wire(%[[VAL_3]] : Core, %[[VAL_31]] : Core)
// CHECK:           AIE.wire(%[[VAL_3]] : DMA, %[[VAL_31]] : DMA)
// CHECK:           AIE.wire(%[[VAL_32]] : North, %[[VAL_31]] : South)
// CHECK:           AIE.wire(%[[VAL_23]] : East, %[[VAL_22]] : West)
// CHECK:           AIE.wire(%[[VAL_4]] : Core, %[[VAL_22]] : Core)
// CHECK:           AIE.wire(%[[VAL_4]] : DMA, %[[VAL_22]] : DMA)
// CHECK:           AIE.wire(%[[VAL_19]] : North, %[[VAL_22]] : South)
// CHECK:           AIE.wire(%[[VAL_24]] : East, %[[VAL_25]] : West)
// CHECK:           AIE.wire(%[[VAL_5]] : Core, %[[VAL_25]] : Core)
// CHECK:           AIE.wire(%[[VAL_5]] : DMA, %[[VAL_25]] : DMA)
// CHECK:           AIE.wire(%[[VAL_22]] : North, %[[VAL_25]] : South)
// CHECK:           AIE.wire(%[[VAL_32]] : East, %[[VAL_33]] : West)
// CHECK:           AIE.wire(%[[VAL_6]] : Core, %[[VAL_33]] : Core)
// CHECK:           AIE.wire(%[[VAL_6]] : DMA, %[[VAL_33]] : DMA)
// CHECK:           AIE.wire(%[[VAL_25]] : North, %[[VAL_33]] : South)
// CHECK:           AIE.wire(%[[VAL_31]] : East, %[[VAL_29]] : West)
// CHECK:           AIE.wire(%[[VAL_7]] : Core, %[[VAL_29]] : Core)
// CHECK:           AIE.wire(%[[VAL_7]] : DMA, %[[VAL_29]] : DMA)
// CHECK:           AIE.wire(%[[VAL_33]] : North, %[[VAL_29]] : South)
// CHECK:           AIE.wire(%[[VAL_19]] : East, %[[VAL_20]] : West)
// CHECK:           AIE.wire(%[[VAL_21]] : North, %[[VAL_20]] : South)
// CHECK:           AIE.wire(%[[VAL_8]] : DMA, %[[VAL_21]] : DMA)
// CHECK:           AIE.wire(%[[VAL_22]] : East, %[[VAL_26]] : West)
// CHECK:           AIE.wire(%[[VAL_9]] : Core, %[[VAL_26]] : Core)
// CHECK:           AIE.wire(%[[VAL_9]] : DMA, %[[VAL_26]] : DMA)
// CHECK:           AIE.wire(%[[VAL_20]] : North, %[[VAL_26]] : South)
// CHECK:           AIE.wire(%[[VAL_25]] : East, %[[VAL_27]] : West)
// CHECK:           AIE.wire(%[[VAL_10]] : Core, %[[VAL_27]] : Core)
// CHECK:           AIE.wire(%[[VAL_10]] : DMA, %[[VAL_27]] : DMA)
// CHECK:           AIE.wire(%[[VAL_26]] : North, %[[VAL_27]] : South)
// CHECK:           AIE.wire(%[[VAL_33]] : East, %[[VAL_28]] : West)
// CHECK:           AIE.wire(%[[VAL_11]] : Core, %[[VAL_28]] : Core)
// CHECK:           AIE.wire(%[[VAL_11]] : DMA, %[[VAL_28]] : DMA)
// CHECK:           AIE.wire(%[[VAL_27]] : North, %[[VAL_28]] : South)
// CHECK:           AIE.wire(%[[VAL_29]] : East, %[[VAL_30]] : West)
// CHECK:           AIE.wire(%[[VAL_12]] : Core, %[[VAL_30]] : Core)
// CHECK:           AIE.wire(%[[VAL_12]] : DMA, %[[VAL_30]] : DMA)
// CHECK:           AIE.wire(%[[VAL_28]] : North, %[[VAL_30]] : South)
// CHECK:           AIE.wire(%[[VAL_20]] : East, %[[VAL_34]] : West)
// CHECK:           AIE.wire(%[[VAL_35]] : North, %[[VAL_34]] : South)
// CHECK:           AIE.wire(%[[VAL_13]] : DMA, %[[VAL_35]] : DMA)
// CHECK:           AIE.wire(%[[VAL_26]] : East, %[[VAL_39]] : West)
// CHECK:           AIE.wire(%[[VAL_14]] : Core, %[[VAL_39]] : Core)
// CHECK:           AIE.wire(%[[VAL_14]] : DMA, %[[VAL_39]] : DMA)
// CHECK:           AIE.wire(%[[VAL_34]] : North, %[[VAL_39]] : South)
// CHECK:           AIE.wire(%[[VAL_27]] : East, %[[VAL_36]] : West)
// CHECK:           AIE.wire(%[[VAL_15]] : Core, %[[VAL_36]] : Core)
// CHECK:           AIE.wire(%[[VAL_15]] : DMA, %[[VAL_36]] : DMA)
// CHECK:           AIE.wire(%[[VAL_39]] : North, %[[VAL_36]] : South)
// CHECK:           AIE.wire(%[[VAL_28]] : East, %[[VAL_37]] : West)
// CHECK:           AIE.wire(%[[VAL_16]] : Core, %[[VAL_37]] : Core)
// CHECK:           AIE.wire(%[[VAL_16]] : DMA, %[[VAL_37]] : DMA)
// CHECK:           AIE.wire(%[[VAL_36]] : North, %[[VAL_37]] : South)
// CHECK:           AIE.wire(%[[VAL_30]] : East, %[[VAL_38]] : West)
// CHECK:           AIE.wire(%[[VAL_17]] : Core, %[[VAL_38]] : Core)
// CHECK:           AIE.wire(%[[VAL_17]] : DMA, %[[VAL_38]] : DMA)
// CHECK:           AIE.wire(%[[VAL_37]] : North, %[[VAL_38]] : South)
// CHECK:         }



module {
    AIE.device(xcvc1902) {
        %t01 = AIE.tile(0, 1)
        %t02 = AIE.tile(0, 2)
        %t03 = AIE.tile(0, 3)
        %t04 = AIE.tile(0, 4)
        %t11 = AIE.tile(1, 1)
        %t12 = AIE.tile(1, 2)
        %t13 = AIE.tile(1, 3)
        %t14 = AIE.tile(1, 4)
        %t20 = AIE.tile(2, 0)
        %t21 = AIE.tile(2, 1)
        %t22 = AIE.tile(2, 2)
        %t23 = AIE.tile(2, 3)
        %t24 = AIE.tile(2, 4)
        %t30 = AIE.tile(3, 0)
        %t31 = AIE.tile(3, 1)
        %t32 = AIE.tile(3, 2)
        %t33 = AIE.tile(3, 3)
        %t34 = AIE.tile(3, 4)

        //TASK 1
        AIE.flow(%t20, DMA : 0, %t11, DMA : 0)
        AIE.flow(%t11, Core : 0, %t01, Core : 0)
        AIE.flow(%t01, Core : 0, %t12, Core : 0)
        AIE.flow(%t12, Core : 0, %t02, Core : 0)
        AIE.flow(%t02, DMA : 0, %t20, DMA : 0)

        //TASK 2
        AIE.flow(%t20, DMA : 1, %t14, DMA : 0)
        AIE.flow(%t14, Core : 0, %t04, Core : 0)
        AIE.flow(%t04, Core : 0, %t13, Core : 0)
        AIE.flow(%t13, DMA : 0, %t20, DMA : 1)

        //TASK 3
        AIE.flow(%t30, DMA : 0, %t21, DMA : 0)
        AIE.flow(%t21, Core : 0, %t33, Core : 0)
        AIE.flow(%t33, Core : 0, %t22, Core : 0)
        AIE.flow(%t22, Core : 0, %t34, Core : 0)
        AIE.flow(%t34, Core : 0, %t24, Core : 0)
        AIE.flow(%t24, Core : 0, %t23, Core : 0)
        AIE.flow(%t23, DMA : 0, %t30, DMA : 0)

        //TASK 4
        AIE.flow(%t30, DMA : 1, %t31, DMA : 1)
        AIE.flow(%t31, Core : 1, %t23, Core : 1)
        AIE.flow(%t23, Core : 1, %t34, Core : 1)
        AIE.flow(%t34, Core : 1, %t24, Core : 1)
        AIE.flow(%t24, Core : 1, %t33, Core : 1)
        AIE.flow(%t33, Core : 1, %t32, Core : 1)
        AIE.flow(%t32, DMA : 1, %t30, DMA : 1)
    }
}
