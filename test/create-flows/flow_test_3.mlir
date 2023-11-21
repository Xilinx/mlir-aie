//===- flow_test_3.mlir ----------------------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_14:.*]] = AIE.tile(7, 1)
// CHECK:           %[[VAL_15:.*]] = AIE.tile(7, 2)
// CHECK:           %[[VAL_16:.*]] = AIE.tile(7, 3)
// CHECK:           %[[VAL_17:.*]] = AIE.tile(7, 4)
// CHECK:           %[[VAL_18:.*]] = AIE.tile(8, 1)
// CHECK:           %[[VAL_19:.*]] = AIE.tile(8, 2)
// CHECK:           %[[VAL_20:.*]] = AIE.tile(8, 3)
// CHECK:           %[[VAL_21:.*]] = AIE.tile(8, 4)
// CHECK:           %[[VAL_22:.*]] = AIE.tile(1, 0)
// CHECK:           %[[VAL_23:.*]] = AIE.switchbox(%[[VAL_22]]) {
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = AIE.switchbox(%[[VAL_8]]) {
// CHECK:             AIE.connect<South : 3, West : 0>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:             AIE.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = AIE.shimmux(%[[VAL_8]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = AIE.switchbox(%[[VAL_0]]) {
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:             AIE.connect<East : 1, Core : 0>
// CHECK:             AIE.connect<Core : 0, East : 0>
// CHECK:             AIE.connect<East : 2, Core : 1>
// CHECK:             AIE.connect<Core : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = AIE.switchbox(%[[VAL_4]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<East : 0, Core : 0>
// CHECK:             AIE.connect<Core : 0, North : 0>
// CHECK:             AIE.connect<East : 1, North : 1>
// CHECK:             AIE.connect<North : 0, West : 1>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<North : 1, West : 2>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = AIE.switchbox(%[[VAL_1]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<East : 0, Core : 1>
// CHECK:             AIE.connect<Core : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = AIE.switchbox(%[[VAL_2]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<Core : 0, East : 0>
// CHECK:             AIE.connect<South : 1, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = AIE.switchbox(%[[VAL_14]]) {
// CHECK:             AIE.connect<North : 0, Core : 0>
// CHECK:             AIE.connect<Core : 0, North : 0>
// CHECK:             AIE.connect<West : 0, Core : 1>
// CHECK:             AIE.connect<Core : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = AIE.switchbox(%[[VAL_15]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<North : 1, East : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<West : 1, Core : 1>
// CHECK:             AIE.connect<Core : 1, West : 1>
// CHECK:             AIE.connect<South : 1, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = AIE.switchbox(%[[VAL_6]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<East : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = AIE.switchbox(%[[VAL_11]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_35:.*]] = AIE.switchbox(%[[VAL_34]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:             AIE.connect<East : 1, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = AIE.tile(4, 3)
// CHECK:           %[[VAL_37:.*]] = AIE.switchbox(%[[VAL_36]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<South : 0, East : 1>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_38:.*]] = AIE.tile(5, 3)
// CHECK:           %[[VAL_39:.*]] = AIE.switchbox(%[[VAL_38]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<South : 0, East : 2>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_40:.*]] = AIE.tile(6, 3)
// CHECK:           %[[VAL_41:.*]] = AIE.switchbox(%[[VAL_40]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<West : 2, East : 2>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_42:.*]] = AIE.switchbox(%[[VAL_16]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<West : 1, Core : 0>
// CHECK:             AIE.connect<Core : 0, South : 1>
// CHECK:             AIE.connect<West : 2, East : 1>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_43:.*]] = AIE.switchbox(%[[VAL_19]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<West : 1, Core : 0>
// CHECK:             AIE.connect<DMA : 0, West : 0>
// CHECK:             AIE.connect<West : 2, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_44:.*]] = AIE.switchbox(%[[VAL_20]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<West : 0, Core : 0>
// CHECK:             AIE.connect<Core : 0, West : 0>
// CHECK:             AIE.connect<West : 1, DMA : 1>
// CHECK:             AIE.connect<Core : 1, West : 1>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_45:.*]] = AIE.switchbox(%[[VAL_21]]) {
// CHECK:             AIE.connect<South : 0, Core : 0>
// CHECK:             AIE.connect<Core : 0, West : 0>
// CHECK:             AIE.connect<South : 1, Core : 1>
// CHECK:             AIE.connect<DMA : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_46:.*]] = AIE.switchbox(%[[VAL_9]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<South : 0, West : 1>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 1, Core : 0>
// CHECK:             AIE.connect<Core : 0, East : 1>
// CHECK:             AIE.connect<South : 1, North : 0>
// CHECK:             AIE.connect<West : 1, East : 2>
// CHECK:             AIE.connect<East : 2, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_47:.*]] = AIE.tile(3, 1)
// CHECK:           %[[VAL_48:.*]] = AIE.switchbox(%[[VAL_47]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<North : 0, West : 1>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<North : 1, South : 0>
// CHECK:             AIE.connect<West : 2, East : 2>
// CHECK:             AIE.connect<East : 1, West : 2>
// CHECK:           }
// CHECK:           %[[VAL_49:.*]] = AIE.tile(4, 1)
// CHECK:           %[[VAL_50:.*]] = AIE.switchbox(%[[VAL_49]]) {
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, North : 0>
// CHECK:             AIE.connect<West : 2, East : 1>
// CHECK:             AIE.connect<North : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_51:.*]] = AIE.tile(4, 2)
// CHECK:           %[[VAL_52:.*]] = AIE.switchbox(%[[VAL_51]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_53:.*]] = AIE.tile(4, 4)
// CHECK:           %[[VAL_54:.*]] = AIE.switchbox(%[[VAL_53]]) {
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_55:.*]] = AIE.tile(5, 4)
// CHECK:           %[[VAL_56:.*]] = AIE.switchbox(%[[VAL_55]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_57:.*]] = AIE.tile(6, 4)
// CHECK:           %[[VAL_58:.*]] = AIE.switchbox(%[[VAL_57]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_59:.*]] = AIE.switchbox(%[[VAL_17]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_60:.*]] = AIE.switchbox(%[[VAL_5]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, North : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<North : 1, South : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_61:.*]] = AIE.switchbox(%[[VAL_10]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_62:.*]] = AIE.switchbox(%[[VAL_12]]) {
// CHECK:             AIE.connect<South : 0, Core : 0>
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<West : 0, Core : 1>
// CHECK:             AIE.connect<Core : 1, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_63:.*]] = AIE.switchbox(%[[VAL_13]]) {
// CHECK:             AIE.connect<South : 3, West : 0>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_64:.*]] = AIE.shimmux(%[[VAL_13]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_65:.*]] = AIE.switchbox(%[[VAL_7]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<Core : 0, South : 0>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_66:.*]] = AIE.tile(5, 1)
// CHECK:           %[[VAL_67:.*]] = AIE.switchbox(%[[VAL_66]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_68:.*]] = AIE.tile(6, 1)
// CHECK:           %[[VAL_69:.*]] = AIE.switchbox(%[[VAL_68]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<West : 1, North : 1>
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_70:.*]] = AIE.tile(6, 2)
// CHECK:           %[[VAL_71:.*]] = AIE.switchbox(%[[VAL_70]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<South : 1, East : 1>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_72:.*]] = AIE.tile(3, 2)
// CHECK:           %[[VAL_73:.*]] = AIE.switchbox(%[[VAL_72]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<East : 0, South : 1>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 1, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_74:.*]] = AIE.tile(5, 2)
// CHECK:           %[[VAL_75:.*]] = AIE.switchbox(%[[VAL_74]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_76:.*]] = AIE.switchbox(%[[VAL_3]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_77:.*]] = AIE.tile(3, 4)
// CHECK:           %[[VAL_78:.*]] = AIE.switchbox(%[[VAL_77]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_0]] : Core, %[[VAL_26]] : Core)
// CHECK:           AIE.wire(%[[VAL_0]] : DMA, %[[VAL_26]] : DMA)
// CHECK:           AIE.wire(%[[VAL_1]] : Core, %[[VAL_28]] : Core)
// CHECK:           AIE.wire(%[[VAL_1]] : DMA, %[[VAL_28]] : DMA)
// CHECK:           AIE.wire(%[[VAL_26]] : North, %[[VAL_28]] : South)
// CHECK:           AIE.wire(%[[VAL_2]] : Core, %[[VAL_29]] : Core)
// CHECK:           AIE.wire(%[[VAL_2]] : DMA, %[[VAL_29]] : DMA)
// CHECK:           AIE.wire(%[[VAL_28]] : North, %[[VAL_29]] : South)
// CHECK:           AIE.wire(%[[VAL_3]] : Core, %[[VAL_76]] : Core)
// CHECK:           AIE.wire(%[[VAL_3]] : DMA, %[[VAL_76]] : DMA)
// CHECK:           AIE.wire(%[[VAL_29]] : North, %[[VAL_76]] : South)
// CHECK:           AIE.wire(%[[VAL_26]] : East, %[[VAL_27]] : West)
// CHECK:           AIE.wire(%[[VAL_4]] : Core, %[[VAL_27]] : Core)
// CHECK:           AIE.wire(%[[VAL_4]] : DMA, %[[VAL_27]] : DMA)
// CHECK:           AIE.wire(%[[VAL_23]] : North, %[[VAL_27]] : South)
// CHECK:           AIE.wire(%[[VAL_28]] : East, %[[VAL_60]] : West)
// CHECK:           AIE.wire(%[[VAL_5]] : Core, %[[VAL_60]] : Core)
// CHECK:           AIE.wire(%[[VAL_5]] : DMA, %[[VAL_60]] : DMA)
// CHECK:           AIE.wire(%[[VAL_27]] : North, %[[VAL_60]] : South)
// CHECK:           AIE.wire(%[[VAL_29]] : East, %[[VAL_32]] : West)
// CHECK:           AIE.wire(%[[VAL_6]] : Core, %[[VAL_32]] : Core)
// CHECK:           AIE.wire(%[[VAL_6]] : DMA, %[[VAL_32]] : DMA)
// CHECK:           AIE.wire(%[[VAL_60]] : North, %[[VAL_32]] : South)
// CHECK:           AIE.wire(%[[VAL_76]] : East, %[[VAL_65]] : West)
// CHECK:           AIE.wire(%[[VAL_7]] : Core, %[[VAL_65]] : Core)
// CHECK:           AIE.wire(%[[VAL_7]] : DMA, %[[VAL_65]] : DMA)
// CHECK:           AIE.wire(%[[VAL_32]] : North, %[[VAL_65]] : South)
// CHECK:           AIE.wire(%[[VAL_23]] : East, %[[VAL_24]] : West)
// CHECK:           AIE.wire(%[[VAL_25]] : North, %[[VAL_24]] : South)
// CHECK:           AIE.wire(%[[VAL_8]] : DMA, %[[VAL_25]] : DMA)
// CHECK:           AIE.wire(%[[VAL_27]] : East, %[[VAL_46]] : West)
// CHECK:           AIE.wire(%[[VAL_9]] : Core, %[[VAL_46]] : Core)
// CHECK:           AIE.wire(%[[VAL_9]] : DMA, %[[VAL_46]] : DMA)
// CHECK:           AIE.wire(%[[VAL_24]] : North, %[[VAL_46]] : South)
// CHECK:           AIE.wire(%[[VAL_60]] : East, %[[VAL_61]] : West)
// CHECK:           AIE.wire(%[[VAL_10]] : Core, %[[VAL_61]] : Core)
// CHECK:           AIE.wire(%[[VAL_10]] : DMA, %[[VAL_61]] : DMA)
// CHECK:           AIE.wire(%[[VAL_46]] : North, %[[VAL_61]] : South)
// CHECK:           AIE.wire(%[[VAL_32]] : East, %[[VAL_33]] : West)
// CHECK:           AIE.wire(%[[VAL_11]] : Core, %[[VAL_33]] : Core)
// CHECK:           AIE.wire(%[[VAL_11]] : DMA, %[[VAL_33]] : DMA)
// CHECK:           AIE.wire(%[[VAL_61]] : North, %[[VAL_33]] : South)
// CHECK:           AIE.wire(%[[VAL_65]] : East, %[[VAL_62]] : West)
// CHECK:           AIE.wire(%[[VAL_12]] : Core, %[[VAL_62]] : Core)
// CHECK:           AIE.wire(%[[VAL_12]] : DMA, %[[VAL_62]] : DMA)
// CHECK:           AIE.wire(%[[VAL_33]] : North, %[[VAL_62]] : South)
// CHECK:           AIE.wire(%[[VAL_24]] : East, %[[VAL_63]] : West)
// CHECK:           AIE.wire(%[[VAL_64]] : North, %[[VAL_63]] : South)
// CHECK:           AIE.wire(%[[VAL_13]] : DMA, %[[VAL_64]] : DMA)
// CHECK:           AIE.wire(%[[VAL_46]] : East, %[[VAL_48]] : West)
// CHECK:           AIE.wire(%[[VAL_47]] : Core, %[[VAL_48]] : Core)
// CHECK:           AIE.wire(%[[VAL_47]] : DMA, %[[VAL_48]] : DMA)
// CHECK:           AIE.wire(%[[VAL_63]] : North, %[[VAL_48]] : South)
// CHECK:           AIE.wire(%[[VAL_61]] : East, %[[VAL_73]] : West)
// CHECK:           AIE.wire(%[[VAL_72]] : Core, %[[VAL_73]] : Core)
// CHECK:           AIE.wire(%[[VAL_72]] : DMA, %[[VAL_73]] : DMA)
// CHECK:           AIE.wire(%[[VAL_48]] : North, %[[VAL_73]] : South)
// CHECK:           AIE.wire(%[[VAL_33]] : East, %[[VAL_35]] : West)
// CHECK:           AIE.wire(%[[VAL_34]] : Core, %[[VAL_35]] : Core)
// CHECK:           AIE.wire(%[[VAL_34]] : DMA, %[[VAL_35]] : DMA)
// CHECK:           AIE.wire(%[[VAL_73]] : North, %[[VAL_35]] : South)
// CHECK:           AIE.wire(%[[VAL_62]] : East, %[[VAL_78]] : West)
// CHECK:           AIE.wire(%[[VAL_77]] : Core, %[[VAL_78]] : Core)
// CHECK:           AIE.wire(%[[VAL_77]] : DMA, %[[VAL_78]] : DMA)
// CHECK:           AIE.wire(%[[VAL_35]] : North, %[[VAL_78]] : South)
// CHECK:           AIE.wire(%[[VAL_48]] : East, %[[VAL_50]] : West)
// CHECK:           AIE.wire(%[[VAL_49]] : Core, %[[VAL_50]] : Core)
// CHECK:           AIE.wire(%[[VAL_49]] : DMA, %[[VAL_50]] : DMA)
// CHECK:           AIE.wire(%[[VAL_73]] : East, %[[VAL_52]] : West)
// CHECK:           AIE.wire(%[[VAL_51]] : Core, %[[VAL_52]] : Core)
// CHECK:           AIE.wire(%[[VAL_51]] : DMA, %[[VAL_52]] : DMA)
// CHECK:           AIE.wire(%[[VAL_50]] : North, %[[VAL_52]] : South)
// CHECK:           AIE.wire(%[[VAL_35]] : East, %[[VAL_37]] : West)
// CHECK:           AIE.wire(%[[VAL_36]] : Core, %[[VAL_37]] : Core)
// CHECK:           AIE.wire(%[[VAL_36]] : DMA, %[[VAL_37]] : DMA)
// CHECK:           AIE.wire(%[[VAL_52]] : North, %[[VAL_37]] : South)
// CHECK:           AIE.wire(%[[VAL_78]] : East, %[[VAL_54]] : West)
// CHECK:           AIE.wire(%[[VAL_53]] : Core, %[[VAL_54]] : Core)
// CHECK:           AIE.wire(%[[VAL_53]] : DMA, %[[VAL_54]] : DMA)
// CHECK:           AIE.wire(%[[VAL_37]] : North, %[[VAL_54]] : South)
// CHECK:           AIE.wire(%[[VAL_50]] : East, %[[VAL_67]] : West)
// CHECK:           AIE.wire(%[[VAL_66]] : Core, %[[VAL_67]] : Core)
// CHECK:           AIE.wire(%[[VAL_66]] : DMA, %[[VAL_67]] : DMA)
// CHECK:           AIE.wire(%[[VAL_52]] : East, %[[VAL_75]] : West)
// CHECK:           AIE.wire(%[[VAL_74]] : Core, %[[VAL_75]] : Core)
// CHECK:           AIE.wire(%[[VAL_74]] : DMA, %[[VAL_75]] : DMA)
// CHECK:           AIE.wire(%[[VAL_67]] : North, %[[VAL_75]] : South)
// CHECK:           AIE.wire(%[[VAL_37]] : East, %[[VAL_39]] : West)
// CHECK:           AIE.wire(%[[VAL_38]] : Core, %[[VAL_39]] : Core)
// CHECK:           AIE.wire(%[[VAL_38]] : DMA, %[[VAL_39]] : DMA)
// CHECK:           AIE.wire(%[[VAL_75]] : North, %[[VAL_39]] : South)
// CHECK:           AIE.wire(%[[VAL_54]] : East, %[[VAL_56]] : West)
// CHECK:           AIE.wire(%[[VAL_55]] : Core, %[[VAL_56]] : Core)
// CHECK:           AIE.wire(%[[VAL_55]] : DMA, %[[VAL_56]] : DMA)
// CHECK:           AIE.wire(%[[VAL_39]] : North, %[[VAL_56]] : South)
// CHECK:           AIE.wire(%[[VAL_67]] : East, %[[VAL_69]] : West)
// CHECK:           AIE.wire(%[[VAL_68]] : Core, %[[VAL_69]] : Core)
// CHECK:           AIE.wire(%[[VAL_68]] : DMA, %[[VAL_69]] : DMA)
// CHECK:           AIE.wire(%[[VAL_75]] : East, %[[VAL_71]] : West)
// CHECK:           AIE.wire(%[[VAL_70]] : Core, %[[VAL_71]] : Core)
// CHECK:           AIE.wire(%[[VAL_70]] : DMA, %[[VAL_71]] : DMA)
// CHECK:           AIE.wire(%[[VAL_69]] : North, %[[VAL_71]] : South)
// CHECK:           AIE.wire(%[[VAL_39]] : East, %[[VAL_41]] : West)
// CHECK:           AIE.wire(%[[VAL_40]] : Core, %[[VAL_41]] : Core)
// CHECK:           AIE.wire(%[[VAL_40]] : DMA, %[[VAL_41]] : DMA)
// CHECK:           AIE.wire(%[[VAL_71]] : North, %[[VAL_41]] : South)
// CHECK:           AIE.wire(%[[VAL_56]] : East, %[[VAL_58]] : West)
// CHECK:           AIE.wire(%[[VAL_57]] : Core, %[[VAL_58]] : Core)
// CHECK:           AIE.wire(%[[VAL_57]] : DMA, %[[VAL_58]] : DMA)
// CHECK:           AIE.wire(%[[VAL_41]] : North, %[[VAL_58]] : South)
// CHECK:           AIE.wire(%[[VAL_69]] : East, %[[VAL_30]] : West)
// CHECK:           AIE.wire(%[[VAL_14]] : Core, %[[VAL_30]] : Core)
// CHECK:           AIE.wire(%[[VAL_14]] : DMA, %[[VAL_30]] : DMA)
// CHECK:           AIE.wire(%[[VAL_71]] : East, %[[VAL_31]] : West)
// CHECK:           AIE.wire(%[[VAL_15]] : Core, %[[VAL_31]] : Core)
// CHECK:           AIE.wire(%[[VAL_15]] : DMA, %[[VAL_31]] : DMA)
// CHECK:           AIE.wire(%[[VAL_30]] : North, %[[VAL_31]] : South)
// CHECK:           AIE.wire(%[[VAL_41]] : East, %[[VAL_42]] : West)
// CHECK:           AIE.wire(%[[VAL_16]] : Core, %[[VAL_42]] : Core)
// CHECK:           AIE.wire(%[[VAL_16]] : DMA, %[[VAL_42]] : DMA)
// CHECK:           AIE.wire(%[[VAL_31]] : North, %[[VAL_42]] : South)
// CHECK:           AIE.wire(%[[VAL_58]] : East, %[[VAL_59]] : West)
// CHECK:           AIE.wire(%[[VAL_17]] : Core, %[[VAL_59]] : Core)
// CHECK:           AIE.wire(%[[VAL_17]] : DMA, %[[VAL_59]] : DMA)
// CHECK:           AIE.wire(%[[VAL_42]] : North, %[[VAL_59]] : South)
// CHECK:           AIE.wire(%[[VAL_31]] : East, %[[VAL_43]] : West)
// CHECK:           AIE.wire(%[[VAL_19]] : Core, %[[VAL_43]] : Core)
// CHECK:           AIE.wire(%[[VAL_19]] : DMA, %[[VAL_43]] : DMA)
// CHECK:           AIE.wire(%[[VAL_42]] : East, %[[VAL_44]] : West)
// CHECK:           AIE.wire(%[[VAL_20]] : Core, %[[VAL_44]] : Core)
// CHECK:           AIE.wire(%[[VAL_20]] : DMA, %[[VAL_44]] : DMA)
// CHECK:           AIE.wire(%[[VAL_43]] : North, %[[VAL_44]] : South)
// CHECK:           AIE.wire(%[[VAL_59]] : East, %[[VAL_45]] : West)
// CHECK:           AIE.wire(%[[VAL_21]] : Core, %[[VAL_45]] : Core)
// CHECK:           AIE.wire(%[[VAL_21]] : DMA, %[[VAL_45]] : DMA)
// CHECK:           AIE.wire(%[[VAL_44]] : North, %[[VAL_45]] : South)
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
        %t71 = AIE.tile(7, 1)
        %t72 = AIE.tile(7, 2)
        %t73 = AIE.tile(7, 3)
        %t74 = AIE.tile(7, 4)
        %t81 = AIE.tile(8, 1)
        %t82 = AIE.tile(8, 2)
        %t83 = AIE.tile(8, 3)
        %t84 = AIE.tile(8, 4)

        //TASK 1
        AIE.flow(%t20, DMA : 0, %t03, DMA : 0)
        AIE.flow(%t03, Core : 0, %t71, Core : 0)
        AIE.flow(%t71, Core : 0, %t84, Core : 0)
        AIE.flow(%t84, Core : 0, %t11, Core : 0)
        AIE.flow(%t11, Core : 0, %t24, Core : 0)
        AIE.flow(%t24, DMA : 0, %t20, DMA : 0)

        //TASK 2
        AIE.flow(%t30, DMA : 0, %t14, DMA : 0)
        AIE.flow(%t14, Core : 0, %t01, Core : 0)
        AIE.flow(%t01, Core : 0, %t83, Core : 0)
        AIE.flow(%t83, Core : 0, %t21, Core : 0)
        AIE.flow(%t21, Core : 0, %t73, Core : 0)
        AIE.flow(%t73, Core : 0, %t82, Core : 0)
        AIE.flow(%t82, DMA : 0, %t30, DMA : 0)

        //TASK 3
        AIE.flow(%t20, DMA : 1, %t83, DMA : 1)
        AIE.flow(%t83, Core : 1, %t01, Core : 1)
        AIE.flow(%t01, Core : 1, %t72, Core : 1)
        AIE.flow(%t72, Core : 1, %t02, Core : 1)
        AIE.flow(%t02, Core : 1, %t24, Core : 1)
        AIE.flow(%t24, Core : 1, %t71, Core : 1)
        AIE.flow(%t71, Core : 1, %t84, Core : 1)
        AIE.flow(%t84, DMA : 1, %t20, DMA : 1)
    }
}
