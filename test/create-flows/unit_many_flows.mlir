//===- many_flows.mlir -----------------------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_11:.*]] = AIE.switchbox(%[[VAL_0]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<North : 1, Core : 0>
// CHECK:             AIE.connect<Core : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = AIE.switchbox(%[[VAL_1]]) {
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<Core : 0, East : 0>
// CHECK:             AIE.connect<Core : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_14:.*]] = AIE.switchbox(%[[VAL_13]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<North : 0, East : 1>
// CHECK:             AIE.connect<West : 1, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_16:.*]] = AIE.switchbox(%[[VAL_15]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = AIE.switchbox(%[[VAL_5]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<DMA : 0, East : 0>
// CHECK:             AIE.connect<West : 1, Core : 0>
// CHECK:             AIE.connect<West : 2, Core : 1>
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = AIE.switchbox(%[[VAL_7]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:             AIE.connect<East : 1, South : 1>
// CHECK:             AIE.connect<DMA : 0, South : 2>
// CHECK:             AIE.connect<DMA : 1, South : 3>
// CHECK:             AIE.connect<East : 2, Core : 0>
// CHECK:             AIE.connect<East : 3, Core : 1>
// CHECK:           }
// CHECK:           %[[VAL_19:.*]] = AIE.tile(4, 1)
// CHECK:           %[[VAL_20:.*]] = AIE.switchbox(%[[VAL_19]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<North : 0, East : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = AIE.tile(5, 1)
// CHECK:           %[[VAL_22:.*]] = AIE.switchbox(%[[VAL_21]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, South : 0>
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<North : 1, West : 1>
// CHECK:             AIE.connect<North : 2, West : 2>
// CHECK:             AIE.connect<North : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = AIE.switchbox(%[[VAL_8]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<North : 1, East : 1>
// CHECK:             AIE.connect<West : 0, South : 2>
// CHECK:             AIE.connect<West : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = AIE.tile(6, 1)
// CHECK:           %[[VAL_25:.*]] = AIE.switchbox(%[[VAL_24]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = AIE.switchbox(%[[VAL_9]]) {
// CHECK:             AIE.connect<West : 0, South : 2>
// CHECK:             AIE.connect<West : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = AIE.shim_mux(%[[VAL_9]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = AIE.switchbox(%[[VAL_3]]) {
// CHECK:             AIE.connect<DMA : 0, East : 0>
// CHECK:             AIE.connect<West : 0, Core : 0>
// CHECK:             AIE.connect<Core : 1, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = AIE.tile(2, 3)
// CHECK:           %[[VAL_30:.*]] = AIE.switchbox(%[[VAL_29]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_32:.*]] = AIE.switchbox(%[[VAL_31]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = AIE.tile(4, 3)
// CHECK:           %[[VAL_34:.*]] = AIE.switchbox(%[[VAL_33]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = AIE.tile(5, 3)
// CHECK:           %[[VAL_36:.*]] = AIE.switchbox(%[[VAL_35]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = AIE.tile(6, 2)
// CHECK:           %[[VAL_38:.*]] = AIE.switchbox(%[[VAL_37]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_39:.*]] = AIE.tile(6, 3)
// CHECK:           %[[VAL_40:.*]] = AIE.switchbox(%[[VAL_39]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = AIE.tile(0, 1)
// CHECK:           %[[VAL_42:.*]] = AIE.switchbox(%[[VAL_41]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_43:.*]] = AIE.tile(1, 0)
// CHECK:           %[[VAL_44:.*]] = AIE.switchbox(%[[VAL_43]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_45:.*]] = AIE.switchbox(%[[VAL_2]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_46:.*]] = AIE.switchbox(%[[VAL_4]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 0, South : 2>
// CHECK:             AIE.connect<East : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_47:.*]] = AIE.switchbox(%[[VAL_6]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<North : 1, South : 2>
// CHECK:             AIE.connect<North : 2, West : 1>
// CHECK:             AIE.connect<North : 3, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_48:.*]] = AIE.tile(4, 0)
// CHECK:           %[[VAL_49:.*]] = AIE.switchbox(%[[VAL_48]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_50:.*]] = AIE.tile(5, 0)
// CHECK:           %[[VAL_51:.*]] = AIE.switchbox(%[[VAL_50]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<North : 0, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_52:.*]] = AIE.shim_mux(%[[VAL_8]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_53:.*]] = AIE.tile(3, 2)
// CHECK:           %[[VAL_54:.*]] = AIE.switchbox(%[[VAL_53]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_55:.*]] = AIE.tile(4, 2)
// CHECK:           %[[VAL_56:.*]] = AIE.switchbox(%[[VAL_55]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_57:.*]] = AIE.shim_mux(%[[VAL_4]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_58:.*]] = AIE.tile(5, 2)
// CHECK:           %[[VAL_59:.*]] = AIE.switchbox(%[[VAL_58]]) {
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:             AIE.connect<East : 1, South : 1>
// CHECK:             AIE.connect<East : 2, South : 2>
// CHECK:             AIE.connect<East : 3, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_60:.*]] = AIE.tile(7, 2)
// CHECK:           %[[VAL_61:.*]] = AIE.switchbox(%[[VAL_60]]) {
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<North : 1, West : 1>
// CHECK:             AIE.connect<North : 2, West : 2>
// CHECK:             AIE.connect<North : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_62:.*]] = AIE.switchbox(%[[VAL_10]]) {
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<DMA : 1, South : 1>
// CHECK:             AIE.connect<Core : 0, South : 2>
// CHECK:             AIE.connect<Core : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_63:.*]] = AIE.shim_mux(%[[VAL_6]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_41]] : Core, %[[VAL_64:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_41]] : DMA, %[[VAL_64]] : DMA)
// CHECK:           AIE.wire(%[[VAL_0]] : Core, %[[VAL_65:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_0]] : DMA, %[[VAL_65]] : DMA)
// CHECK:           AIE.wire(%[[VAL_64]] : North, %[[VAL_65]] : South)
// CHECK:           AIE.wire(%[[VAL_1]] : Core, %[[VAL_66:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_1]] : DMA, %[[VAL_66]] : DMA)
// CHECK:           AIE.wire(%[[VAL_65]] : North, %[[VAL_66]] : South)
// CHECK:           AIE.wire(%[[VAL_64]] : East, %[[VAL_67:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_2]] : Core, %[[VAL_67]] : Core)
// CHECK:           AIE.wire(%[[VAL_2]] : DMA, %[[VAL_67]] : DMA)
// CHECK:           AIE.wire(%[[VAL_68:.*]] : North, %[[VAL_67]] : South)
// CHECK:           AIE.wire(%[[VAL_65]] : East, %[[VAL_69:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_13]] : Core, %[[VAL_69]] : Core)
// CHECK:           AIE.wire(%[[VAL_13]] : DMA, %[[VAL_69]] : DMA)
// CHECK:           AIE.wire(%[[VAL_67]] : North, %[[VAL_69]] : South)
// CHECK:           AIE.wire(%[[VAL_66]] : East, %[[VAL_70:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_3]] : Core, %[[VAL_70]] : Core)
// CHECK:           AIE.wire(%[[VAL_3]] : DMA, %[[VAL_70]] : DMA)
// CHECK:           AIE.wire(%[[VAL_69]] : North, %[[VAL_70]] : South)
// CHECK:           AIE.wire(%[[VAL_68]] : East, %[[VAL_71:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_72:.*]] : North, %[[VAL_71]] : South)
// CHECK:           AIE.wire(%[[VAL_4]] : DMA, %[[VAL_72]] : DMA)
// CHECK:           AIE.wire(%[[VAL_67]] : East, %[[VAL_73:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_15]] : Core, %[[VAL_73]] : Core)
// CHECK:           AIE.wire(%[[VAL_15]] : DMA, %[[VAL_73]] : DMA)
// CHECK:           AIE.wire(%[[VAL_71]] : North, %[[VAL_73]] : South)
// CHECK:           AIE.wire(%[[VAL_69]] : East, %[[VAL_74:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_5]] : Core, %[[VAL_74]] : Core)
// CHECK:           AIE.wire(%[[VAL_5]] : DMA, %[[VAL_74]] : DMA)
// CHECK:           AIE.wire(%[[VAL_73]] : North, %[[VAL_74]] : South)
// CHECK:           AIE.wire(%[[VAL_70]] : East, %[[VAL_75:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_29]] : Core, %[[VAL_75]] : Core)
// CHECK:           AIE.wire(%[[VAL_29]] : DMA, %[[VAL_75]] : DMA)
// CHECK:           AIE.wire(%[[VAL_74]] : North, %[[VAL_75]] : South)
// CHECK:           AIE.wire(%[[VAL_71]] : East, %[[VAL_76:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_77:.*]] : North, %[[VAL_76]] : South)
// CHECK:           AIE.wire(%[[VAL_6]] : DMA, %[[VAL_77]] : DMA)
// CHECK:           AIE.wire(%[[VAL_73]] : East, %[[VAL_78:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_7]] : Core, %[[VAL_78]] : Core)
// CHECK:           AIE.wire(%[[VAL_7]] : DMA, %[[VAL_78]] : DMA)
// CHECK:           AIE.wire(%[[VAL_76]] : North, %[[VAL_78]] : South)
// CHECK:           AIE.wire(%[[VAL_74]] : East, %[[VAL_79:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_53]] : Core, %[[VAL_79]] : Core)
// CHECK:           AIE.wire(%[[VAL_53]] : DMA, %[[VAL_79]] : DMA)
// CHECK:           AIE.wire(%[[VAL_78]] : North, %[[VAL_79]] : South)
// CHECK:           AIE.wire(%[[VAL_75]] : East, %[[VAL_80:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_31]] : Core, %[[VAL_80]] : Core)
// CHECK:           AIE.wire(%[[VAL_31]] : DMA, %[[VAL_80]] : DMA)
// CHECK:           AIE.wire(%[[VAL_79]] : North, %[[VAL_80]] : South)
// CHECK:           AIE.wire(%[[VAL_76]] : East, %[[VAL_81:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_78]] : East, %[[VAL_82:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_19]] : Core, %[[VAL_82]] : Core)
// CHECK:           AIE.wire(%[[VAL_19]] : DMA, %[[VAL_82]] : DMA)
// CHECK:           AIE.wire(%[[VAL_81]] : North, %[[VAL_82]] : South)
// CHECK:           AIE.wire(%[[VAL_79]] : East, %[[VAL_83:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_55]] : Core, %[[VAL_83]] : Core)
// CHECK:           AIE.wire(%[[VAL_55]] : DMA, %[[VAL_83]] : DMA)
// CHECK:           AIE.wire(%[[VAL_82]] : North, %[[VAL_83]] : South)
// CHECK:           AIE.wire(%[[VAL_80]] : East, %[[VAL_84:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_33]] : Core, %[[VAL_84]] : Core)
// CHECK:           AIE.wire(%[[VAL_33]] : DMA, %[[VAL_84]] : DMA)
// CHECK:           AIE.wire(%[[VAL_83]] : North, %[[VAL_84]] : South)
// CHECK:           AIE.wire(%[[VAL_81]] : East, %[[VAL_85:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_82]] : East, %[[VAL_86:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_21]] : Core, %[[VAL_86]] : Core)
// CHECK:           AIE.wire(%[[VAL_21]] : DMA, %[[VAL_86]] : DMA)
// CHECK:           AIE.wire(%[[VAL_85]] : North, %[[VAL_86]] : South)
// CHECK:           AIE.wire(%[[VAL_83]] : East, %[[VAL_87:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_58]] : Core, %[[VAL_87]] : Core)
// CHECK:           AIE.wire(%[[VAL_58]] : DMA, %[[VAL_87]] : DMA)
// CHECK:           AIE.wire(%[[VAL_86]] : North, %[[VAL_87]] : South)
// CHECK:           AIE.wire(%[[VAL_84]] : East, %[[VAL_88:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_35]] : Core, %[[VAL_88]] : Core)
// CHECK:           AIE.wire(%[[VAL_35]] : DMA, %[[VAL_88]] : DMA)
// CHECK:           AIE.wire(%[[VAL_87]] : North, %[[VAL_88]] : South)
// CHECK:           AIE.wire(%[[VAL_85]] : East, %[[VAL_89:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_90:.*]] : North, %[[VAL_89]] : South)
// CHECK:           AIE.wire(%[[VAL_8]] : DMA, %[[VAL_90]] : DMA)
// CHECK:           AIE.wire(%[[VAL_86]] : East, %[[VAL_91:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_24]] : Core, %[[VAL_91]] : Core)
// CHECK:           AIE.wire(%[[VAL_24]] : DMA, %[[VAL_91]] : DMA)
// CHECK:           AIE.wire(%[[VAL_89]] : North, %[[VAL_91]] : South)
// CHECK:           AIE.wire(%[[VAL_87]] : East, %[[VAL_92:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_37]] : Core, %[[VAL_92]] : Core)
// CHECK:           AIE.wire(%[[VAL_37]] : DMA, %[[VAL_92]] : DMA)
// CHECK:           AIE.wire(%[[VAL_91]] : North, %[[VAL_92]] : South)
// CHECK:           AIE.wire(%[[VAL_88]] : East, %[[VAL_93:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_39]] : Core, %[[VAL_93]] : Core)
// CHECK:           AIE.wire(%[[VAL_39]] : DMA, %[[VAL_93]] : DMA)
// CHECK:           AIE.wire(%[[VAL_92]] : North, %[[VAL_93]] : South)
// CHECK:           AIE.wire(%[[VAL_89]] : East, %[[VAL_94:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_95:.*]] : North, %[[VAL_94]] : South)
// CHECK:           AIE.wire(%[[VAL_9]] : DMA, %[[VAL_95]] : DMA)
// CHECK:           AIE.wire(%[[VAL_92]] : East, %[[VAL_96:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_60]] : Core, %[[VAL_96]] : Core)
// CHECK:           AIE.wire(%[[VAL_60]] : DMA, %[[VAL_96]] : DMA)
// CHECK:           AIE.wire(%[[VAL_93]] : East, %[[VAL_97:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_10]] : Core, %[[VAL_97]] : Core)
// CHECK:           AIE.wire(%[[VAL_10]] : DMA, %[[VAL_97]] : DMA)
// CHECK:           AIE.wire(%[[VAL_96]] : North, %[[VAL_97]] : South)
// CHECK:         }

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

        AIE.flow(%t03, DMA : 0, %t70, DMA : 0)
        AIE.flow(%t13, DMA : 0, %t70, DMA : 1)
        AIE.flow(%t02, DMA : 0, %t60, DMA : 0)
        AIE.flow(%t22, DMA : 0, %t60, DMA : 1)

        AIE.flow(%t03, Core : 0, %t13, Core : 0)
        AIE.flow(%t03, Core : 1, %t02, Core : 0)
        AIE.flow(%t13, Core : 1, %t22, Core : 0)
        AIE.flow(%t02, Core : 1, %t22, Core : 1)

        AIE.flow(%t73, DMA : 0, %t20, DMA : 0)
        AIE.flow(%t73, DMA : 1, %t30, DMA : 0)
        AIE.flow(%t31, DMA : 0, %t20, DMA : 1)
        AIE.flow(%t31, DMA : 1, %t30, DMA : 1)

        AIE.flow(%t73, Core : 0, %t31, Core : 0)
        AIE.flow(%t73, Core : 1, %t31, Core : 1)
    }
}
