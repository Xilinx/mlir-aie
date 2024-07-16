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

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 3)
// CHECK:           %[[VAL_2:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_4:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_5:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_6:.*]] = aie.tile(3, 0)
// CHECK:           %[[VAL_7:.*]] = aie.tile(3, 1)
// CHECK:           %[[VAL_8:.*]] = aie.tile(6, 0)
// CHECK:           %[[VAL_9:.*]] = aie.tile(7, 0)
// CHECK:           %[[VAL_10:.*]] = aie.tile(7, 3)
// CHECK:           %[[VAL_11:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:             aie.connect<North : 1, East : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<North : 2, Core : 0>
// CHECK:             aie.connect<North : 3, Core : 1>
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = aie.switchbox(%[[VAL_1]]) {
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<DMA : 1, South : 1>
// CHECK:             aie.connect<Core : 1, South : 2>
// CHECK:             aie.connect<Core : 0, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_14:.*]] = aie.switchbox(%[[VAL_13]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<North : 0, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = aie.switchbox(%[[VAL_4]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:             aie.connect<West : 0, East : 1>
// CHECK:             aie.connect<North : 1, South : 2>
// CHECK:             aie.connect<East : 0, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_17:.*]] = aie.switchbox(%[[VAL_16]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<North : 1, East : 0>
// CHECK:             aie.connect<North : 2, South : 1>
// CHECK:             aie.connect<North : 3, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = aie.switchbox(%[[VAL_5]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<West : 1, South : 1>
// CHECK:             aie.connect<DMA : 0, South : 2>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:             aie.connect<North : 0, Core : 1>
// CHECK:             aie.connect<West : 2, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_19:.*]] = aie.switchbox(%[[VAL_6]]) {
// CHECK:             aie.connect<West : 0, South : 2>
// CHECK:             aie.connect<West : 1, East : 0>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = aie.shim_mux(%[[VAL_6]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = aie.switchbox(%[[VAL_7]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<DMA : 1, South : 1>
// CHECK:             aie.connect<West : 1, Core : 1>
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = aie.tile(4, 1)
// CHECK:           %[[VAL_23:.*]] = aie.switchbox(%[[VAL_22]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = aie.tile(5, 1)
// CHECK:           %[[VAL_25:.*]] = aie.switchbox(%[[VAL_24]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = aie.switchbox(%[[VAL_8]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:             aie.connect<West : 0, South : 2>
// CHECK:             aie.connect<East : 0, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = aie.tile(6, 1)
// CHECK:           %[[VAL_28:.*]] = aie.switchbox(%[[VAL_27]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = aie.switchbox(%[[VAL_9]]) {
// CHECK:             aie.connect<West : 0, South : 3>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<North : 1, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = aie.shim_mux(%[[VAL_9]]) {
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_32:.*]] = aie.switchbox(%[[VAL_31]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_34:.*]] = aie.switchbox(%[[VAL_33]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = aie.tile(4, 0)
// CHECK:           %[[VAL_37:.*]] = aie.switchbox(%[[VAL_36]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_38:.*]] = aie.tile(5, 0)
// CHECK:           %[[VAL_39:.*]] = aie.switchbox(%[[VAL_38]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_40:.*]] = aie.shim_mux(%[[VAL_8]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = aie.shim_mux(%[[VAL_4]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_42:.*]] = aie.switchbox(%[[VAL_3]]) {
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<Core : 1, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_43:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_44:.*]] = aie.switchbox(%[[VAL_43]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_45:.*]] = aie.tile(5, 2)
// CHECK:           %[[VAL_46:.*]] = aie.switchbox(%[[VAL_45]]) {
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_47:.*]] = aie.tile(6, 2)
// CHECK:           %[[VAL_48:.*]] = aie.switchbox(%[[VAL_47]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_49:.*]] = aie.tile(7, 2)
// CHECK:           %[[VAL_50:.*]] = aie.switchbox(%[[VAL_49]]) {
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<North : 1, South : 0>
// CHECK:             aie.connect<North : 2, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_51:.*]] = aie.switchbox(%[[VAL_10]]) {
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:             aie.connect<Core : 1, West : 0>
// CHECK:             aie.connect<DMA : 0, South : 1>
// CHECK:             aie.connect<DMA : 1, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_52:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_53:.*]] = aie.switchbox(%[[VAL_52]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = aie.tile(4, 3)
// CHECK:           %[[VAL_55:.*]] = aie.switchbox(%[[VAL_54]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_56:.*]] = aie.tile(5, 3)
// CHECK:           %[[VAL_57:.*]] = aie.switchbox(%[[VAL_56]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_58:.*]] = aie.tile(6, 3)
// CHECK:           %[[VAL_59:.*]] = aie.switchbox(%[[VAL_58]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_60:.*]] = aie.tile(7, 1)
// CHECK:           %[[VAL_61:.*]] = aie.switchbox(%[[VAL_60]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           aie.wire(%[[VAL_31]] : Core, %[[VAL_62:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_31]] : DMA, %[[VAL_62]] : DMA)
// CHECK:           aie.wire(%[[VAL_0]] : Core, %[[VAL_63:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_0]] : DMA, %[[VAL_63]] : DMA)
// CHECK:           aie.wire(%[[VAL_62]] : North, %[[VAL_63]] : South)
// CHECK:           aie.wire(%[[VAL_1]] : Core, %[[VAL_64:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_1]] : DMA, %[[VAL_64]] : DMA)
// CHECK:           aie.wire(%[[VAL_63]] : North, %[[VAL_64]] : South)
// CHECK:           aie.wire(%[[VAL_62]] : East, %[[VAL_65:.*]] : West)
// CHECK:           aie.wire(%[[VAL_2]] : Core, %[[VAL_65]] : Core)
// CHECK:           aie.wire(%[[VAL_2]] : DMA, %[[VAL_65]] : DMA)
// CHECK:           aie.wire(%[[VAL_66:.*]] : North, %[[VAL_65]] : South)
// CHECK:           aie.wire(%[[VAL_63]] : East, %[[VAL_67:.*]] : West)
// CHECK:           aie.wire(%[[VAL_13]] : Core, %[[VAL_67]] : Core)
// CHECK:           aie.wire(%[[VAL_13]] : DMA, %[[VAL_67]] : DMA)
// CHECK:           aie.wire(%[[VAL_65]] : North, %[[VAL_67]] : South)
// CHECK:           aie.wire(%[[VAL_64]] : East, %[[VAL_68:.*]] : West)
// CHECK:           aie.wire(%[[VAL_3]] : Core, %[[VAL_68]] : Core)
// CHECK:           aie.wire(%[[VAL_3]] : DMA, %[[VAL_68]] : DMA)
// CHECK:           aie.wire(%[[VAL_67]] : North, %[[VAL_68]] : South)
// CHECK:           aie.wire(%[[VAL_66]] : East, %[[VAL_69:.*]] : West)
// CHECK:           aie.wire(%[[VAL_70:.*]] : North, %[[VAL_69]] : South)
// CHECK:           aie.wire(%[[VAL_4]] : DMA, %[[VAL_70]] : DMA)
// CHECK:           aie.wire(%[[VAL_65]] : East, %[[VAL_71:.*]] : West)
// CHECK:           aie.wire(%[[VAL_16]] : Core, %[[VAL_71]] : Core)
// CHECK:           aie.wire(%[[VAL_16]] : DMA, %[[VAL_71]] : DMA)
// CHECK:           aie.wire(%[[VAL_69]] : North, %[[VAL_71]] : South)
// CHECK:           aie.wire(%[[VAL_67]] : East, %[[VAL_72:.*]] : West)
// CHECK:           aie.wire(%[[VAL_5]] : Core, %[[VAL_72]] : Core)
// CHECK:           aie.wire(%[[VAL_5]] : DMA, %[[VAL_72]] : DMA)
// CHECK:           aie.wire(%[[VAL_71]] : North, %[[VAL_72]] : South)
// CHECK:           aie.wire(%[[VAL_68]] : East, %[[VAL_73:.*]] : West)
// CHECK:           aie.wire(%[[VAL_43]] : Core, %[[VAL_73]] : Core)
// CHECK:           aie.wire(%[[VAL_43]] : DMA, %[[VAL_73]] : DMA)
// CHECK:           aie.wire(%[[VAL_72]] : North, %[[VAL_73]] : South)
// CHECK:           aie.wire(%[[VAL_69]] : East, %[[VAL_74:.*]] : West)
// CHECK:           aie.wire(%[[VAL_75:.*]] : North, %[[VAL_74]] : South)
// CHECK:           aie.wire(%[[VAL_6]] : DMA, %[[VAL_75]] : DMA)
// CHECK:           aie.wire(%[[VAL_71]] : East, %[[VAL_76:.*]] : West)
// CHECK:           aie.wire(%[[VAL_7]] : Core, %[[VAL_76]] : Core)
// CHECK:           aie.wire(%[[VAL_7]] : DMA, %[[VAL_76]] : DMA)
// CHECK:           aie.wire(%[[VAL_74]] : North, %[[VAL_76]] : South)
// CHECK:           aie.wire(%[[VAL_73]] : East, %[[VAL_77:.*]] : West)
// CHECK:           aie.wire(%[[VAL_52]] : Core, %[[VAL_77]] : Core)
// CHECK:           aie.wire(%[[VAL_52]] : DMA, %[[VAL_77]] : DMA)
// CHECK:           aie.wire(%[[VAL_74]] : East, %[[VAL_78:.*]] : West)
// CHECK:           aie.wire(%[[VAL_76]] : East, %[[VAL_79:.*]] : West)
// CHECK:           aie.wire(%[[VAL_22]] : Core, %[[VAL_79]] : Core)
// CHECK:           aie.wire(%[[VAL_22]] : DMA, %[[VAL_79]] : DMA)
// CHECK:           aie.wire(%[[VAL_78]] : North, %[[VAL_79]] : South)
// CHECK:           aie.wire(%[[VAL_77]] : East, %[[VAL_80:.*]] : West)
// CHECK:           aie.wire(%[[VAL_54]] : Core, %[[VAL_80]] : Core)
// CHECK:           aie.wire(%[[VAL_54]] : DMA, %[[VAL_80]] : DMA)
// CHECK:           aie.wire(%[[VAL_78]] : East, %[[VAL_81:.*]] : West)
// CHECK:           aie.wire(%[[VAL_79]] : East, %[[VAL_82:.*]] : West)
// CHECK:           aie.wire(%[[VAL_24]] : Core, %[[VAL_82]] : Core)
// CHECK:           aie.wire(%[[VAL_24]] : DMA, %[[VAL_82]] : DMA)
// CHECK:           aie.wire(%[[VAL_81]] : North, %[[VAL_82]] : South)
// CHECK:           aie.wire(%[[VAL_45]] : Core, %[[VAL_83:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_45]] : DMA, %[[VAL_83]] : DMA)
// CHECK:           aie.wire(%[[VAL_82]] : North, %[[VAL_83]] : South)
// CHECK:           aie.wire(%[[VAL_80]] : East, %[[VAL_84:.*]] : West)
// CHECK:           aie.wire(%[[VAL_56]] : Core, %[[VAL_84]] : Core)
// CHECK:           aie.wire(%[[VAL_56]] : DMA, %[[VAL_84]] : DMA)
// CHECK:           aie.wire(%[[VAL_83]] : North, %[[VAL_84]] : South)
// CHECK:           aie.wire(%[[VAL_81]] : East, %[[VAL_85:.*]] : West)
// CHECK:           aie.wire(%[[VAL_86:.*]] : North, %[[VAL_85]] : South)
// CHECK:           aie.wire(%[[VAL_8]] : DMA, %[[VAL_86]] : DMA)
// CHECK:           aie.wire(%[[VAL_82]] : East, %[[VAL_87:.*]] : West)
// CHECK:           aie.wire(%[[VAL_27]] : Core, %[[VAL_87]] : Core)
// CHECK:           aie.wire(%[[VAL_27]] : DMA, %[[VAL_87]] : DMA)
// CHECK:           aie.wire(%[[VAL_85]] : North, %[[VAL_87]] : South)
// CHECK:           aie.wire(%[[VAL_83]] : East, %[[VAL_88:.*]] : West)
// CHECK:           aie.wire(%[[VAL_47]] : Core, %[[VAL_88]] : Core)
// CHECK:           aie.wire(%[[VAL_47]] : DMA, %[[VAL_88]] : DMA)
// CHECK:           aie.wire(%[[VAL_87]] : North, %[[VAL_88]] : South)
// CHECK:           aie.wire(%[[VAL_84]] : East, %[[VAL_89:.*]] : West)
// CHECK:           aie.wire(%[[VAL_58]] : Core, %[[VAL_89]] : Core)
// CHECK:           aie.wire(%[[VAL_58]] : DMA, %[[VAL_89]] : DMA)
// CHECK:           aie.wire(%[[VAL_88]] : North, %[[VAL_89]] : South)
// CHECK:           aie.wire(%[[VAL_85]] : East, %[[VAL_90:.*]] : West)
// CHECK:           aie.wire(%[[VAL_91:.*]] : North, %[[VAL_90]] : South)
// CHECK:           aie.wire(%[[VAL_9]] : DMA, %[[VAL_91]] : DMA)
// CHECK:           aie.wire(%[[VAL_87]] : East, %[[VAL_92:.*]] : West)
// CHECK:           aie.wire(%[[VAL_60]] : Core, %[[VAL_92]] : Core)
// CHECK:           aie.wire(%[[VAL_60]] : DMA, %[[VAL_92]] : DMA)
// CHECK:           aie.wire(%[[VAL_90]] : North, %[[VAL_92]] : South)
// CHECK:           aie.wire(%[[VAL_88]] : East, %[[VAL_93:.*]] : West)
// CHECK:           aie.wire(%[[VAL_49]] : Core, %[[VAL_93]] : Core)
// CHECK:           aie.wire(%[[VAL_49]] : DMA, %[[VAL_93]] : DMA)
// CHECK:           aie.wire(%[[VAL_92]] : North, %[[VAL_93]] : South)
// CHECK:           aie.wire(%[[VAL_89]] : East, %[[VAL_94:.*]] : West)
// CHECK:           aie.wire(%[[VAL_10]] : Core, %[[VAL_94]] : Core)
// CHECK:           aie.wire(%[[VAL_10]] : DMA, %[[VAL_94]] : DMA)
// CHECK:           aie.wire(%[[VAL_93]] : North, %[[VAL_94]] : South)
// CHECK:         }

module {
    aie.device(xcvc1902) {
        %t02 = aie.tile(0, 2)
        %t03 = aie.tile(0, 3)
        %t11 = aie.tile(1, 1)
        %t13 = aie.tile(1, 3)
        %t20 = aie.tile(2, 0)
        %t22 = aie.tile(2, 2)
        %t30 = aie.tile(3, 0)
        %t31 = aie.tile(3, 1)
        %t60 = aie.tile(6, 0)
        %t70 = aie.tile(7, 0)
        %t73 = aie.tile(7, 3)

        aie.flow(%t03, DMA : 0, %t30, DMA : 0)
        aie.flow(%t03, DMA : 1, %t70, DMA : 1)
        aie.flow(%t02, DMA : 0, %t60, DMA : 0)
        aie.flow(%t22, DMA : 0, %t20, DMA : 0)

        aie.flow(%t22, Core : 0, %t13, Core : 0)
        aie.flow(%t03, Core : 1, %t02, Core : 0)
        aie.flow(%t73, Core : 0, %t31, Core : 0)
        aie.flow(%t73, Core : 1, %t22, Core : 1)

        aie.flow(%t73, DMA : 0, %t60, DMA : 1)
        aie.flow(%t73, DMA : 1, %t70, DMA : 0)
        aie.flow(%t31, DMA : 0, %t20, DMA : 1)
        aie.flow(%t31, DMA : 1, %t30, DMA : 1)

        aie.flow(%t03, Core : 0, %t02, Core : 1)
        aie.flow(%t13, Core : 1, %t31, Core : 1)
    }
}
