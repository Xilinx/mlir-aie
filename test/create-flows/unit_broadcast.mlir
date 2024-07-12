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

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 3)
// CHECK:           %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:           }
// CHECK:           %[[VAL_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_3:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_4:.*]] = aie.switchbox(%[[VAL_3]]) {
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_6:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_7:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_8:.*]] = aie.switchbox(%[[VAL_7]]) {
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_10:.*]] = aie.tile(3, 0)
// CHECK:           %[[VAL_11:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_12:.*]] = aie.tile(3, 1)
// CHECK:           %[[VAL_13:.*]] = aie.tile(6, 0)
// CHECK:           %[[VAL_14:.*]] = aie.tile(7, 0)
// CHECK:           %[[VAL_15:.*]] = aie.tile(7, 1)
// CHECK:           %[[VAL_16:.*]] = aie.tile(7, 2)
// CHECK:           %[[VAL_17:.*]] = aie.tile(7, 3)
// CHECK:           %[[VAL_18:.*]] = aie.tile(8, 0)
// CHECK:           %[[VAL_19:.*]] = aie.switchbox(%[[VAL_18]]) {
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = aie.tile(8, 2)
// CHECK:           %[[VAL_21:.*]] = aie.tile(8, 3)
// CHECK:           %[[VAL_22:.*]] = aie.switchbox(%[[VAL_9]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 3, East : 0>
// CHECK:             aie.connect<East : 0, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = aie.shim_mux(%[[VAL_9]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = aie.switchbox(%[[VAL_10]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = aie.tile(4, 0)
// CHECK:           %[[VAL_26:.*]] = aie.switchbox(%[[VAL_25]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = aie.tile(5, 0)
// CHECK:           %[[VAL_28:.*]] = aie.switchbox(%[[VAL_27]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = aie.switchbox(%[[VAL_13]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = aie.switchbox(%[[VAL_14]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_32:.*]] = aie.switchbox(%[[VAL_31]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = aie.switchbox(%[[VAL_12]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<South : 0, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = aie.tile(6, 1)
// CHECK:           %[[VAL_35:.*]] = aie.switchbox(%[[VAL_34]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = aie.switchbox(%[[VAL_15]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = aie.switchbox(%[[VAL_11]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<East : 0, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_38:.*]] = aie.tile(6, 2)
// CHECK:           %[[VAL_39:.*]] = aie.switchbox(%[[VAL_38]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_40:.*]] = aie.switchbox(%[[VAL_16]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = aie.switchbox(%[[VAL_20]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_42:.*]] = aie.switchbox(%[[VAL_5]]) {
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_43:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_44:.*]] = aie.switchbox(%[[VAL_43]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_45:.*]] = aie.shim_mux(%[[VAL_13]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_46:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_47:.*]] = aie.switchbox(%[[VAL_46]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_48:.*]] = aie.switchbox(%[[VAL_6]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_49:.*]] = aie.tile(4, 1)
// CHECK:           %[[VAL_50:.*]] = aie.switchbox(%[[VAL_49]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_51:.*]] = aie.tile(5, 1)
// CHECK:           %[[VAL_52:.*]] = aie.switchbox(%[[VAL_51]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_53:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK:             aie.connect<South : 0, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = aie.tile(3, 2)
// CHECK:           %[[VAL_55:.*]] = aie.switchbox(%[[VAL_54]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_56:.*]] = aie.tile(4, 2)
// CHECK:           %[[VAL_57:.*]] = aie.switchbox(%[[VAL_56]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_58:.*]] = aie.switchbox(%[[VAL_17]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_59:.*]] = aie.switchbox(%[[VAL_21]]) {
// CHECK:             aie.connect<West : 0, DMA : 1>
// CHECK:           }
// CHECK:           aie.wire(%[[VAL_46]] : Core, %[[VAL_60:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_46]] : DMA, %[[VAL_60]] : DMA)
// CHECK:           aie.wire(%[[VAL_2]] : Core, %[[VAL_61:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_2]] : DMA, %[[VAL_61]] : DMA)
// CHECK:           aie.wire(%[[VAL_60]] : North, %[[VAL_61]] : South)
// CHECK:           aie.wire(%[[VAL_60]] : East, %[[VAL_62:.*]] : West)
// CHECK:           aie.wire(%[[VAL_6]] : Core, %[[VAL_62]] : Core)
// CHECK:           aie.wire(%[[VAL_6]] : DMA, %[[VAL_62]] : DMA)
// CHECK:           aie.wire(%[[VAL_5]] : Core, %[[VAL_63:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_5]] : DMA, %[[VAL_63]] : DMA)
// CHECK:           aie.wire(%[[VAL_64:.*]] : North, %[[VAL_65:.*]] : South)
// CHECK:           aie.wire(%[[VAL_9]] : DMA, %[[VAL_64]] : DMA)
// CHECK:           aie.wire(%[[VAL_62]] : East, %[[VAL_66:.*]] : West)
// CHECK:           aie.wire(%[[VAL_31]] : Core, %[[VAL_66]] : Core)
// CHECK:           aie.wire(%[[VAL_31]] : DMA, %[[VAL_66]] : DMA)
// CHECK:           aie.wire(%[[VAL_65]] : North, %[[VAL_66]] : South)
// CHECK:           aie.wire(%[[VAL_11]] : Core, %[[VAL_67:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_11]] : DMA, %[[VAL_67]] : DMA)
// CHECK:           aie.wire(%[[VAL_66]] : North, %[[VAL_67]] : South)
// CHECK:           aie.wire(%[[VAL_63]] : East, %[[VAL_68:.*]] : West)
// CHECK:           aie.wire(%[[VAL_43]] : Core, %[[VAL_68]] : Core)
// CHECK:           aie.wire(%[[VAL_43]] : DMA, %[[VAL_68]] : DMA)
// CHECK:           aie.wire(%[[VAL_67]] : North, %[[VAL_68]] : South)
// CHECK:           aie.wire(%[[VAL_65]] : East, %[[VAL_69:.*]] : West)
// CHECK:           aie.wire(%[[VAL_66]] : East, %[[VAL_70:.*]] : West)
// CHECK:           aie.wire(%[[VAL_12]] : Core, %[[VAL_70]] : Core)
// CHECK:           aie.wire(%[[VAL_12]] : DMA, %[[VAL_70]] : DMA)
// CHECK:           aie.wire(%[[VAL_69]] : North, %[[VAL_70]] : South)
// CHECK:           aie.wire(%[[VAL_67]] : East, %[[VAL_71:.*]] : West)
// CHECK:           aie.wire(%[[VAL_54]] : Core, %[[VAL_71]] : Core)
// CHECK:           aie.wire(%[[VAL_54]] : DMA, %[[VAL_71]] : DMA)
// CHECK:           aie.wire(%[[VAL_70]] : North, %[[VAL_71]] : South)
// CHECK:           aie.wire(%[[VAL_69]] : East, %[[VAL_72:.*]] : West)
// CHECK:           aie.wire(%[[VAL_70]] : East, %[[VAL_73:.*]] : West)
// CHECK:           aie.wire(%[[VAL_49]] : Core, %[[VAL_73]] : Core)
// CHECK:           aie.wire(%[[VAL_49]] : DMA, %[[VAL_73]] : DMA)
// CHECK:           aie.wire(%[[VAL_72]] : North, %[[VAL_73]] : South)
// CHECK:           aie.wire(%[[VAL_71]] : East, %[[VAL_74:.*]] : West)
// CHECK:           aie.wire(%[[VAL_56]] : Core, %[[VAL_74]] : Core)
// CHECK:           aie.wire(%[[VAL_56]] : DMA, %[[VAL_74]] : DMA)
// CHECK:           aie.wire(%[[VAL_73]] : North, %[[VAL_74]] : South)
// CHECK:           aie.wire(%[[VAL_72]] : East, %[[VAL_75:.*]] : West)
// CHECK:           aie.wire(%[[VAL_73]] : East, %[[VAL_76:.*]] : West)
// CHECK:           aie.wire(%[[VAL_51]] : Core, %[[VAL_76]] : Core)
// CHECK:           aie.wire(%[[VAL_51]] : DMA, %[[VAL_76]] : DMA)
// CHECK:           aie.wire(%[[VAL_75]] : North, %[[VAL_76]] : South)
// CHECK:           aie.wire(%[[VAL_75]] : East, %[[VAL_77:.*]] : West)
// CHECK:           aie.wire(%[[VAL_78:.*]] : North, %[[VAL_77]] : South)
// CHECK:           aie.wire(%[[VAL_13]] : DMA, %[[VAL_78]] : DMA)
// CHECK:           aie.wire(%[[VAL_76]] : East, %[[VAL_79:.*]] : West)
// CHECK:           aie.wire(%[[VAL_34]] : Core, %[[VAL_79]] : Core)
// CHECK:           aie.wire(%[[VAL_34]] : DMA, %[[VAL_79]] : DMA)
// CHECK:           aie.wire(%[[VAL_77]] : North, %[[VAL_79]] : South)
// CHECK:           aie.wire(%[[VAL_38]] : Core, %[[VAL_80:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_38]] : DMA, %[[VAL_80]] : DMA)
// CHECK:           aie.wire(%[[VAL_79]] : North, %[[VAL_80]] : South)
// CHECK:           aie.wire(%[[VAL_77]] : East, %[[VAL_81:.*]] : West)
// CHECK:           aie.wire(%[[VAL_79]] : East, %[[VAL_82:.*]] : West)
// CHECK:           aie.wire(%[[VAL_15]] : Core, %[[VAL_82]] : Core)
// CHECK:           aie.wire(%[[VAL_15]] : DMA, %[[VAL_82]] : DMA)
// CHECK:           aie.wire(%[[VAL_81]] : North, %[[VAL_82]] : South)
// CHECK:           aie.wire(%[[VAL_80]] : East, %[[VAL_83:.*]] : West)
// CHECK:           aie.wire(%[[VAL_16]] : Core, %[[VAL_83]] : Core)
// CHECK:           aie.wire(%[[VAL_16]] : DMA, %[[VAL_83]] : DMA)
// CHECK:           aie.wire(%[[VAL_82]] : North, %[[VAL_83]] : South)
// CHECK:           aie.wire(%[[VAL_17]] : Core, %[[VAL_84:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_17]] : DMA, %[[VAL_84]] : DMA)
// CHECK:           aie.wire(%[[VAL_83]] : North, %[[VAL_84]] : South)
// CHECK:           aie.wire(%[[VAL_83]] : East, %[[VAL_85:.*]] : West)
// CHECK:           aie.wire(%[[VAL_20]] : Core, %[[VAL_85]] : Core)
// CHECK:           aie.wire(%[[VAL_20]] : DMA, %[[VAL_85]] : DMA)
// CHECK:           aie.wire(%[[VAL_84]] : East, %[[VAL_86:.*]] : West)
// CHECK:           aie.wire(%[[VAL_21]] : Core, %[[VAL_86]] : Core)
// CHECK:           aie.wire(%[[VAL_21]] : DMA, %[[VAL_86]] : DMA)
// CHECK:           aie.wire(%[[VAL_85]] : North, %[[VAL_86]] : South)
// CHECK:         }

module {
    aie.device(xcvc1902) {
        %t03 = aie.tile(0, 3)
        %t02 = aie.tile(0, 2)
        %t00 = aie.tile(0, 0)
        %t13 = aie.tile(1, 3)
        %t11 = aie.tile(1, 1)
        %t10 = aie.tile(1, 0)
        %t20 = aie.tile(2, 0)
        %t30 = aie.tile(3, 0)
        %t22 = aie.tile(2, 2)
        %t31 = aie.tile(3, 1)
        %t60 = aie.tile(6, 0)
        %t70 = aie.tile(7, 0)
        %t71 = aie.tile(7, 1)
        %t72 = aie.tile(7, 2)
        %t73 = aie.tile(7, 3)
        %t80 = aie.tile(8, 0)
        %t82 = aie.tile(8, 2)
        %t83 = aie.tile(8, 3)

        aie.flow(%t20, DMA : 0, %t13, DMA : 0)
        aie.flow(%t20, DMA : 0, %t31, DMA : 0)
        aie.flow(%t20, DMA : 0, %t71, DMA : 0)
        aie.flow(%t20, DMA : 0, %t82, DMA : 0)

        aie.flow(%t60, DMA : 0, %t02, DMA : 1)
        aie.flow(%t60, DMA : 0, %t83, DMA : 1)
        aie.flow(%t60, DMA : 0, %t22, DMA : 1)
        aie.flow(%t60, DMA : 0, %t31, DMA : 1)
    }
}

