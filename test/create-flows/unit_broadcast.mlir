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
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_2:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_4:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_5:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_6:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_7:.*]] = aie.tile(3, 0)
// CHECK:           %[[VAL_8:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_9:.*]] = aie.tile(3, 1)
// CHECK:           %[[VAL_10:.*]] = aie.tile(6, 0)
// CHECK:           %[[VAL_11:.*]] = aie.tile(7, 0)
// CHECK:           %[[VAL_12:.*]] = aie.tile(7, 1)
// CHECK:           %[[VAL_13:.*]] = aie.tile(7, 2)
// CHECK:           %[[VAL_14:.*]] = aie.tile(7, 3)
// CHECK:           %[[VAL_15:.*]] = aie.tile(8, 0)
// CHECK:           %[[VAL_16:.*]] = aie.tile(8, 2)
// CHECK:           %[[VAL_17:.*]] = aie.tile(8, 3)
// CHECK:           %[[VAL_18:.*]] = aie.switchbox(%[[VAL_3]]) {
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_19:.*]] = aie.switchbox(%[[VAL_6]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 3, East : 0>
// CHECK:             aie.connect<East : 0, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = aie.shim_mux(%[[VAL_6]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_22:.*]] = aie.switchbox(%[[VAL_21]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = aie.switchbox(%[[VAL_8]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<East : 0, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_25:.*]] = aie.switchbox(%[[VAL_24]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = aie.switchbox(%[[VAL_7]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = aie.switchbox(%[[VAL_9]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<South : 0, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = aie.tile(4, 0)
// CHECK:           %[[VAL_29:.*]] = aie.switchbox(%[[VAL_28]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = aie.tile(5, 0)
// CHECK:           %[[VAL_31:.*]] = aie.switchbox(%[[VAL_30]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = aie.switchbox(%[[VAL_10]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = aie.tile(6, 1)
// CHECK:           %[[VAL_34:.*]] = aie.switchbox(%[[VAL_33]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = aie.tile(6, 2)
// CHECK:           %[[VAL_36:.*]] = aie.switchbox(%[[VAL_35]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = aie.switchbox(%[[VAL_11]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_38:.*]] = aie.switchbox(%[[VAL_12]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_39:.*]] = aie.switchbox(%[[VAL_13]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_40:.*]] = aie.switchbox(%[[VAL_16]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_42:.*]] = aie.switchbox(%[[VAL_41]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_43:.*]] = aie.switchbox(%[[VAL_1]]) {
// CHECK:             aie.connect<South : 0, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_44:.*]] = aie.switchbox(%[[VAL_4]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_45:.*]] = aie.tile(3, 2)
// CHECK:           %[[VAL_46:.*]] = aie.switchbox(%[[VAL_45]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_47:.*]] = aie.tile(4, 1)
// CHECK:           %[[VAL_48:.*]] = aie.switchbox(%[[VAL_47]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_49:.*]] = aie.tile(4, 2)
// CHECK:           %[[VAL_50:.*]] = aie.switchbox(%[[VAL_49]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_51:.*]] = aie.tile(5, 1)
// CHECK:           %[[VAL_52:.*]] = aie.switchbox(%[[VAL_51]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_53:.*]] = aie.shim_mux(%[[VAL_10]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = aie.switchbox(%[[VAL_14]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_55:.*]] = aie.switchbox(%[[VAL_17]]) {
// CHECK:             aie.connect<West : 0, DMA : 1>
// CHECK:           }
// CHECK:           aie.wire(%[[VAL_41]] : Core, %[[VAL_56:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_41]] : DMA, %[[VAL_56]] : DMA)
// CHECK:           aie.wire(%[[VAL_1]] : Core, %[[VAL_57:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_1]] : DMA, %[[VAL_57]] : DMA)
// CHECK:           aie.wire(%[[VAL_56]] : North, %[[VAL_57]] : South)
// CHECK:           aie.wire(%[[VAL_56]] : East, %[[VAL_58:.*]] : West)
// CHECK:           aie.wire(%[[VAL_4]] : Core, %[[VAL_58]] : Core)
// CHECK:           aie.wire(%[[VAL_4]] : DMA, %[[VAL_58]] : DMA)
// CHECK:           aie.wire(%[[VAL_3]] : Core, %[[VAL_59:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_3]] : DMA, %[[VAL_59]] : DMA)
// CHECK:           aie.wire(%[[VAL_60:.*]] : North, %[[VAL_61:.*]] : South)
// CHECK:           aie.wire(%[[VAL_6]] : DMA, %[[VAL_60]] : DMA)
// CHECK:           aie.wire(%[[VAL_58]] : East, %[[VAL_62:.*]] : West)
// CHECK:           aie.wire(%[[VAL_21]] : Core, %[[VAL_62]] : Core)
// CHECK:           aie.wire(%[[VAL_21]] : DMA, %[[VAL_62]] : DMA)
// CHECK:           aie.wire(%[[VAL_61]] : North, %[[VAL_62]] : South)
// CHECK:           aie.wire(%[[VAL_8]] : Core, %[[VAL_63:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_8]] : DMA, %[[VAL_63]] : DMA)
// CHECK:           aie.wire(%[[VAL_62]] : North, %[[VAL_63]] : South)
// CHECK:           aie.wire(%[[VAL_59]] : East, %[[VAL_64:.*]] : West)
// CHECK:           aie.wire(%[[VAL_24]] : Core, %[[VAL_64]] : Core)
// CHECK:           aie.wire(%[[VAL_24]] : DMA, %[[VAL_64]] : DMA)
// CHECK:           aie.wire(%[[VAL_63]] : North, %[[VAL_64]] : South)
// CHECK:           aie.wire(%[[VAL_61]] : East, %[[VAL_65:.*]] : West)
// CHECK:           aie.wire(%[[VAL_62]] : East, %[[VAL_66:.*]] : West)
// CHECK:           aie.wire(%[[VAL_9]] : Core, %[[VAL_66]] : Core)
// CHECK:           aie.wire(%[[VAL_9]] : DMA, %[[VAL_66]] : DMA)
// CHECK:           aie.wire(%[[VAL_65]] : North, %[[VAL_66]] : South)
// CHECK:           aie.wire(%[[VAL_63]] : East, %[[VAL_67:.*]] : West)
// CHECK:           aie.wire(%[[VAL_45]] : Core, %[[VAL_67]] : Core)
// CHECK:           aie.wire(%[[VAL_45]] : DMA, %[[VAL_67]] : DMA)
// CHECK:           aie.wire(%[[VAL_66]] : North, %[[VAL_67]] : South)
// CHECK:           aie.wire(%[[VAL_65]] : East, %[[VAL_68:.*]] : West)
// CHECK:           aie.wire(%[[VAL_66]] : East, %[[VAL_69:.*]] : West)
// CHECK:           aie.wire(%[[VAL_47]] : Core, %[[VAL_69]] : Core)
// CHECK:           aie.wire(%[[VAL_47]] : DMA, %[[VAL_69]] : DMA)
// CHECK:           aie.wire(%[[VAL_68]] : North, %[[VAL_69]] : South)
// CHECK:           aie.wire(%[[VAL_67]] : East, %[[VAL_70:.*]] : West)
// CHECK:           aie.wire(%[[VAL_49]] : Core, %[[VAL_70]] : Core)
// CHECK:           aie.wire(%[[VAL_49]] : DMA, %[[VAL_70]] : DMA)
// CHECK:           aie.wire(%[[VAL_69]] : North, %[[VAL_70]] : South)
// CHECK:           aie.wire(%[[VAL_68]] : East, %[[VAL_71:.*]] : West)
// CHECK:           aie.wire(%[[VAL_69]] : East, %[[VAL_72:.*]] : West)
// CHECK:           aie.wire(%[[VAL_51]] : Core, %[[VAL_72]] : Core)
// CHECK:           aie.wire(%[[VAL_51]] : DMA, %[[VAL_72]] : DMA)
// CHECK:           aie.wire(%[[VAL_71]] : North, %[[VAL_72]] : South)
// CHECK:           aie.wire(%[[VAL_71]] : East, %[[VAL_73:.*]] : West)
// CHECK:           aie.wire(%[[VAL_74:.*]] : North, %[[VAL_73]] : South)
// CHECK:           aie.wire(%[[VAL_10]] : DMA, %[[VAL_74]] : DMA)
// CHECK:           aie.wire(%[[VAL_72]] : East, %[[VAL_75:.*]] : West)
// CHECK:           aie.wire(%[[VAL_33]] : Core, %[[VAL_75]] : Core)
// CHECK:           aie.wire(%[[VAL_33]] : DMA, %[[VAL_75]] : DMA)
// CHECK:           aie.wire(%[[VAL_73]] : North, %[[VAL_75]] : South)
// CHECK:           aie.wire(%[[VAL_35]] : Core, %[[VAL_76:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_35]] : DMA, %[[VAL_76]] : DMA)
// CHECK:           aie.wire(%[[VAL_75]] : North, %[[VAL_76]] : South)
// CHECK:           aie.wire(%[[VAL_73]] : East, %[[VAL_77:.*]] : West)
// CHECK:           aie.wire(%[[VAL_75]] : East, %[[VAL_78:.*]] : West)
// CHECK:           aie.wire(%[[VAL_12]] : Core, %[[VAL_78]] : Core)
// CHECK:           aie.wire(%[[VAL_12]] : DMA, %[[VAL_78]] : DMA)
// CHECK:           aie.wire(%[[VAL_77]] : North, %[[VAL_78]] : South)
// CHECK:           aie.wire(%[[VAL_76]] : East, %[[VAL_79:.*]] : West)
// CHECK:           aie.wire(%[[VAL_13]] : Core, %[[VAL_79]] : Core)
// CHECK:           aie.wire(%[[VAL_13]] : DMA, %[[VAL_79]] : DMA)
// CHECK:           aie.wire(%[[VAL_78]] : North, %[[VAL_79]] : South)
// CHECK:           aie.wire(%[[VAL_14]] : Core, %[[VAL_80:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_14]] : DMA, %[[VAL_80]] : DMA)
// CHECK:           aie.wire(%[[VAL_79]] : North, %[[VAL_80]] : South)
// CHECK:           aie.wire(%[[VAL_79]] : East, %[[VAL_81:.*]] : West)
// CHECK:           aie.wire(%[[VAL_16]] : Core, %[[VAL_81]] : Core)
// CHECK:           aie.wire(%[[VAL_16]] : DMA, %[[VAL_81]] : DMA)
// CHECK:           aie.wire(%[[VAL_80]] : East, %[[VAL_82:.*]] : West)
// CHECK:           aie.wire(%[[VAL_17]] : Core, %[[VAL_82]] : Core)
// CHECK:           aie.wire(%[[VAL_17]] : DMA, %[[VAL_82]] : DMA)
// CHECK:           aie.wire(%[[VAL_81]] : North, %[[VAL_82]] : South)
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
