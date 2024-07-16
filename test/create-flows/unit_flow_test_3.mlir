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
// CHECK:           %[[VAL_14:.*]] = aie.tile(7, 1)
// CHECK:           %[[VAL_15:.*]] = aie.tile(7, 2)
// CHECK:           %[[VAL_16:.*]] = aie.tile(7, 3)
// CHECK:           %[[VAL_17:.*]] = aie.tile(7, 4)
// CHECK:           %[[VAL_18:.*]] = aie.tile(8, 1)
// CHECK:           %[[VAL_19:.*]] = aie.tile(8, 2)
// CHECK:           %[[VAL_20:.*]] = aie.tile(8, 3)
// CHECK:           %[[VAL_21:.*]] = aie.tile(8, 4)
// CHECK:           %[[VAL_22:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, Core : 0>
// CHECK:             aie.connect<Core : 0, East : 0>
// CHECK:             aie.connect<East : 2, Core : 1>
// CHECK:             aie.connect<Core : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = aie.switchbox(%[[VAL_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<East : 0, Core : 1>
// CHECK:             aie.connect<Core : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, East : 0>
// CHECK:             aie.connect<South : 1, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_26:.*]] = aie.switchbox(%[[VAL_25]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = aie.switchbox(%[[VAL_4]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:             aie.connect<North : 0, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<North : 1, West : 2>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = aie.switchbox(%[[VAL_8]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = aie.shim_mux(%[[VAL_8]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = aie.switchbox(%[[VAL_6]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<East : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = aie.switchbox(%[[VAL_11]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_33:.*]] = aie.switchbox(%[[VAL_32]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:             aie.connect<East : 1, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = aie.tile(4, 3)
// CHECK:           %[[VAL_35:.*]] = aie.switchbox(%[[VAL_34]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<South : 0, East : 1>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = aie.tile(5, 3)
// CHECK:           %[[VAL_37:.*]] = aie.switchbox(%[[VAL_36]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<South : 0, East : 2>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_38:.*]] = aie.tile(6, 3)
// CHECK:           %[[VAL_39:.*]] = aie.switchbox(%[[VAL_38]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<West : 2, East : 2>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_40:.*]] = aie.switchbox(%[[VAL_14]]) {
// CHECK:             aie.connect<North : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:             aie.connect<West : 0, Core : 1>
// CHECK:             aie.connect<Core : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = aie.switchbox(%[[VAL_15]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<North : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 1, Core : 1>
// CHECK:             aie.connect<Core : 1, West : 1>
// CHECK:             aie.connect<South : 1, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_42:.*]] = aie.switchbox(%[[VAL_16]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 1, Core : 0>
// CHECK:             aie.connect<Core : 0, South : 1>
// CHECK:             aie.connect<West : 2, East : 1>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_43:.*]] = aie.switchbox(%[[VAL_19]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, Core : 0>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:             aie.connect<West : 2, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_44:.*]] = aie.switchbox(%[[VAL_20]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<West : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:             aie.connect<West : 1, DMA : 1>
// CHECK:             aie.connect<Core : 1, West : 1>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_45:.*]] = aie.switchbox(%[[VAL_21]]) {
// CHECK:             aie.connect<South : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:             aie.connect<South : 1, Core : 1>
// CHECK:             aie.connect<DMA : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_46:.*]] = aie.switchbox(%[[VAL_9]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<South : 0, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 1, Core : 0>
// CHECK:             aie.connect<Core : 0, East : 1>
// CHECK:             aie.connect<South : 1, North : 0>
// CHECK:             aie.connect<West : 1, East : 2>
// CHECK:             aie.connect<East : 2, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_47:.*]] = aie.tile(3, 1)
// CHECK:           %[[VAL_48:.*]] = aie.switchbox(%[[VAL_47]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<North : 0, West : 1>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<North : 1, South : 0>
// CHECK:             aie.connect<West : 2, East : 2>
// CHECK:             aie.connect<East : 1, West : 2>
// CHECK:           }
// CHECK:           %[[VAL_49:.*]] = aie.tile(4, 1)
// CHECK:           %[[VAL_50:.*]] = aie.switchbox(%[[VAL_49]]) {
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, North : 0>
// CHECK:             aie.connect<West : 2, East : 1>
// CHECK:             aie.connect<North : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_51:.*]] = aie.tile(4, 2)
// CHECK:           %[[VAL_52:.*]] = aie.switchbox(%[[VAL_51]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_53:.*]] = aie.tile(4, 4)
// CHECK:           %[[VAL_54:.*]] = aie.switchbox(%[[VAL_53]]) {
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_55:.*]] = aie.tile(5, 4)
// CHECK:           %[[VAL_56:.*]] = aie.switchbox(%[[VAL_55]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_57:.*]] = aie.tile(6, 4)
// CHECK:           %[[VAL_58:.*]] = aie.switchbox(%[[VAL_57]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_59:.*]] = aie.switchbox(%[[VAL_17]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_60:.*]] = aie.switchbox(%[[VAL_5]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_61:.*]] = aie.switchbox(%[[VAL_10]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_62:.*]] = aie.switchbox(%[[VAL_12]]) {
// CHECK:             aie.connect<South : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<West : 0, Core : 1>
// CHECK:             aie.connect<Core : 1, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_63:.*]] = aie.switchbox(%[[VAL_7]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_64:.*]] = aie.switchbox(%[[VAL_13]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_65:.*]] = aie.shim_mux(%[[VAL_13]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_66:.*]] = aie.tile(5, 1)
// CHECK:           %[[VAL_67:.*]] = aie.switchbox(%[[VAL_66]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_68:.*]] = aie.tile(6, 1)
// CHECK:           %[[VAL_69:.*]] = aie.switchbox(%[[VAL_68]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, North : 1>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_70:.*]] = aie.tile(6, 2)
// CHECK:           %[[VAL_71:.*]] = aie.switchbox(%[[VAL_70]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_72:.*]] = aie.tile(3, 2)
// CHECK:           %[[VAL_73:.*]] = aie.switchbox(%[[VAL_72]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<East : 0, South : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 1, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_74:.*]] = aie.tile(5, 2)
// CHECK:           %[[VAL_75:.*]] = aie.switchbox(%[[VAL_74]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_76:.*]] = aie.switchbox(%[[VAL_3]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_77:.*]] = aie.tile(3, 4)
// CHECK:           %[[VAL_78:.*]] = aie.switchbox(%[[VAL_77]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[VAL_0]] : Core, %[[VAL_79:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_0]] : DMA, %[[VAL_79]] : DMA)
// CHECK:           aie.wire(%[[VAL_1]] : Core, %[[VAL_80:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_1]] : DMA, %[[VAL_80]] : DMA)
// CHECK:           aie.wire(%[[VAL_79]] : North, %[[VAL_80]] : South)
// CHECK:           aie.wire(%[[VAL_2]] : Core, %[[VAL_81:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_2]] : DMA, %[[VAL_81]] : DMA)
// CHECK:           aie.wire(%[[VAL_80]] : North, %[[VAL_81]] : South)
// CHECK:           aie.wire(%[[VAL_3]] : Core, %[[VAL_82:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_3]] : DMA, %[[VAL_82]] : DMA)
// CHECK:           aie.wire(%[[VAL_81]] : North, %[[VAL_82]] : South)
// CHECK:           aie.wire(%[[VAL_79]] : East, %[[VAL_83:.*]] : West)
// CHECK:           aie.wire(%[[VAL_4]] : Core, %[[VAL_83]] : Core)
// CHECK:           aie.wire(%[[VAL_4]] : DMA, %[[VAL_83]] : DMA)
// CHECK:           aie.wire(%[[VAL_84:.*]] : North, %[[VAL_83]] : South)
// CHECK:           aie.wire(%[[VAL_80]] : East, %[[VAL_85:.*]] : West)
// CHECK:           aie.wire(%[[VAL_5]] : Core, %[[VAL_85]] : Core)
// CHECK:           aie.wire(%[[VAL_5]] : DMA, %[[VAL_85]] : DMA)
// CHECK:           aie.wire(%[[VAL_83]] : North, %[[VAL_85]] : South)
// CHECK:           aie.wire(%[[VAL_81]] : East, %[[VAL_86:.*]] : West)
// CHECK:           aie.wire(%[[VAL_6]] : Core, %[[VAL_86]] : Core)
// CHECK:           aie.wire(%[[VAL_6]] : DMA, %[[VAL_86]] : DMA)
// CHECK:           aie.wire(%[[VAL_85]] : North, %[[VAL_86]] : South)
// CHECK:           aie.wire(%[[VAL_82]] : East, %[[VAL_87:.*]] : West)
// CHECK:           aie.wire(%[[VAL_7]] : Core, %[[VAL_87]] : Core)
// CHECK:           aie.wire(%[[VAL_7]] : DMA, %[[VAL_87]] : DMA)
// CHECK:           aie.wire(%[[VAL_86]] : North, %[[VAL_87]] : South)
// CHECK:           aie.wire(%[[VAL_84]] : East, %[[VAL_88:.*]] : West)
// CHECK:           aie.wire(%[[VAL_89:.*]] : North, %[[VAL_88]] : South)
// CHECK:           aie.wire(%[[VAL_8]] : DMA, %[[VAL_89]] : DMA)
// CHECK:           aie.wire(%[[VAL_83]] : East, %[[VAL_90:.*]] : West)
// CHECK:           aie.wire(%[[VAL_9]] : Core, %[[VAL_90]] : Core)
// CHECK:           aie.wire(%[[VAL_9]] : DMA, %[[VAL_90]] : DMA)
// CHECK:           aie.wire(%[[VAL_88]] : North, %[[VAL_90]] : South)
// CHECK:           aie.wire(%[[VAL_85]] : East, %[[VAL_91:.*]] : West)
// CHECK:           aie.wire(%[[VAL_10]] : Core, %[[VAL_91]] : Core)
// CHECK:           aie.wire(%[[VAL_10]] : DMA, %[[VAL_91]] : DMA)
// CHECK:           aie.wire(%[[VAL_90]] : North, %[[VAL_91]] : South)
// CHECK:           aie.wire(%[[VAL_86]] : East, %[[VAL_92:.*]] : West)
// CHECK:           aie.wire(%[[VAL_11]] : Core, %[[VAL_92]] : Core)
// CHECK:           aie.wire(%[[VAL_11]] : DMA, %[[VAL_92]] : DMA)
// CHECK:           aie.wire(%[[VAL_91]] : North, %[[VAL_92]] : South)
// CHECK:           aie.wire(%[[VAL_87]] : East, %[[VAL_93:.*]] : West)
// CHECK:           aie.wire(%[[VAL_12]] : Core, %[[VAL_93]] : Core)
// CHECK:           aie.wire(%[[VAL_12]] : DMA, %[[VAL_93]] : DMA)
// CHECK:           aie.wire(%[[VAL_92]] : North, %[[VAL_93]] : South)
// CHECK:           aie.wire(%[[VAL_88]] : East, %[[VAL_94:.*]] : West)
// CHECK:           aie.wire(%[[VAL_95:.*]] : North, %[[VAL_94]] : South)
// CHECK:           aie.wire(%[[VAL_13]] : DMA, %[[VAL_95]] : DMA)
// CHECK:           aie.wire(%[[VAL_90]] : East, %[[VAL_96:.*]] : West)
// CHECK:           aie.wire(%[[VAL_47]] : Core, %[[VAL_96]] : Core)
// CHECK:           aie.wire(%[[VAL_47]] : DMA, %[[VAL_96]] : DMA)
// CHECK:           aie.wire(%[[VAL_94]] : North, %[[VAL_96]] : South)
// CHECK:           aie.wire(%[[VAL_91]] : East, %[[VAL_97:.*]] : West)
// CHECK:           aie.wire(%[[VAL_72]] : Core, %[[VAL_97]] : Core)
// CHECK:           aie.wire(%[[VAL_72]] : DMA, %[[VAL_97]] : DMA)
// CHECK:           aie.wire(%[[VAL_96]] : North, %[[VAL_97]] : South)
// CHECK:           aie.wire(%[[VAL_92]] : East, %[[VAL_98:.*]] : West)
// CHECK:           aie.wire(%[[VAL_32]] : Core, %[[VAL_98]] : Core)
// CHECK:           aie.wire(%[[VAL_32]] : DMA, %[[VAL_98]] : DMA)
// CHECK:           aie.wire(%[[VAL_97]] : North, %[[VAL_98]] : South)
// CHECK:           aie.wire(%[[VAL_93]] : East, %[[VAL_99:.*]] : West)
// CHECK:           aie.wire(%[[VAL_77]] : Core, %[[VAL_99]] : Core)
// CHECK:           aie.wire(%[[VAL_77]] : DMA, %[[VAL_99]] : DMA)
// CHECK:           aie.wire(%[[VAL_98]] : North, %[[VAL_99]] : South)
// CHECK:           aie.wire(%[[VAL_96]] : East, %[[VAL_100:.*]] : West)
// CHECK:           aie.wire(%[[VAL_49]] : Core, %[[VAL_100]] : Core)
// CHECK:           aie.wire(%[[VAL_49]] : DMA, %[[VAL_100]] : DMA)
// CHECK:           aie.wire(%[[VAL_97]] : East, %[[VAL_101:.*]] : West)
// CHECK:           aie.wire(%[[VAL_51]] : Core, %[[VAL_101]] : Core)
// CHECK:           aie.wire(%[[VAL_51]] : DMA, %[[VAL_101]] : DMA)
// CHECK:           aie.wire(%[[VAL_100]] : North, %[[VAL_101]] : South)
// CHECK:           aie.wire(%[[VAL_98]] : East, %[[VAL_102:.*]] : West)
// CHECK:           aie.wire(%[[VAL_34]] : Core, %[[VAL_102]] : Core)
// CHECK:           aie.wire(%[[VAL_34]] : DMA, %[[VAL_102]] : DMA)
// CHECK:           aie.wire(%[[VAL_101]] : North, %[[VAL_102]] : South)
// CHECK:           aie.wire(%[[VAL_99]] : East, %[[VAL_103:.*]] : West)
// CHECK:           aie.wire(%[[VAL_53]] : Core, %[[VAL_103]] : Core)
// CHECK:           aie.wire(%[[VAL_53]] : DMA, %[[VAL_103]] : DMA)
// CHECK:           aie.wire(%[[VAL_102]] : North, %[[VAL_103]] : South)
// CHECK:           aie.wire(%[[VAL_100]] : East, %[[VAL_104:.*]] : West)
// CHECK:           aie.wire(%[[VAL_66]] : Core, %[[VAL_104]] : Core)
// CHECK:           aie.wire(%[[VAL_66]] : DMA, %[[VAL_104]] : DMA)
// CHECK:           aie.wire(%[[VAL_101]] : East, %[[VAL_105:.*]] : West)
// CHECK:           aie.wire(%[[VAL_74]] : Core, %[[VAL_105]] : Core)
// CHECK:           aie.wire(%[[VAL_74]] : DMA, %[[VAL_105]] : DMA)
// CHECK:           aie.wire(%[[VAL_104]] : North, %[[VAL_105]] : South)
// CHECK:           aie.wire(%[[VAL_102]] : East, %[[VAL_106:.*]] : West)
// CHECK:           aie.wire(%[[VAL_36]] : Core, %[[VAL_106]] : Core)
// CHECK:           aie.wire(%[[VAL_36]] : DMA, %[[VAL_106]] : DMA)
// CHECK:           aie.wire(%[[VAL_105]] : North, %[[VAL_106]] : South)
// CHECK:           aie.wire(%[[VAL_103]] : East, %[[VAL_107:.*]] : West)
// CHECK:           aie.wire(%[[VAL_55]] : Core, %[[VAL_107]] : Core)
// CHECK:           aie.wire(%[[VAL_55]] : DMA, %[[VAL_107]] : DMA)
// CHECK:           aie.wire(%[[VAL_106]] : North, %[[VAL_107]] : South)
// CHECK:           aie.wire(%[[VAL_104]] : East, %[[VAL_108:.*]] : West)
// CHECK:           aie.wire(%[[VAL_68]] : Core, %[[VAL_108]] : Core)
// CHECK:           aie.wire(%[[VAL_68]] : DMA, %[[VAL_108]] : DMA)
// CHECK:           aie.wire(%[[VAL_105]] : East, %[[VAL_109:.*]] : West)
// CHECK:           aie.wire(%[[VAL_70]] : Core, %[[VAL_109]] : Core)
// CHECK:           aie.wire(%[[VAL_70]] : DMA, %[[VAL_109]] : DMA)
// CHECK:           aie.wire(%[[VAL_108]] : North, %[[VAL_109]] : South)
// CHECK:           aie.wire(%[[VAL_106]] : East, %[[VAL_110:.*]] : West)
// CHECK:           aie.wire(%[[VAL_38]] : Core, %[[VAL_110]] : Core)
// CHECK:           aie.wire(%[[VAL_38]] : DMA, %[[VAL_110]] : DMA)
// CHECK:           aie.wire(%[[VAL_109]] : North, %[[VAL_110]] : South)
// CHECK:           aie.wire(%[[VAL_107]] : East, %[[VAL_111:.*]] : West)
// CHECK:           aie.wire(%[[VAL_57]] : Core, %[[VAL_111]] : Core)
// CHECK:           aie.wire(%[[VAL_57]] : DMA, %[[VAL_111]] : DMA)
// CHECK:           aie.wire(%[[VAL_110]] : North, %[[VAL_111]] : South)
// CHECK:           aie.wire(%[[VAL_108]] : East, %[[VAL_112:.*]] : West)
// CHECK:           aie.wire(%[[VAL_14]] : Core, %[[VAL_112]] : Core)
// CHECK:           aie.wire(%[[VAL_14]] : DMA, %[[VAL_112]] : DMA)
// CHECK:           aie.wire(%[[VAL_109]] : East, %[[VAL_113:.*]] : West)
// CHECK:           aie.wire(%[[VAL_15]] : Core, %[[VAL_113]] : Core)
// CHECK:           aie.wire(%[[VAL_15]] : DMA, %[[VAL_113]] : DMA)
// CHECK:           aie.wire(%[[VAL_112]] : North, %[[VAL_113]] : South)
// CHECK:           aie.wire(%[[VAL_110]] : East, %[[VAL_114:.*]] : West)
// CHECK:           aie.wire(%[[VAL_16]] : Core, %[[VAL_114]] : Core)
// CHECK:           aie.wire(%[[VAL_16]] : DMA, %[[VAL_114]] : DMA)
// CHECK:           aie.wire(%[[VAL_113]] : North, %[[VAL_114]] : South)
// CHECK:           aie.wire(%[[VAL_111]] : East, %[[VAL_115:.*]] : West)
// CHECK:           aie.wire(%[[VAL_17]] : Core, %[[VAL_115]] : Core)
// CHECK:           aie.wire(%[[VAL_17]] : DMA, %[[VAL_115]] : DMA)
// CHECK:           aie.wire(%[[VAL_114]] : North, %[[VAL_115]] : South)
// CHECK:           aie.wire(%[[VAL_113]] : East, %[[VAL_116:.*]] : West)
// CHECK:           aie.wire(%[[VAL_19]] : Core, %[[VAL_116]] : Core)
// CHECK:           aie.wire(%[[VAL_19]] : DMA, %[[VAL_116]] : DMA)
// CHECK:           aie.wire(%[[VAL_114]] : East, %[[VAL_117:.*]] : West)
// CHECK:           aie.wire(%[[VAL_20]] : Core, %[[VAL_117]] : Core)
// CHECK:           aie.wire(%[[VAL_20]] : DMA, %[[VAL_117]] : DMA)
// CHECK:           aie.wire(%[[VAL_116]] : North, %[[VAL_117]] : South)
// CHECK:           aie.wire(%[[VAL_115]] : East, %[[VAL_118:.*]] : West)
// CHECK:           aie.wire(%[[VAL_21]] : Core, %[[VAL_118]] : Core)
// CHECK:           aie.wire(%[[VAL_21]] : DMA, %[[VAL_118]] : DMA)
// CHECK:           aie.wire(%[[VAL_117]] : North, %[[VAL_118]] : South)
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
        %t71 = aie.tile(7, 1)
        %t72 = aie.tile(7, 2)
        %t73 = aie.tile(7, 3)
        %t74 = aie.tile(7, 4)
        %t81 = aie.tile(8, 1)
        %t82 = aie.tile(8, 2)
        %t83 = aie.tile(8, 3)
        %t84 = aie.tile(8, 4)

        //TASK 1
        aie.flow(%t20, DMA : 0, %t03, DMA : 0)
        aie.flow(%t03, Core : 0, %t71, Core : 0)
        aie.flow(%t71, Core : 0, %t84, Core : 0)
        aie.flow(%t84, Core : 0, %t11, Core : 0)
        aie.flow(%t11, Core : 0, %t24, Core : 0)
        aie.flow(%t24, DMA : 0, %t20, DMA : 0)

        //TASK 2
        aie.flow(%t30, DMA : 0, %t14, DMA : 0)
        aie.flow(%t14, Core : 0, %t01, Core : 0)
        aie.flow(%t01, Core : 0, %t83, Core : 0)
        aie.flow(%t83, Core : 0, %t21, Core : 0)
        aie.flow(%t21, Core : 0, %t73, Core : 0)
        aie.flow(%t73, Core : 0, %t82, Core : 0)
        aie.flow(%t82, DMA : 0, %t30, DMA : 0)

        //TASK 3
        aie.flow(%t20, DMA : 1, %t83, DMA : 1)
        aie.flow(%t83, Core : 1, %t01, Core : 1)
        aie.flow(%t01, Core : 1, %t72, Core : 1)
        aie.flow(%t72, Core : 1, %t02, Core : 1)
        aie.flow(%t02, Core : 1, %t24, Core : 1)
        aie.flow(%t24, Core : 1, %t71, Core : 1)
        aie.flow(%t71, Core : 1, %t84, Core : 1)
        aie.flow(%t84, DMA : 1, %t20, DMA : 1)
    }
}
