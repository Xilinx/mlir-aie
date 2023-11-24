//===- flow_test_1.mlir ----------------------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(3, 0)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(3, 4)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(4, 3)
// CHECK:           %[[VAL_4:.*]] = AIE.tile(4, 4)
// CHECK:           %[[VAL_5:.*]] = AIE.tile(5, 4)
// CHECK:           %[[VAL_6:.*]] = AIE.tile(6, 0)
// CHECK:           %[[VAL_7:.*]] = AIE.tile(6, 3)
// CHECK:           %[[VAL_8:.*]] = AIE.tile(7, 0)
// CHECK:           %[[VAL_9:.*]] = AIE.tile(7, 2)
// CHECK:           %[[VAL_10:.*]] = AIE.tile(8, 3)
// CHECK:           %[[VAL_11:.*]] = AIE.tile(8, 4)
// CHECK:           %[[VAL_12:.*]] = AIE.switchbox(%[[VAL_0]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:             AIE.connect<East : 0, South : 3>
// CHECK:             AIE.connect<East : 1, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = AIE.shimmux(%[[VAL_0]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_15:.*]] = AIE.switchbox(%[[VAL_14]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_17:.*]] = AIE.switchbox(%[[VAL_16]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = AIE.tile(3, 2)
// CHECK:           %[[VAL_19:.*]] = AIE.switchbox(%[[VAL_18]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<South : 0, East : 2>
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = AIE.tile(4, 2)
// CHECK:           %[[VAL_21:.*]] = AIE.switchbox(%[[VAL_20]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<West : 2, East : 2>
// CHECK:             AIE.connect<North : 0, East : 3>
// CHECK:             AIE.connect<North : 1, South : 0>
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = AIE.tile(5, 2)
// CHECK:           %[[VAL_23:.*]] = AIE.switchbox(%[[VAL_22]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<West : 1, North : 1>
// CHECK:             AIE.connect<West : 2, North : 2>
// CHECK:             AIE.connect<West : 3, South : 0>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:             AIE.connect<North : 1, South : 2>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = AIE.tile(5, 3)
// CHECK:           %[[VAL_25:.*]] = AIE.switchbox(%[[VAL_24]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, East : 1>
// CHECK:             AIE.connect<South : 2, North : 0>
// CHECK:             AIE.connect<West : 0, East : 2>
// CHECK:             AIE.connect<West : 1, South : 0>
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<North : 1, South : 1>
// CHECK:             AIE.connect<East : 0, West : 1>
// CHECK:             AIE.connect<East : 1, North : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = AIE.switchbox(%[[VAL_7]]) {
// CHECK:             AIE.connect<West : 0, DMA : 0>
// CHECK:             AIE.connect<West : 1, East : 0>
// CHECK:             AIE.connect<North : 0, Core : 1>
// CHECK:             AIE.connect<West : 2, North : 0>
// CHECK:             AIE.connect<South : 0, North : 1>
// CHECK:             AIE.connect<Core : 0, West : 0>
// CHECK:             AIE.connect<DMA : 1, South : 0>
// CHECK:             AIE.connect<East : 0, West : 1>
// CHECK:             AIE.connect<East : 1, West : 2>
// CHECK:             AIE.connect<East : 2, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = AIE.tile(7, 3)
// CHECK:           %[[VAL_28:.*]] = AIE.switchbox(%[[VAL_27]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, East : 1>
// CHECK:             AIE.connect<South : 2, East : 2>
// CHECK:             AIE.connect<East : 0, West : 1>
// CHECK:             AIE.connect<East : 1, West : 2>
// CHECK:             AIE.connect<East : 2, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = AIE.switchbox(%[[VAL_10]]) {
// CHECK:             AIE.connect<West : 0, DMA : 0>
// CHECK:             AIE.connect<West : 1, North : 0>
// CHECK:             AIE.connect<West : 2, Core : 1>
// CHECK:             AIE.connect<Core : 0, West : 0>
// CHECK:             AIE.connect<DMA : 1, West : 1>
// CHECK:             AIE.connect<North : 0, West : 2>
// CHECK:             AIE.connect<North : 1, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = AIE.switchbox(%[[VAL_1]]) {
// CHECK:             AIE.connect<South : 3, East : 0>
// CHECK:             AIE.connect<South : 7, North : 0>
// CHECK:             AIE.connect<East : 0, South : 3>
// CHECK:             AIE.connect<East : 1, West : 0>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:             AIE.connect<North : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = AIE.shimmux(%[[VAL_1]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = AIE.tile(4, 0)
// CHECK:           %[[VAL_33:.*]] = AIE.switchbox(%[[VAL_32]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = AIE.tile(5, 0)
// CHECK:           %[[VAL_35:.*]] = AIE.switchbox(%[[VAL_34]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<North : 0, East : 1>
// CHECK:             AIE.connect<North : 1, West : 0>
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:             AIE.connect<North : 2, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = AIE.switchbox(%[[VAL_6]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<North : 1, South : 3>
// CHECK:             AIE.connect<West : 1, South : 2>
// CHECK:             AIE.connect<South : 3, North : 1>
// CHECK:             AIE.connect<South : 7, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = AIE.tile(6, 1)
// CHECK:           %[[VAL_38:.*]] = AIE.switchbox(%[[VAL_37]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<West : 1, South : 1>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_39:.*]] = AIE.tile(6, 2)
// CHECK:           %[[VAL_40:.*]] = AIE.switchbox(%[[VAL_39]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, North : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = AIE.switchbox(%[[VAL_9]]) {
// CHECK:             AIE.connect<West : 0, DMA : 0>
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:             AIE.connect<Core : 0, North : 2>
// CHECK:             AIE.connect<DMA : 1, West : 0>
// CHECK:             AIE.connect<North : 0, Core : 1>
// CHECK:           }
// CHECK:           %[[VAL_42:.*]] = AIE.tile(3, 1)
// CHECK:           %[[VAL_43:.*]] = AIE.switchbox(%[[VAL_42]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_44:.*]] = AIE.switchbox(%[[VAL_5]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, Core : 1>
// CHECK:             AIE.connect<Core : 0, South : 0>
// CHECK:             AIE.connect<DMA : 1, South : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_45:.*]] = AIE.switchbox(%[[VAL_2]]) {
// CHECK:             AIE.connect<Core : 0, East : 0>
// CHECK:             AIE.connect<DMA : 1, South : 0>
// CHECK:             AIE.connect<East : 0, Core : 1>
// CHECK:             AIE.connect<East : 1, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_46:.*]] = AIE.switchbox(%[[VAL_4]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<Core : 0, East : 1>
// CHECK:             AIE.connect<DMA : 1, South : 0>
// CHECK:             AIE.connect<East : 0, DMA : 0>
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<South : 1, Core : 1>
// CHECK:           }
// CHECK:           %[[VAL_47:.*]] = AIE.tile(6, 4)
// CHECK:           %[[VAL_48:.*]] = AIE.switchbox(%[[VAL_47]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_49:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_50:.*]] = AIE.switchbox(%[[VAL_49]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_51:.*]] = AIE.switchbox(%[[VAL_3]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<Core : 0, East : 0>
// CHECK:             AIE.connect<DMA : 1, East : 1>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:             AIE.connect<East : 0, Core : 1>
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<East : 1, North : 0>
// CHECK:             AIE.connect<East : 2, North : 1>
// CHECK:             AIE.connect<East : 3, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_52:.*]] = AIE.tile(5, 1)
// CHECK:           %[[VAL_53:.*]] = AIE.switchbox(%[[VAL_52]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<North : 1, East : 1>
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<North : 2, South : 1>
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<East : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = AIE.switchbox(%[[VAL_8]]) {
// CHECK:             AIE.connect<West : 0, South : 2>
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:             AIE.connect<East : 0, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_55:.*]] = AIE.shimmux(%[[VAL_8]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_56:.*]] = AIE.tile(7, 4)
// CHECK:           %[[VAL_57:.*]] = AIE.switchbox(%[[VAL_56]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_58:.*]] = AIE.switchbox(%[[VAL_11]]) {
// CHECK:             AIE.connect<West : 0, Core : 1>
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<Core : 0, South : 0>
// CHECK:             AIE.connect<DMA : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_59:.*]] = AIE.shimmux(%[[VAL_6]]) {
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_60:.*]] = AIE.tile(4, 1)
// CHECK:           %[[VAL_61:.*]] = AIE.switchbox(%[[VAL_60]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_62:.*]] = AIE.tile(7, 1)
// CHECK:           %[[VAL_63:.*]] = AIE.switchbox(%[[VAL_62]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_64:.*]] = AIE.tile(8, 0)
// CHECK:           %[[VAL_65:.*]] = AIE.switchbox(%[[VAL_64]]) {
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_66:.*]] = AIE.tile(8, 1)
// CHECK:           %[[VAL_67:.*]] = AIE.switchbox(%[[VAL_66]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_68:.*]] = AIE.tile(8, 2)
// CHECK:           %[[VAL_69:.*]] = AIE.switchbox(%[[VAL_68]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_70:.*]] : North, %[[VAL_71:.*]] : South)
// CHECK:           AIE.wire(%[[VAL_0]] : DMA, %[[VAL_70]] : DMA)
// CHECK:           AIE.wire(%[[VAL_14]] : Core, %[[VAL_72:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_14]] : DMA, %[[VAL_72]] : DMA)
// CHECK:           AIE.wire(%[[VAL_71]] : North, %[[VAL_72]] : South)
// CHECK:           AIE.wire(%[[VAL_16]] : Core, %[[VAL_73:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_16]] : DMA, %[[VAL_73]] : DMA)
// CHECK:           AIE.wire(%[[VAL_72]] : North, %[[VAL_73]] : South)
// CHECK:           AIE.wire(%[[VAL_71]] : East, %[[VAL_74:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_75:.*]] : North, %[[VAL_74]] : South)
// CHECK:           AIE.wire(%[[VAL_1]] : DMA, %[[VAL_75]] : DMA)
// CHECK:           AIE.wire(%[[VAL_72]] : East, %[[VAL_76:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_42]] : Core, %[[VAL_76]] : Core)
// CHECK:           AIE.wire(%[[VAL_42]] : DMA, %[[VAL_76]] : DMA)
// CHECK:           AIE.wire(%[[VAL_74]] : North, %[[VAL_76]] : South)
// CHECK:           AIE.wire(%[[VAL_73]] : East, %[[VAL_77:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_18]] : Core, %[[VAL_77]] : Core)
// CHECK:           AIE.wire(%[[VAL_18]] : DMA, %[[VAL_77]] : DMA)
// CHECK:           AIE.wire(%[[VAL_76]] : North, %[[VAL_77]] : South)
// CHECK:           AIE.wire(%[[VAL_49]] : Core, %[[VAL_78:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_49]] : DMA, %[[VAL_78]] : DMA)
// CHECK:           AIE.wire(%[[VAL_77]] : North, %[[VAL_78]] : South)
// CHECK:           AIE.wire(%[[VAL_2]] : Core, %[[VAL_79:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_2]] : DMA, %[[VAL_79]] : DMA)
// CHECK:           AIE.wire(%[[VAL_78]] : North, %[[VAL_79]] : South)
// CHECK:           AIE.wire(%[[VAL_74]] : East, %[[VAL_80:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_76]] : East, %[[VAL_81:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_60]] : Core, %[[VAL_81]] : Core)
// CHECK:           AIE.wire(%[[VAL_60]] : DMA, %[[VAL_81]] : DMA)
// CHECK:           AIE.wire(%[[VAL_80]] : North, %[[VAL_81]] : South)
// CHECK:           AIE.wire(%[[VAL_77]] : East, %[[VAL_82:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_20]] : Core, %[[VAL_82]] : Core)
// CHECK:           AIE.wire(%[[VAL_20]] : DMA, %[[VAL_82]] : DMA)
// CHECK:           AIE.wire(%[[VAL_81]] : North, %[[VAL_82]] : South)
// CHECK:           AIE.wire(%[[VAL_78]] : East, %[[VAL_83:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_3]] : Core, %[[VAL_83]] : Core)
// CHECK:           AIE.wire(%[[VAL_3]] : DMA, %[[VAL_83]] : DMA)
// CHECK:           AIE.wire(%[[VAL_82]] : North, %[[VAL_83]] : South)
// CHECK:           AIE.wire(%[[VAL_79]] : East, %[[VAL_84:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_4]] : Core, %[[VAL_84]] : Core)
// CHECK:           AIE.wire(%[[VAL_4]] : DMA, %[[VAL_84]] : DMA)
// CHECK:           AIE.wire(%[[VAL_83]] : North, %[[VAL_84]] : South)
// CHECK:           AIE.wire(%[[VAL_80]] : East, %[[VAL_85:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_81]] : East, %[[VAL_86:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_52]] : Core, %[[VAL_86]] : Core)
// CHECK:           AIE.wire(%[[VAL_52]] : DMA, %[[VAL_86]] : DMA)
// CHECK:           AIE.wire(%[[VAL_85]] : North, %[[VAL_86]] : South)
// CHECK:           AIE.wire(%[[VAL_82]] : East, %[[VAL_87:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_22]] : Core, %[[VAL_87]] : Core)
// CHECK:           AIE.wire(%[[VAL_22]] : DMA, %[[VAL_87]] : DMA)
// CHECK:           AIE.wire(%[[VAL_86]] : North, %[[VAL_87]] : South)
// CHECK:           AIE.wire(%[[VAL_83]] : East, %[[VAL_88:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_24]] : Core, %[[VAL_88]] : Core)
// CHECK:           AIE.wire(%[[VAL_24]] : DMA, %[[VAL_88]] : DMA)
// CHECK:           AIE.wire(%[[VAL_87]] : North, %[[VAL_88]] : South)
// CHECK:           AIE.wire(%[[VAL_84]] : East, %[[VAL_89:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_5]] : Core, %[[VAL_89]] : Core)
// CHECK:           AIE.wire(%[[VAL_5]] : DMA, %[[VAL_89]] : DMA)
// CHECK:           AIE.wire(%[[VAL_88]] : North, %[[VAL_89]] : South)
// CHECK:           AIE.wire(%[[VAL_85]] : East, %[[VAL_90:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_91:.*]] : North, %[[VAL_90]] : South)
// CHECK:           AIE.wire(%[[VAL_6]] : DMA, %[[VAL_91]] : DMA)
// CHECK:           AIE.wire(%[[VAL_86]] : East, %[[VAL_92:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_37]] : Core, %[[VAL_92]] : Core)
// CHECK:           AIE.wire(%[[VAL_37]] : DMA, %[[VAL_92]] : DMA)
// CHECK:           AIE.wire(%[[VAL_90]] : North, %[[VAL_92]] : South)
// CHECK:           AIE.wire(%[[VAL_87]] : East, %[[VAL_93:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_39]] : Core, %[[VAL_93]] : Core)
// CHECK:           AIE.wire(%[[VAL_39]] : DMA, %[[VAL_93]] : DMA)
// CHECK:           AIE.wire(%[[VAL_92]] : North, %[[VAL_93]] : South)
// CHECK:           AIE.wire(%[[VAL_88]] : East, %[[VAL_94:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_7]] : Core, %[[VAL_94]] : Core)
// CHECK:           AIE.wire(%[[VAL_7]] : DMA, %[[VAL_94]] : DMA)
// CHECK:           AIE.wire(%[[VAL_93]] : North, %[[VAL_94]] : South)
// CHECK:           AIE.wire(%[[VAL_89]] : East, %[[VAL_95:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_47]] : Core, %[[VAL_95]] : Core)
// CHECK:           AIE.wire(%[[VAL_47]] : DMA, %[[VAL_95]] : DMA)
// CHECK:           AIE.wire(%[[VAL_94]] : North, %[[VAL_95]] : South)
// CHECK:           AIE.wire(%[[VAL_90]] : East, %[[VAL_96:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_97:.*]] : North, %[[VAL_96]] : South)
// CHECK:           AIE.wire(%[[VAL_8]] : DMA, %[[VAL_97]] : DMA)
// CHECK:           AIE.wire(%[[VAL_92]] : East, %[[VAL_98:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_62]] : Core, %[[VAL_98]] : Core)
// CHECK:           AIE.wire(%[[VAL_62]] : DMA, %[[VAL_98]] : DMA)
// CHECK:           AIE.wire(%[[VAL_96]] : North, %[[VAL_98]] : South)
// CHECK:           AIE.wire(%[[VAL_93]] : East, %[[VAL_99:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_9]] : Core, %[[VAL_99]] : Core)
// CHECK:           AIE.wire(%[[VAL_9]] : DMA, %[[VAL_99]] : DMA)
// CHECK:           AIE.wire(%[[VAL_98]] : North, %[[VAL_99]] : South)
// CHECK:           AIE.wire(%[[VAL_94]] : East, %[[VAL_100:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_27]] : Core, %[[VAL_100]] : Core)
// CHECK:           AIE.wire(%[[VAL_27]] : DMA, %[[VAL_100]] : DMA)
// CHECK:           AIE.wire(%[[VAL_99]] : North, %[[VAL_100]] : South)
// CHECK:           AIE.wire(%[[VAL_95]] : East, %[[VAL_101:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_56]] : Core, %[[VAL_101]] : Core)
// CHECK:           AIE.wire(%[[VAL_56]] : DMA, %[[VAL_101]] : DMA)
// CHECK:           AIE.wire(%[[VAL_100]] : North, %[[VAL_101]] : South)
// CHECK:           AIE.wire(%[[VAL_96]] : East, %[[VAL_102:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_98]] : East, %[[VAL_103:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_66]] : Core, %[[VAL_103]] : Core)
// CHECK:           AIE.wire(%[[VAL_66]] : DMA, %[[VAL_103]] : DMA)
// CHECK:           AIE.wire(%[[VAL_102]] : North, %[[VAL_103]] : South)
// CHECK:           AIE.wire(%[[VAL_99]] : East, %[[VAL_104:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_68]] : Core, %[[VAL_104]] : Core)
// CHECK:           AIE.wire(%[[VAL_68]] : DMA, %[[VAL_104]] : DMA)
// CHECK:           AIE.wire(%[[VAL_103]] : North, %[[VAL_104]] : South)
// CHECK:           AIE.wire(%[[VAL_100]] : East, %[[VAL_105:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_10]] : Core, %[[VAL_105]] : Core)
// CHECK:           AIE.wire(%[[VAL_10]] : DMA, %[[VAL_105]] : DMA)
// CHECK:           AIE.wire(%[[VAL_104]] : North, %[[VAL_105]] : South)
// CHECK:           AIE.wire(%[[VAL_101]] : East, %[[VAL_106:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_11]] : Core, %[[VAL_106]] : Core)
// CHECK:           AIE.wire(%[[VAL_11]] : DMA, %[[VAL_106]] : DMA)
// CHECK:           AIE.wire(%[[VAL_105]] : North, %[[VAL_106]] : South)
// CHECK:         }

module {
    AIE.device(xcvc1902) {
        %t20 = AIE.tile(2, 0)
        %t30 = AIE.tile(3, 0)
        %t34 = AIE.tile(3, 4)
        %t43 = AIE.tile(4, 3)
        %t44 = AIE.tile(4, 4)
        %t54 = AIE.tile(5, 4)
        %t60 = AIE.tile(6, 0)
        %t63 = AIE.tile(6, 3)
        %t70 = AIE.tile(7, 0)
        %t72 = AIE.tile(7, 2)
        %t83 = AIE.tile(8, 3)
        %t84 = AIE.tile(8, 4)

        AIE.flow(%t20, DMA : 0, %t63, DMA : 0)
        AIE.flow(%t20, DMA : 1, %t83, DMA : 0)
        AIE.flow(%t30, DMA : 0, %t72, DMA : 0)
        AIE.flow(%t30, DMA : 1, %t54, DMA : 0)

        AIE.flow(%t34, Core : 0, %t63, Core : 1)
        AIE.flow(%t34, DMA : 1, %t70, DMA : 0)
        AIE.flow(%t43, Core : 0, %t84, Core : 1)
        AIE.flow(%t43, DMA : 1, %t60, DMA : 1)

        AIE.flow(%t44, Core : 0, %t54, Core : 1)
        AIE.flow(%t44, DMA : 1, %t60, DMA : 0)
        AIE.flow(%t54, Core : 0, %t43, Core : 1)
        AIE.flow(%t54, DMA : 1, %t30, DMA : 1)

        AIE.flow(%t60, DMA : 0, %t44, DMA : 0)
        AIE.flow(%t60, DMA : 1, %t43, DMA : 0)
        AIE.flow(%t63, Core : 0, %t34, Core : 1)
        AIE.flow(%t63, DMA : 1, %t20, DMA : 1)

        AIE.flow(%t70, DMA : 0, %t34, DMA : 0)
        AIE.flow(%t70, DMA : 1, %t84, DMA : 0)
        AIE.flow(%t72, Core : 0, %t83, Core : 1)
        AIE.flow(%t72, DMA : 1, %t30, DMA : 0)

        AIE.flow(%t83, Core : 0, %t44, Core : 1)
        AIE.flow(%t83, DMA : 1, %t20, DMA : 0)
        AIE.flow(%t84, Core : 0, %t72, Core : 1)
        AIE.flow(%t84, DMA : 1, %t70, DMA : 1)
    }
}
