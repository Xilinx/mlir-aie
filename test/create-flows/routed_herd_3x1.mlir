//===- routed_herd_3x1.mlir ------------------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_0:.*]] = AIE.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(1, 0)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(3, 0)
// CHECK:           %[[VAL_4:.*]] = AIE.tile(4, 0)
// CHECK:           %[[VAL_5:.*]] = AIE.tile(5, 0)
// CHECK:           %[[VAL_6:.*]] = AIE.tile(6, 0)
// CHECK:           %[[VAL_7:.*]] = AIE.tile(7, 0)
// CHECK:           %[[VAL_8:.*]] = AIE.tile(8, 0)
// CHECK:           %[[VAL_9:.*]] = AIE.tile(9, 0)
// CHECK:           %[[VAL_10:.*]] = AIE.tile(10, 0)
// CHECK:           %[[VAL_11:.*]] = AIE.tile(11, 0)
// CHECK:           %[[VAL_12:.*]] = AIE.tile(18, 0)
// CHECK:           %[[VAL_13:.*]] = AIE.tile(19, 0)
// CHECK:           %[[VAL_14:.*]] = AIE.tile(0, 1)
// CHECK:           %[[VAL_15:.*]] = AIE.tile(0, 2)
// CHECK:           %[[VAL_16:.*]] = AIE.tile(0, 3)
// CHECK:           %[[VAL_17:.*]] = AIE.tile(0, 4)
// CHECK:           %[[VAL_18:.*]] = AIE.tile(1, 1)
// CHECK:           %[[VAL_19:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_20:.*]] = AIE.tile(1, 3)
// CHECK:           %[[VAL_21:.*]] = AIE.tile(1, 4)
// CHECK:           %[[VAL_22:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_23:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_24:.*]] = AIE.tile(2, 3)
// CHECK:           %[[VAL_25:.*]] = AIE.tile(2, 4)
// CHECK:           %[[VAL_26:.*]] = AIE.tile(3, 1)
// CHECK:           %[[VAL_27:.*]] = AIE.tile(3, 2)
// CHECK:           %[[VAL_28:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_29:.*]] = AIE.tile(3, 4)
// CHECK:           %[[VAL_30:.*]] = AIE.tile(4, 1)
// CHECK:           %[[VAL_31:.*]] = AIE.tile(4, 2)
// CHECK:           %[[VAL_32:.*]] = AIE.tile(4, 3)
// CHECK:           %[[VAL_33:.*]] = AIE.tile(4, 4)
// CHECK:           %[[VAL_34:.*]] = AIE.tile(5, 1)
// CHECK:           %[[VAL_35:.*]] = AIE.tile(5, 2)
// CHECK:           %[[VAL_36:.*]] = AIE.tile(5, 3)
// CHECK:           %[[VAL_37:.*]] = AIE.tile(5, 4)
// CHECK:           %[[VAL_38:.*]] = AIE.tile(6, 1)
// CHECK:           %[[VAL_39:.*]] = AIE.tile(6, 2)
// CHECK:           %[[VAL_40:.*]] = AIE.tile(6, 3)
// CHECK:           %[[VAL_41:.*]] = AIE.tile(6, 4)
// CHECK:           %[[VAL_42:.*]] = AIE.tile(7, 1)
// CHECK:           %[[VAL_43:.*]] = AIE.tile(7, 2)
// CHECK:           %[[VAL_44:.*]] = AIE.tile(7, 3)
// CHECK:           %[[VAL_45:.*]] = AIE.tile(7, 4)
// CHECK:           %[[VAL_46:.*]] = AIE.tile(8, 1)
// CHECK:           %[[VAL_47:.*]] = AIE.tile(8, 2)
// CHECK:           %[[VAL_48:.*]] = AIE.tile(8, 3)
// CHECK:           %[[VAL_49:.*]] = AIE.tile(8, 4)
// CHECK:           %[[VAL_50:.*]] = AIE.tile(9, 1)
// CHECK:           %[[VAL_51:.*]] = AIE.tile(9, 2)
// CHECK:           %[[VAL_52:.*]] = AIE.tile(9, 3)
// CHECK:           %[[VAL_53:.*]] = AIE.tile(9, 4)
// CHECK:           %[[VAL_54:.*]] = AIE.tile(10, 1)
// CHECK:           %[[VAL_55:.*]] = AIE.tile(10, 2)
// CHECK:           %[[VAL_56:.*]] = AIE.tile(10, 3)
// CHECK:           %[[VAL_57:.*]] = AIE.tile(10, 4)
// CHECK:           %[[VAL_58:.*]] = AIE.tile(11, 1)
// CHECK:           %[[VAL_59:.*]] = AIE.tile(11, 2)
// CHECK:           %[[VAL_60:.*]] = AIE.tile(11, 3)
// CHECK:           %[[VAL_61:.*]] = AIE.tile(11, 4)
// CHECK:           %[[VAL_62:.*]] = AIE.tile(12, 1)
// CHECK:           %[[VAL_63:.*]] = AIE.tile(12, 2)
// CHECK:           %[[VAL_64:.*]] = AIE.tile(12, 3)
// CHECK:           %[[VAL_65:.*]] = AIE.tile(12, 4)
// CHECK:           %[[VAL_66:.*]] = AIE.switchbox(%[[VAL_14]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_67:.*]] = AIE.switchbox(%[[VAL_15]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_68:.*]] = AIE.switchbox(%[[VAL_16]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<East : 0, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_69:.*]] = AIE.switchbox(%[[VAL_17]]) {
// CHECK:           }
// CHECK:           %[[VAL_70:.*]] = AIE.switchbox(%[[VAL_18]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_71:.*]] = AIE.switchbox(%[[VAL_19]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_72:.*]] = AIE.switchbox(%[[VAL_20]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_73:.*]] = AIE.switchbox(%[[VAL_21]]) {
// CHECK:             AIE.connect<East : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_74:.*]] = AIE.switchbox(%[[VAL_22]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_75:.*]] = AIE.switchbox(%[[VAL_23]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_76:.*]] = AIE.switchbox(%[[VAL_24]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_77:.*]] = AIE.switchbox(%[[VAL_25]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_78:.*]] = AIE.switchbox(%[[VAL_26]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_79:.*]] = AIE.switchbox(%[[VAL_27]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_80:.*]] = AIE.switchbox(%[[VAL_28]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_81:.*]] = AIE.switchbox(%[[VAL_29]]) {
// CHECK:           }
// CHECK:           %[[VAL_82:.*]] = AIE.switchbox(%[[VAL_30]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_83:.*]] = AIE.switchbox(%[[VAL_31]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_84:.*]] = AIE.switchbox(%[[VAL_32]]) {
// CHECK:           }
// CHECK:           %[[VAL_85:.*]] = AIE.switchbox(%[[VAL_33]]) {
// CHECK:           }
// CHECK:           %[[VAL_86:.*]] = AIE.switchbox(%[[VAL_34]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_87:.*]] = AIE.switchbox(%[[VAL_35]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_88:.*]] = AIE.switchbox(%[[VAL_36]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_89:.*]] = AIE.switchbox(%[[VAL_37]]) {
// CHECK:           }
// CHECK:           %[[VAL_90:.*]] = AIE.switchbox(%[[VAL_38]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_91:.*]] = AIE.switchbox(%[[VAL_39]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_92:.*]] = AIE.switchbox(%[[VAL_40]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<South : 1, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_93:.*]] = AIE.switchbox(%[[VAL_41]]) {
// CHECK:           }
// CHECK:           %[[VAL_94:.*]] = AIE.switchbox(%[[VAL_42]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_95:.*]] = AIE.switchbox(%[[VAL_43]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_96:.*]] = AIE.switchbox(%[[VAL_44]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_97:.*]] = AIE.switchbox(%[[VAL_45]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<South : 1, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_98:.*]] = AIE.switchbox(%[[VAL_46]]) {
// CHECK:           }
// CHECK:           %[[VAL_99:.*]] = AIE.switchbox(%[[VAL_47]]) {
// CHECK:           }
// CHECK:           %[[VAL_100:.*]] = AIE.switchbox(%[[VAL_48]]) {
// CHECK:           }
// CHECK:           %[[VAL_101:.*]] = AIE.switchbox(%[[VAL_49]]) {
// CHECK:           }
// CHECK:           %[[VAL_102:.*]] = AIE.switchbox(%[[VAL_50]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_103:.*]] = AIE.switchbox(%[[VAL_51]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_104:.*]] = AIE.switchbox(%[[VAL_52]]) {
// CHECK:           }
// CHECK:           %[[VAL_105:.*]] = AIE.switchbox(%[[VAL_53]]) {
// CHECK:           }
// CHECK:           %[[VAL_106:.*]] = AIE.switchbox(%[[VAL_54]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_107:.*]] = AIE.switchbox(%[[VAL_55]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_108:.*]] = AIE.switchbox(%[[VAL_56]]) {
// CHECK:           }
// CHECK:           %[[VAL_109:.*]] = AIE.switchbox(%[[VAL_57]]) {
// CHECK:           }
// CHECK:           %[[VAL_110:.*]] = AIE.switchbox(%[[VAL_58]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_111:.*]] = AIE.switchbox(%[[VAL_59]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_112:.*]] = AIE.switchbox(%[[VAL_60]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<South : 1, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_113:.*]] = AIE.switchbox(%[[VAL_61]]) {
// CHECK:           }
// CHECK:           %[[VAL_114:.*]] = AIE.switchbox(%[[VAL_2]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, East : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_115:.*]] = AIE.shimmux(%[[VAL_2]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_116:.*]] = AIE.switchbox(%[[VAL_3]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, East : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_117:.*]] = AIE.switchbox(%[[VAL_4]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, North : 0>
// CHECK:             AIE.connect<East : 2, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_118:.*]] = AIE.switchbox(%[[VAL_5]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_119:.*]] = AIE.switchbox(%[[VAL_6]]) {
// CHECK:             AIE.connect<West : 0, North : 1>
// CHECK:             AIE.connect<West : 1, East : 0>
// CHECK:             AIE.connect<South : 3, West : 0>
// CHECK:             AIE.connect<South : 7, West : 1>
// CHECK:             AIE.connect<East : 0, West : 2>
// CHECK:             AIE.connect<East : 1, West : 3>
// CHECK:             AIE.connect<East : 2, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_120:.*]] = AIE.shimmux(%[[VAL_3]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_121:.*]] = AIE.switchbox(%[[VAL_7]]) {
// CHECK:             AIE.connect<West : 0, North : 1>
// CHECK:             AIE.connect<South : 3, West : 0>
// CHECK:             AIE.connect<South : 7, West : 1>
// CHECK:             AIE.connect<East : 0, West : 2>
// CHECK:             AIE.connect<East : 1, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_122:.*]] = AIE.switchbox(%[[VAL_0]]) {
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_123:.*]] = AIE.switchbox(%[[VAL_1]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_124:.*]] = AIE.shimmux(%[[VAL_6]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_125:.*]] = AIE.shimmux(%[[VAL_7]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_126:.*]] = AIE.switchbox(%[[VAL_10]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:           }
// CHECK:           %[[VAL_127:.*]] = AIE.shimmux(%[[VAL_10]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_128:.*]] = AIE.switchbox(%[[VAL_11]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_129:.*]] = AIE.shimmux(%[[VAL_11]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_130:.*]] = AIE.switchbox(%[[VAL_8]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_131:.*]] = AIE.switchbox(%[[VAL_9]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, North : 0>
// CHECK:             AIE.connect<East : 2, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_132:.*]] = AIE.tile(12, 0)
// CHECK:           %[[VAL_133:.*]] = AIE.switchbox(%[[VAL_132]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_134:.*]] = AIE.tile(13, 0)
// CHECK:           %[[VAL_135:.*]] = AIE.switchbox(%[[VAL_134]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_136:.*]] = AIE.tile(14, 0)
// CHECK:           %[[VAL_137:.*]] = AIE.switchbox(%[[VAL_136]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_138:.*]] = AIE.tile(15, 0)
// CHECK:           %[[VAL_139:.*]] = AIE.switchbox(%[[VAL_138]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_140:.*]] = AIE.tile(16, 0)
// CHECK:           %[[VAL_141:.*]] = AIE.switchbox(%[[VAL_140]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_142:.*]] = AIE.tile(17, 0)
// CHECK:           %[[VAL_143:.*]] = AIE.switchbox(%[[VAL_142]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_144:.*]] = AIE.switchbox(%[[VAL_12]]) {
// CHECK:             AIE.connect<South : 3, West : 0>
// CHECK:             AIE.connect<South : 7, West : 1>
// CHECK:             AIE.connect<East : 0, West : 2>
// CHECK:             AIE.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_145:.*]] = AIE.shimmux(%[[VAL_12]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_146:.*]] = AIE.switchbox(%[[VAL_13]]) {
// CHECK:             AIE.connect<South : 3, West : 0>
// CHECK:             AIE.connect<South : 7, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_147:.*]] = AIE.shimmux(%[[VAL_13]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_14]] : Core, %[[VAL_66]] : Core)
// CHECK:           AIE.wire(%[[VAL_14]] : DMA, %[[VAL_66]] : DMA)
// CHECK:           AIE.wire(%[[VAL_122]] : North, %[[VAL_66]] : South)
// CHECK:           AIE.wire(%[[VAL_15]] : Core, %[[VAL_67]] : Core)
// CHECK:           AIE.wire(%[[VAL_15]] : DMA, %[[VAL_67]] : DMA)
// CHECK:           AIE.wire(%[[VAL_66]] : North, %[[VAL_67]] : South)
// CHECK:           AIE.wire(%[[VAL_16]] : Core, %[[VAL_68]] : Core)
// CHECK:           AIE.wire(%[[VAL_16]] : DMA, %[[VAL_68]] : DMA)
// CHECK:           AIE.wire(%[[VAL_67]] : North, %[[VAL_68]] : South)
// CHECK:           AIE.wire(%[[VAL_17]] : Core, %[[VAL_69]] : Core)
// CHECK:           AIE.wire(%[[VAL_17]] : DMA, %[[VAL_69]] : DMA)
// CHECK:           AIE.wire(%[[VAL_68]] : North, %[[VAL_69]] : South)
// CHECK:           AIE.wire(%[[VAL_122]] : East, %[[VAL_123]] : West)
// CHECK:           AIE.wire(%[[VAL_66]] : East, %[[VAL_70]] : West)
// CHECK:           AIE.wire(%[[VAL_18]] : Core, %[[VAL_70]] : Core)
// CHECK:           AIE.wire(%[[VAL_18]] : DMA, %[[VAL_70]] : DMA)
// CHECK:           AIE.wire(%[[VAL_123]] : North, %[[VAL_70]] : South)
// CHECK:           AIE.wire(%[[VAL_67]] : East, %[[VAL_71]] : West)
// CHECK:           AIE.wire(%[[VAL_19]] : Core, %[[VAL_71]] : Core)
// CHECK:           AIE.wire(%[[VAL_19]] : DMA, %[[VAL_71]] : DMA)
// CHECK:           AIE.wire(%[[VAL_70]] : North, %[[VAL_71]] : South)
// CHECK:           AIE.wire(%[[VAL_68]] : East, %[[VAL_72]] : West)
// CHECK:           AIE.wire(%[[VAL_20]] : Core, %[[VAL_72]] : Core)
// CHECK:           AIE.wire(%[[VAL_20]] : DMA, %[[VAL_72]] : DMA)
// CHECK:           AIE.wire(%[[VAL_71]] : North, %[[VAL_72]] : South)
// CHECK:           AIE.wire(%[[VAL_69]] : East, %[[VAL_73]] : West)
// CHECK:           AIE.wire(%[[VAL_21]] : Core, %[[VAL_73]] : Core)
// CHECK:           AIE.wire(%[[VAL_21]] : DMA, %[[VAL_73]] : DMA)
// CHECK:           AIE.wire(%[[VAL_72]] : North, %[[VAL_73]] : South)
// CHECK:           AIE.wire(%[[VAL_123]] : East, %[[VAL_114]] : West)
// CHECK:           AIE.wire(%[[VAL_115]] : North, %[[VAL_114]] : South)
// CHECK:           AIE.wire(%[[VAL_2]] : DMA, %[[VAL_115]] : DMA)
// CHECK:           AIE.wire(%[[VAL_70]] : East, %[[VAL_74]] : West)
// CHECK:           AIE.wire(%[[VAL_22]] : Core, %[[VAL_74]] : Core)
// CHECK:           AIE.wire(%[[VAL_22]] : DMA, %[[VAL_74]] : DMA)
// CHECK:           AIE.wire(%[[VAL_114]] : North, %[[VAL_74]] : South)
// CHECK:           AIE.wire(%[[VAL_71]] : East, %[[VAL_75]] : West)
// CHECK:           AIE.wire(%[[VAL_23]] : Core, %[[VAL_75]] : Core)
// CHECK:           AIE.wire(%[[VAL_23]] : DMA, %[[VAL_75]] : DMA)
// CHECK:           AIE.wire(%[[VAL_74]] : North, %[[VAL_75]] : South)
// CHECK:           AIE.wire(%[[VAL_72]] : East, %[[VAL_76]] : West)
// CHECK:           AIE.wire(%[[VAL_24]] : Core, %[[VAL_76]] : Core)
// CHECK:           AIE.wire(%[[VAL_24]] : DMA, %[[VAL_76]] : DMA)
// CHECK:           AIE.wire(%[[VAL_75]] : North, %[[VAL_76]] : South)
// CHECK:           AIE.wire(%[[VAL_73]] : East, %[[VAL_77]] : West)
// CHECK:           AIE.wire(%[[VAL_25]] : Core, %[[VAL_77]] : Core)
// CHECK:           AIE.wire(%[[VAL_25]] : DMA, %[[VAL_77]] : DMA)
// CHECK:           AIE.wire(%[[VAL_76]] : North, %[[VAL_77]] : South)
// CHECK:           AIE.wire(%[[VAL_114]] : East, %[[VAL_116]] : West)
// CHECK:           AIE.wire(%[[VAL_120]] : North, %[[VAL_116]] : South)
// CHECK:           AIE.wire(%[[VAL_3]] : DMA, %[[VAL_120]] : DMA)
// CHECK:           AIE.wire(%[[VAL_74]] : East, %[[VAL_78]] : West)
// CHECK:           AIE.wire(%[[VAL_26]] : Core, %[[VAL_78]] : Core)
// CHECK:           AIE.wire(%[[VAL_26]] : DMA, %[[VAL_78]] : DMA)
// CHECK:           AIE.wire(%[[VAL_116]] : North, %[[VAL_78]] : South)
// CHECK:           AIE.wire(%[[VAL_75]] : East, %[[VAL_79]] : West)
// CHECK:           AIE.wire(%[[VAL_27]] : Core, %[[VAL_79]] : Core)
// CHECK:           AIE.wire(%[[VAL_27]] : DMA, %[[VAL_79]] : DMA)
// CHECK:           AIE.wire(%[[VAL_78]] : North, %[[VAL_79]] : South)
// CHECK:           AIE.wire(%[[VAL_76]] : East, %[[VAL_80]] : West)
// CHECK:           AIE.wire(%[[VAL_28]] : Core, %[[VAL_80]] : Core)
// CHECK:           AIE.wire(%[[VAL_28]] : DMA, %[[VAL_80]] : DMA)
// CHECK:           AIE.wire(%[[VAL_79]] : North, %[[VAL_80]] : South)
// CHECK:           AIE.wire(%[[VAL_77]] : East, %[[VAL_81]] : West)
// CHECK:           AIE.wire(%[[VAL_29]] : Core, %[[VAL_81]] : Core)
// CHECK:           AIE.wire(%[[VAL_29]] : DMA, %[[VAL_81]] : DMA)
// CHECK:           AIE.wire(%[[VAL_80]] : North, %[[VAL_81]] : South)
// CHECK:           AIE.wire(%[[VAL_116]] : East, %[[VAL_117]] : West)
// CHECK:           AIE.wire(%[[VAL_78]] : East, %[[VAL_82]] : West)
// CHECK:           AIE.wire(%[[VAL_30]] : Core, %[[VAL_82]] : Core)
// CHECK:           AIE.wire(%[[VAL_30]] : DMA, %[[VAL_82]] : DMA)
// CHECK:           AIE.wire(%[[VAL_117]] : North, %[[VAL_82]] : South)
// CHECK:           AIE.wire(%[[VAL_79]] : East, %[[VAL_83]] : West)
// CHECK:           AIE.wire(%[[VAL_31]] : Core, %[[VAL_83]] : Core)
// CHECK:           AIE.wire(%[[VAL_31]] : DMA, %[[VAL_83]] : DMA)
// CHECK:           AIE.wire(%[[VAL_82]] : North, %[[VAL_83]] : South)
// CHECK:           AIE.wire(%[[VAL_80]] : East, %[[VAL_84]] : West)
// CHECK:           AIE.wire(%[[VAL_32]] : Core, %[[VAL_84]] : Core)
// CHECK:           AIE.wire(%[[VAL_32]] : DMA, %[[VAL_84]] : DMA)
// CHECK:           AIE.wire(%[[VAL_83]] : North, %[[VAL_84]] : South)
// CHECK:           AIE.wire(%[[VAL_81]] : East, %[[VAL_85]] : West)
// CHECK:           AIE.wire(%[[VAL_33]] : Core, %[[VAL_85]] : Core)
// CHECK:           AIE.wire(%[[VAL_33]] : DMA, %[[VAL_85]] : DMA)
// CHECK:           AIE.wire(%[[VAL_84]] : North, %[[VAL_85]] : South)
// CHECK:           AIE.wire(%[[VAL_117]] : East, %[[VAL_118]] : West)
// CHECK:           AIE.wire(%[[VAL_82]] : East, %[[VAL_86]] : West)
// CHECK:           AIE.wire(%[[VAL_34]] : Core, %[[VAL_86]] : Core)
// CHECK:           AIE.wire(%[[VAL_34]] : DMA, %[[VAL_86]] : DMA)
// CHECK:           AIE.wire(%[[VAL_118]] : North, %[[VAL_86]] : South)
// CHECK:           AIE.wire(%[[VAL_83]] : East, %[[VAL_87]] : West)
// CHECK:           AIE.wire(%[[VAL_35]] : Core, %[[VAL_87]] : Core)
// CHECK:           AIE.wire(%[[VAL_35]] : DMA, %[[VAL_87]] : DMA)
// CHECK:           AIE.wire(%[[VAL_86]] : North, %[[VAL_87]] : South)
// CHECK:           AIE.wire(%[[VAL_84]] : East, %[[VAL_88]] : West)
// CHECK:           AIE.wire(%[[VAL_36]] : Core, %[[VAL_88]] : Core)
// CHECK:           AIE.wire(%[[VAL_36]] : DMA, %[[VAL_88]] : DMA)
// CHECK:           AIE.wire(%[[VAL_87]] : North, %[[VAL_88]] : South)
// CHECK:           AIE.wire(%[[VAL_85]] : East, %[[VAL_89]] : West)
// CHECK:           AIE.wire(%[[VAL_37]] : Core, %[[VAL_89]] : Core)
// CHECK:           AIE.wire(%[[VAL_37]] : DMA, %[[VAL_89]] : DMA)
// CHECK:           AIE.wire(%[[VAL_88]] : North, %[[VAL_89]] : South)
// CHECK:           AIE.wire(%[[VAL_118]] : East, %[[VAL_119]] : West)
// CHECK:           AIE.wire(%[[VAL_124]] : North, %[[VAL_119]] : South)
// CHECK:           AIE.wire(%[[VAL_6]] : DMA, %[[VAL_124]] : DMA)
// CHECK:           AIE.wire(%[[VAL_86]] : East, %[[VAL_90]] : West)
// CHECK:           AIE.wire(%[[VAL_38]] : Core, %[[VAL_90]] : Core)
// CHECK:           AIE.wire(%[[VAL_38]] : DMA, %[[VAL_90]] : DMA)
// CHECK:           AIE.wire(%[[VAL_119]] : North, %[[VAL_90]] : South)
// CHECK:           AIE.wire(%[[VAL_87]] : East, %[[VAL_91]] : West)
// CHECK:           AIE.wire(%[[VAL_39]] : Core, %[[VAL_91]] : Core)
// CHECK:           AIE.wire(%[[VAL_39]] : DMA, %[[VAL_91]] : DMA)
// CHECK:           AIE.wire(%[[VAL_90]] : North, %[[VAL_91]] : South)
// CHECK:           AIE.wire(%[[VAL_88]] : East, %[[VAL_92]] : West)
// CHECK:           AIE.wire(%[[VAL_40]] : Core, %[[VAL_92]] : Core)
// CHECK:           AIE.wire(%[[VAL_40]] : DMA, %[[VAL_92]] : DMA)
// CHECK:           AIE.wire(%[[VAL_91]] : North, %[[VAL_92]] : South)
// CHECK:           AIE.wire(%[[VAL_89]] : East, %[[VAL_93]] : West)
// CHECK:           AIE.wire(%[[VAL_41]] : Core, %[[VAL_93]] : Core)
// CHECK:           AIE.wire(%[[VAL_41]] : DMA, %[[VAL_93]] : DMA)
// CHECK:           AIE.wire(%[[VAL_92]] : North, %[[VAL_93]] : South)
// CHECK:           AIE.wire(%[[VAL_119]] : East, %[[VAL_121]] : West)
// CHECK:           AIE.wire(%[[VAL_125]] : North, %[[VAL_121]] : South)
// CHECK:           AIE.wire(%[[VAL_7]] : DMA, %[[VAL_125]] : DMA)
// CHECK:           AIE.wire(%[[VAL_90]] : East, %[[VAL_94]] : West)
// CHECK:           AIE.wire(%[[VAL_42]] : Core, %[[VAL_94]] : Core)
// CHECK:           AIE.wire(%[[VAL_42]] : DMA, %[[VAL_94]] : DMA)
// CHECK:           AIE.wire(%[[VAL_121]] : North, %[[VAL_94]] : South)
// CHECK:           AIE.wire(%[[VAL_91]] : East, %[[VAL_95]] : West)
// CHECK:           AIE.wire(%[[VAL_43]] : Core, %[[VAL_95]] : Core)
// CHECK:           AIE.wire(%[[VAL_43]] : DMA, %[[VAL_95]] : DMA)
// CHECK:           AIE.wire(%[[VAL_94]] : North, %[[VAL_95]] : South)
// CHECK:           AIE.wire(%[[VAL_92]] : East, %[[VAL_96]] : West)
// CHECK:           AIE.wire(%[[VAL_44]] : Core, %[[VAL_96]] : Core)
// CHECK:           AIE.wire(%[[VAL_44]] : DMA, %[[VAL_96]] : DMA)
// CHECK:           AIE.wire(%[[VAL_95]] : North, %[[VAL_96]] : South)
// CHECK:           AIE.wire(%[[VAL_93]] : East, %[[VAL_97]] : West)
// CHECK:           AIE.wire(%[[VAL_45]] : Core, %[[VAL_97]] : Core)
// CHECK:           AIE.wire(%[[VAL_45]] : DMA, %[[VAL_97]] : DMA)
// CHECK:           AIE.wire(%[[VAL_96]] : North, %[[VAL_97]] : South)
// CHECK:           AIE.wire(%[[VAL_121]] : East, %[[VAL_130]] : West)
// CHECK:           AIE.wire(%[[VAL_94]] : East, %[[VAL_98]] : West)
// CHECK:           AIE.wire(%[[VAL_46]] : Core, %[[VAL_98]] : Core)
// CHECK:           AIE.wire(%[[VAL_46]] : DMA, %[[VAL_98]] : DMA)
// CHECK:           AIE.wire(%[[VAL_130]] : North, %[[VAL_98]] : South)
// CHECK:           AIE.wire(%[[VAL_95]] : East, %[[VAL_99]] : West)
// CHECK:           AIE.wire(%[[VAL_47]] : Core, %[[VAL_99]] : Core)
// CHECK:           AIE.wire(%[[VAL_47]] : DMA, %[[VAL_99]] : DMA)
// CHECK:           AIE.wire(%[[VAL_98]] : North, %[[VAL_99]] : South)
// CHECK:           AIE.wire(%[[VAL_96]] : East, %[[VAL_100]] : West)
// CHECK:           AIE.wire(%[[VAL_48]] : Core, %[[VAL_100]] : Core)
// CHECK:           AIE.wire(%[[VAL_48]] : DMA, %[[VAL_100]] : DMA)
// CHECK:           AIE.wire(%[[VAL_99]] : North, %[[VAL_100]] : South)
// CHECK:           AIE.wire(%[[VAL_97]] : East, %[[VAL_101]] : West)
// CHECK:           AIE.wire(%[[VAL_49]] : Core, %[[VAL_101]] : Core)
// CHECK:           AIE.wire(%[[VAL_49]] : DMA, %[[VAL_101]] : DMA)
// CHECK:           AIE.wire(%[[VAL_100]] : North, %[[VAL_101]] : South)
// CHECK:           AIE.wire(%[[VAL_130]] : East, %[[VAL_131]] : West)
// CHECK:           AIE.wire(%[[VAL_98]] : East, %[[VAL_102]] : West)
// CHECK:           AIE.wire(%[[VAL_50]] : Core, %[[VAL_102]] : Core)
// CHECK:           AIE.wire(%[[VAL_50]] : DMA, %[[VAL_102]] : DMA)
// CHECK:           AIE.wire(%[[VAL_131]] : North, %[[VAL_102]] : South)
// CHECK:           AIE.wire(%[[VAL_99]] : East, %[[VAL_103]] : West)
// CHECK:           AIE.wire(%[[VAL_51]] : Core, %[[VAL_103]] : Core)
// CHECK:           AIE.wire(%[[VAL_51]] : DMA, %[[VAL_103]] : DMA)
// CHECK:           AIE.wire(%[[VAL_102]] : North, %[[VAL_103]] : South)
// CHECK:           AIE.wire(%[[VAL_100]] : East, %[[VAL_104]] : West)
// CHECK:           AIE.wire(%[[VAL_52]] : Core, %[[VAL_104]] : Core)
// CHECK:           AIE.wire(%[[VAL_52]] : DMA, %[[VAL_104]] : DMA)
// CHECK:           AIE.wire(%[[VAL_103]] : North, %[[VAL_104]] : South)
// CHECK:           AIE.wire(%[[VAL_101]] : East, %[[VAL_105]] : West)
// CHECK:           AIE.wire(%[[VAL_53]] : Core, %[[VAL_105]] : Core)
// CHECK:           AIE.wire(%[[VAL_53]] : DMA, %[[VAL_105]] : DMA)
// CHECK:           AIE.wire(%[[VAL_104]] : North, %[[VAL_105]] : South)
// CHECK:           AIE.wire(%[[VAL_131]] : East, %[[VAL_126]] : West)
// CHECK:           AIE.wire(%[[VAL_127]] : North, %[[VAL_126]] : South)
// CHECK:           AIE.wire(%[[VAL_10]] : DMA, %[[VAL_127]] : DMA)
// CHECK:           AIE.wire(%[[VAL_102]] : East, %[[VAL_106]] : West)
// CHECK:           AIE.wire(%[[VAL_54]] : Core, %[[VAL_106]] : Core)
// CHECK:           AIE.wire(%[[VAL_54]] : DMA, %[[VAL_106]] : DMA)
// CHECK:           AIE.wire(%[[VAL_126]] : North, %[[VAL_106]] : South)
// CHECK:           AIE.wire(%[[VAL_103]] : East, %[[VAL_107]] : West)
// CHECK:           AIE.wire(%[[VAL_55]] : Core, %[[VAL_107]] : Core)
// CHECK:           AIE.wire(%[[VAL_55]] : DMA, %[[VAL_107]] : DMA)
// CHECK:           AIE.wire(%[[VAL_106]] : North, %[[VAL_107]] : South)
// CHECK:           AIE.wire(%[[VAL_104]] : East, %[[VAL_108]] : West)
// CHECK:           AIE.wire(%[[VAL_56]] : Core, %[[VAL_108]] : Core)
// CHECK:           AIE.wire(%[[VAL_56]] : DMA, %[[VAL_108]] : DMA)
// CHECK:           AIE.wire(%[[VAL_107]] : North, %[[VAL_108]] : South)
// CHECK:           AIE.wire(%[[VAL_105]] : East, %[[VAL_109]] : West)
// CHECK:           AIE.wire(%[[VAL_57]] : Core, %[[VAL_109]] : Core)
// CHECK:           AIE.wire(%[[VAL_57]] : DMA, %[[VAL_109]] : DMA)
// CHECK:           AIE.wire(%[[VAL_108]] : North, %[[VAL_109]] : South)
// CHECK:           AIE.wire(%[[VAL_126]] : East, %[[VAL_128]] : West)
// CHECK:           AIE.wire(%[[VAL_129]] : North, %[[VAL_128]] : South)
// CHECK:           AIE.wire(%[[VAL_11]] : DMA, %[[VAL_129]] : DMA)
// CHECK:           AIE.wire(%[[VAL_106]] : East, %[[VAL_110]] : West)
// CHECK:           AIE.wire(%[[VAL_58]] : Core, %[[VAL_110]] : Core)
// CHECK:           AIE.wire(%[[VAL_58]] : DMA, %[[VAL_110]] : DMA)
// CHECK:           AIE.wire(%[[VAL_128]] : North, %[[VAL_110]] : South)
// CHECK:           AIE.wire(%[[VAL_107]] : East, %[[VAL_111]] : West)
// CHECK:           AIE.wire(%[[VAL_59]] : Core, %[[VAL_111]] : Core)
// CHECK:           AIE.wire(%[[VAL_59]] : DMA, %[[VAL_111]] : DMA)
// CHECK:           AIE.wire(%[[VAL_110]] : North, %[[VAL_111]] : South)
// CHECK:           AIE.wire(%[[VAL_108]] : East, %[[VAL_112]] : West)
// CHECK:           AIE.wire(%[[VAL_60]] : Core, %[[VAL_112]] : Core)
// CHECK:           AIE.wire(%[[VAL_60]] : DMA, %[[VAL_112]] : DMA)
// CHECK:           AIE.wire(%[[VAL_111]] : North, %[[VAL_112]] : South)
// CHECK:           AIE.wire(%[[VAL_109]] : East, %[[VAL_113]] : West)
// CHECK:           AIE.wire(%[[VAL_61]] : Core, %[[VAL_113]] : Core)
// CHECK:           AIE.wire(%[[VAL_61]] : DMA, %[[VAL_113]] : DMA)
// CHECK:           AIE.wire(%[[VAL_112]] : North, %[[VAL_113]] : South)
// CHECK:           AIE.wire(%[[VAL_128]] : East, %[[VAL_133]] : West)
// CHECK:           AIE.wire(%[[VAL_133]] : East, %[[VAL_135]] : West)
// CHECK:           AIE.wire(%[[VAL_135]] : East, %[[VAL_137]] : West)
// CHECK:           AIE.wire(%[[VAL_137]] : East, %[[VAL_139]] : West)
// CHECK:           AIE.wire(%[[VAL_139]] : East, %[[VAL_141]] : West)
// CHECK:           AIE.wire(%[[VAL_141]] : East, %[[VAL_143]] : West)
// CHECK:           AIE.wire(%[[VAL_143]] : East, %[[VAL_144]] : West)
// CHECK:           AIE.wire(%[[VAL_145]] : North, %[[VAL_144]] : South)
// CHECK:           AIE.wire(%[[VAL_12]] : DMA, %[[VAL_145]] : DMA)
// CHECK:           AIE.wire(%[[VAL_144]] : East, %[[VAL_146]] : West)
// CHECK:           AIE.wire(%[[VAL_147]] : North, %[[VAL_146]] : South)
// CHECK:           AIE.wire(%[[VAL_13]] : DMA, %[[VAL_147]] : DMA)
// CHECK:         }

//

module {
	AIE.device(xcvc1902) {
		%t00 = AIE.tile(0, 0)
		%t10 = AIE.tile(1, 0)
		%t20 = AIE.tile(2, 0)
		%t30 = AIE.tile(3, 0)
		%t40 = AIE.tile(4, 0)
		%t50 = AIE.tile(5, 0)
		%t60 = AIE.tile(6, 0)
		%t70 = AIE.tile(7, 0)
		%t80 = AIE.tile(8, 0)
		%t90 = AIE.tile(9, 0)
		%t100 = AIE.tile(10, 0)
		%t110 = AIE.tile(11, 0)
		%t180 = AIE.tile(18, 0)
		%t190 = AIE.tile(19, 0)

		%t01 = AIE.tile(0, 1)
		%t02 = AIE.tile(0, 2)
		%t03 = AIE.tile(0, 3)
		%t04 = AIE.tile(0, 4)
		%t11 = AIE.tile(1, 1)
		%t12 = AIE.tile(1, 2)
		%t13 = AIE.tile(1, 3)
		%t14 = AIE.tile(1, 4)
		%t21 = AIE.tile(2, 1)
		%t22 = AIE.tile(2, 2)
		%t23 = AIE.tile(2, 3)
		%t24 = AIE.tile(2, 4)
		%t31 = AIE.tile(3, 1)
		%t32 = AIE.tile(3, 2)
		%t33 = AIE.tile(3, 3)
		%t34 = AIE.tile(3, 4)
		%t41 = AIE.tile(4, 1)
		%t42 = AIE.tile(4, 2)
		%t43 = AIE.tile(4, 3)
		%t44 = AIE.tile(4, 4)
		%t51 = AIE.tile(5, 1)
		%t52 = AIE.tile(5, 2)
		%t53 = AIE.tile(5, 3)
		%t54 = AIE.tile(5, 4)
		%t61 = AIE.tile(6, 1)
		%t62 = AIE.tile(6, 2)
		%t63 = AIE.tile(6, 3)
		%t64 = AIE.tile(6, 4)
		%t71 = AIE.tile(7, 1)
		%t72 = AIE.tile(7, 2)
		%t73 = AIE.tile(7, 3)
		%t74 = AIE.tile(7, 4)
		%t81 = AIE.tile(8, 1)
		%t82 = AIE.tile(8, 2)
		%t83 = AIE.tile(8, 3)
		%t84 = AIE.tile(8, 4)
		%t91 = AIE.tile(9, 1)
		%t92 = AIE.tile(9, 2)
		%t93 = AIE.tile(9, 3)
		%t94 = AIE.tile(9, 4)
		%t101 = AIE.tile(10, 1)
		%t102 = AIE.tile(10, 2)
		%t103 = AIE.tile(10, 3)
		%t104 = AIE.tile(10, 4)
		%t111 = AIE.tile(11, 1)
		%t112 = AIE.tile(11, 2)
		%t113 = AIE.tile(11, 3)
		%t114 = AIE.tile(11, 4)
		%t121 = AIE.tile(12, 1)
		%t122 = AIE.tile(12, 2)
		%t123 = AIE.tile(12, 3)
		%t124 = AIE.tile(12, 4)

		%sb01 = AIE.switchbox(%t01) {
			AIE.connect<South : 0, North : 0>
		}
		%sb02 = AIE.switchbox(%t02) {
			AIE.connect<South : 0, North : 0>
		}
		%sb03 = AIE.switchbox(%t03) {
			AIE.connect<South : 0, DMA : 0>
			AIE.connect<East : 0, DMA : 1>
		}
		%sb04 = AIE.switchbox(%t04) {
		}
		%sb11 = AIE.switchbox(%t11) {
			AIE.connect<South : 0, North : 0>
		}
		%sb12 = AIE.switchbox(%t12) {
			AIE.connect<South : 0, North : 0>
		}
		%sb13 = AIE.switchbox(%t13) {
			AIE.connect<South : 0, West : 0>
		}
		%sb14 = AIE.switchbox(%t14) {
			AIE.connect<East : 0, DMA : 0>
		}
		%sb21 = AIE.switchbox(%t21) {
			AIE.connect<South : 0, North : 0>
		}
		%sb22 = AIE.switchbox(%t22) {
			AIE.connect<South : 0, North : 0>
		}
		%sb23 = AIE.switchbox(%t23) {
			AIE.connect<South : 0, North : 0>
		}
		%sb24 = AIE.switchbox(%t24) {
			AIE.connect<South : 0, West : 0>
		}
		%sb31 = AIE.switchbox(%t31) {
			AIE.connect<South : 0, North : 0>
		}
		%sb32 = AIE.switchbox(%t32) {
			AIE.connect<South : 0, North : 0>
		}
		%sb33 = AIE.switchbox(%t33) {
			AIE.connect<South : 0, DMA : 0>
		}
		%sb34 = AIE.switchbox(%t34) {
		}
		%sb41 = AIE.switchbox(%t41) {
			AIE.connect<South : 0, North : 0>
		}
		%sb42 = AIE.switchbox(%t42) {
			AIE.connect<South : 0, DMA : 0>
		}
		%sb43 = AIE.switchbox(%t43) {
		}
		%sb44 = AIE.switchbox(%t44) {
		}
		%sb51 = AIE.switchbox(%t51) {
			AIE.connect<South : 0, North : 0>
		}
		%sb52 = AIE.switchbox(%t52) {
			AIE.connect<South : 0, North : 0>
		}
		%sb53 = AIE.switchbox(%t53) {
			AIE.connect<South : 0, DMA : 0>
		}
		%sb54 = AIE.switchbox(%t54) {
		}
		%sb61 = AIE.switchbox(%t61) {
			AIE.connect<South : 0, North : 0>
			AIE.connect<South : 1, North : 1>
		}
		%sb62 = AIE.switchbox(%t62) {
			AIE.connect<South : 0, North : 0>
			AIE.connect<South : 1, North : 1>
		}
		%sb63 = AIE.switchbox(%t63) {
			AIE.connect<South : 0, DMA : 0>
			AIE.connect<South : 1, DMA : 1>
		}
		%sb64 = AIE.switchbox(%t64) {
		}
		%sb71 = AIE.switchbox(%t71) {
			AIE.connect<South : 0, North : 0>
			AIE.connect<South : 1, North : 1>
		}
		%sb72 = AIE.switchbox(%t72) {
			AIE.connect<South : 0, North : 0>
			AIE.connect<South : 1, North : 1>
		}
		%sb73 = AIE.switchbox(%t73) {
			AIE.connect<South : 0, North : 0>
			AIE.connect<South : 1, North : 1>
		}
		%sb74 = AIE.switchbox(%t74) {
			AIE.connect<South : 0, DMA : 0>
			AIE.connect<South : 1, DMA : 1>
		}
		%sb81 = AIE.switchbox(%t81) {
		}
		%sb82 = AIE.switchbox(%t82) {
		}
		%sb83 = AIE.switchbox(%t83) {
		}
		%sb84 = AIE.switchbox(%t84) {
		}
		%sb91 = AIE.switchbox(%t91) {
			AIE.connect<South : 0, North : 0>
		}
		%sb92 = AIE.switchbox(%t92) {
			AIE.connect<South : 0, DMA : 0>
		}
		%sb93 = AIE.switchbox(%t93) {
		}
		%sb94 = AIE.switchbox(%t94) {
		}
		%sb101 = AIE.switchbox(%t101) {
			AIE.connect<South : 0, North : 0>
		}
		%sb102 = AIE.switchbox(%t102) {
			AIE.connect<South : 0, DMA : 0>
		}
		%sb103 = AIE.switchbox(%t103) {
		}
		%sb104 = AIE.switchbox(%t104) {
		}
		%sb111 = AIE.switchbox(%t111) {
			AIE.connect<South : 0, North : 0>
			AIE.connect<South : 1, North : 1>
		}
		%sb112 = AIE.switchbox(%t112) {
			AIE.connect<South : 0, North : 0>
			AIE.connect<South : 1, North : 1>
		}
		%sb113 = AIE.switchbox(%t113) {
			AIE.connect<South : 0, DMA : 0>
			AIE.connect<South : 1, DMA : 1>
		}
		%sb114 = AIE.switchbox(%t114) {
		}

		AIE.flow(%t20, DMA : 0, %t20, North: 0)
		AIE.flow(%t20, DMA : 1, %t60, North: 1)
		AIE.flow(%t30, DMA : 0, %t30, North: 0)
		AIE.flow(%t30, DMA : 1, %t70, North: 1)
		AIE.flow(%t60, DMA : 0, %t00, North: 0)
		AIE.flow(%t60, DMA : 1, %t40, North: 0)
		AIE.flow(%t70, DMA : 0, %t10, North: 0)
		AIE.flow(%t70, DMA : 1, %t50, North: 0)
		AIE.flow(%t100, DMA : 0, %t100, North: 0)
		AIE.flow(%t110, DMA : 0, %t110, North: 0)
		AIE.flow(%t180, DMA : 0, %t60, North: 0)
		AIE.flow(%t180, DMA : 1, %t90, North: 0)
		AIE.flow(%t190, DMA : 0, %t70, North: 0)
		AIE.flow(%t190, DMA : 1, %t110, North: 1)
	}
}
