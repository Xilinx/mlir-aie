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
// XFAIL: *

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
// CHECK:           AIE.wire(%[[VAL_14]] : Core, %[[VAL_148:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_14]] : DMA, %[[VAL_148]] : DMA)
// CHECK:           AIE.wire(%[[VAL_149:.*]] : North, %[[VAL_148]] : South)
// CHECK:           AIE.wire(%[[VAL_15]] : Core, %[[VAL_150:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_15]] : DMA, %[[VAL_150]] : DMA)
// CHECK:           AIE.wire(%[[VAL_148]] : North, %[[VAL_150]] : South)
// CHECK:           AIE.wire(%[[VAL_16]] : Core, %[[VAL_151:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_16]] : DMA, %[[VAL_151]] : DMA)
// CHECK:           AIE.wire(%[[VAL_150]] : North, %[[VAL_151]] : South)
// CHECK:           AIE.wire(%[[VAL_17]] : Core, %[[VAL_152:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_17]] : DMA, %[[VAL_152]] : DMA)
// CHECK:           AIE.wire(%[[VAL_151]] : North, %[[VAL_152]] : South)
// CHECK:           AIE.wire(%[[VAL_149]] : East, %[[VAL_153:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_148]] : East, %[[VAL_154:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_18]] : Core, %[[VAL_154]] : Core)
// CHECK:           AIE.wire(%[[VAL_18]] : DMA, %[[VAL_154]] : DMA)
// CHECK:           AIE.wire(%[[VAL_153]] : North, %[[VAL_154]] : South)
// CHECK:           AIE.wire(%[[VAL_150]] : East, %[[VAL_155:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_19]] : Core, %[[VAL_155]] : Core)
// CHECK:           AIE.wire(%[[VAL_19]] : DMA, %[[VAL_155]] : DMA)
// CHECK:           AIE.wire(%[[VAL_154]] : North, %[[VAL_155]] : South)
// CHECK:           AIE.wire(%[[VAL_151]] : East, %[[VAL_156:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_20]] : Core, %[[VAL_156]] : Core)
// CHECK:           AIE.wire(%[[VAL_20]] : DMA, %[[VAL_156]] : DMA)
// CHECK:           AIE.wire(%[[VAL_155]] : North, %[[VAL_156]] : South)
// CHECK:           AIE.wire(%[[VAL_152]] : East, %[[VAL_157:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_21]] : Core, %[[VAL_157]] : Core)
// CHECK:           AIE.wire(%[[VAL_21]] : DMA, %[[VAL_157]] : DMA)
// CHECK:           AIE.wire(%[[VAL_156]] : North, %[[VAL_157]] : South)
// CHECK:           AIE.wire(%[[VAL_153]] : East, %[[VAL_158:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_159:.*]] : North, %[[VAL_158]] : South)
// CHECK:           AIE.wire(%[[VAL_2]] : DMA, %[[VAL_159]] : DMA)
// CHECK:           AIE.wire(%[[VAL_154]] : East, %[[VAL_160:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_22]] : Core, %[[VAL_160]] : Core)
// CHECK:           AIE.wire(%[[VAL_22]] : DMA, %[[VAL_160]] : DMA)
// CHECK:           AIE.wire(%[[VAL_158]] : North, %[[VAL_160]] : South)
// CHECK:           AIE.wire(%[[VAL_155]] : East, %[[VAL_161:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_23]] : Core, %[[VAL_161]] : Core)
// CHECK:           AIE.wire(%[[VAL_23]] : DMA, %[[VAL_161]] : DMA)
// CHECK:           AIE.wire(%[[VAL_160]] : North, %[[VAL_161]] : South)
// CHECK:           AIE.wire(%[[VAL_156]] : East, %[[VAL_162:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_24]] : Core, %[[VAL_162]] : Core)
// CHECK:           AIE.wire(%[[VAL_24]] : DMA, %[[VAL_162]] : DMA)
// CHECK:           AIE.wire(%[[VAL_161]] : North, %[[VAL_162]] : South)
// CHECK:           AIE.wire(%[[VAL_157]] : East, %[[VAL_163:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_25]] : Core, %[[VAL_163]] : Core)
// CHECK:           AIE.wire(%[[VAL_25]] : DMA, %[[VAL_163]] : DMA)
// CHECK:           AIE.wire(%[[VAL_162]] : North, %[[VAL_163]] : South)
// CHECK:           AIE.wire(%[[VAL_158]] : East, %[[VAL_164:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_165:.*]] : North, %[[VAL_164]] : South)
// CHECK:           AIE.wire(%[[VAL_3]] : DMA, %[[VAL_165]] : DMA)
// CHECK:           AIE.wire(%[[VAL_160]] : East, %[[VAL_166:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_26]] : Core, %[[VAL_166]] : Core)
// CHECK:           AIE.wire(%[[VAL_26]] : DMA, %[[VAL_166]] : DMA)
// CHECK:           AIE.wire(%[[VAL_164]] : North, %[[VAL_166]] : South)
// CHECK:           AIE.wire(%[[VAL_161]] : East, %[[VAL_167:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_27]] : Core, %[[VAL_167]] : Core)
// CHECK:           AIE.wire(%[[VAL_27]] : DMA, %[[VAL_167]] : DMA)
// CHECK:           AIE.wire(%[[VAL_166]] : North, %[[VAL_167]] : South)
// CHECK:           AIE.wire(%[[VAL_162]] : East, %[[VAL_168:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_28]] : Core, %[[VAL_168]] : Core)
// CHECK:           AIE.wire(%[[VAL_28]] : DMA, %[[VAL_168]] : DMA)
// CHECK:           AIE.wire(%[[VAL_167]] : North, %[[VAL_168]] : South)
// CHECK:           AIE.wire(%[[VAL_163]] : East, %[[VAL_169:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_29]] : Core, %[[VAL_169]] : Core)
// CHECK:           AIE.wire(%[[VAL_29]] : DMA, %[[VAL_169]] : DMA)
// CHECK:           AIE.wire(%[[VAL_168]] : North, %[[VAL_169]] : South)
// CHECK:           AIE.wire(%[[VAL_164]] : East, %[[VAL_170:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_166]] : East, %[[VAL_171:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_30]] : Core, %[[VAL_171]] : Core)
// CHECK:           AIE.wire(%[[VAL_30]] : DMA, %[[VAL_171]] : DMA)
// CHECK:           AIE.wire(%[[VAL_170]] : North, %[[VAL_171]] : South)
// CHECK:           AIE.wire(%[[VAL_167]] : East, %[[VAL_172:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_31]] : Core, %[[VAL_172]] : Core)
// CHECK:           AIE.wire(%[[VAL_31]] : DMA, %[[VAL_172]] : DMA)
// CHECK:           AIE.wire(%[[VAL_171]] : North, %[[VAL_172]] : South)
// CHECK:           AIE.wire(%[[VAL_168]] : East, %[[VAL_173:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_32]] : Core, %[[VAL_173]] : Core)
// CHECK:           AIE.wire(%[[VAL_32]] : DMA, %[[VAL_173]] : DMA)
// CHECK:           AIE.wire(%[[VAL_172]] : North, %[[VAL_173]] : South)
// CHECK:           AIE.wire(%[[VAL_169]] : East, %[[VAL_174:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_33]] : Core, %[[VAL_174]] : Core)
// CHECK:           AIE.wire(%[[VAL_33]] : DMA, %[[VAL_174]] : DMA)
// CHECK:           AIE.wire(%[[VAL_173]] : North, %[[VAL_174]] : South)
// CHECK:           AIE.wire(%[[VAL_170]] : East, %[[VAL_175:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_171]] : East, %[[VAL_176:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_34]] : Core, %[[VAL_176]] : Core)
// CHECK:           AIE.wire(%[[VAL_34]] : DMA, %[[VAL_176]] : DMA)
// CHECK:           AIE.wire(%[[VAL_175]] : North, %[[VAL_176]] : South)
// CHECK:           AIE.wire(%[[VAL_172]] : East, %[[VAL_177:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_35]] : Core, %[[VAL_177]] : Core)
// CHECK:           AIE.wire(%[[VAL_35]] : DMA, %[[VAL_177]] : DMA)
// CHECK:           AIE.wire(%[[VAL_176]] : North, %[[VAL_177]] : South)
// CHECK:           AIE.wire(%[[VAL_173]] : East, %[[VAL_178:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_36]] : Core, %[[VAL_178]] : Core)
// CHECK:           AIE.wire(%[[VAL_36]] : DMA, %[[VAL_178]] : DMA)
// CHECK:           AIE.wire(%[[VAL_177]] : North, %[[VAL_178]] : South)
// CHECK:           AIE.wire(%[[VAL_174]] : East, %[[VAL_179:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_37]] : Core, %[[VAL_179]] : Core)
// CHECK:           AIE.wire(%[[VAL_37]] : DMA, %[[VAL_179]] : DMA)
// CHECK:           AIE.wire(%[[VAL_178]] : North, %[[VAL_179]] : South)
// CHECK:           AIE.wire(%[[VAL_175]] : East, %[[VAL_180:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_181:.*]] : North, %[[VAL_180]] : South)
// CHECK:           AIE.wire(%[[VAL_6]] : DMA, %[[VAL_181]] : DMA)
// CHECK:           AIE.wire(%[[VAL_176]] : East, %[[VAL_182:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_38]] : Core, %[[VAL_182]] : Core)
// CHECK:           AIE.wire(%[[VAL_38]] : DMA, %[[VAL_182]] : DMA)
// CHECK:           AIE.wire(%[[VAL_180]] : North, %[[VAL_182]] : South)
// CHECK:           AIE.wire(%[[VAL_177]] : East, %[[VAL_183:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_39]] : Core, %[[VAL_183]] : Core)
// CHECK:           AIE.wire(%[[VAL_39]] : DMA, %[[VAL_183]] : DMA)
// CHECK:           AIE.wire(%[[VAL_182]] : North, %[[VAL_183]] : South)
// CHECK:           AIE.wire(%[[VAL_178]] : East, %[[VAL_184:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_40]] : Core, %[[VAL_184]] : Core)
// CHECK:           AIE.wire(%[[VAL_40]] : DMA, %[[VAL_184]] : DMA)
// CHECK:           AIE.wire(%[[VAL_183]] : North, %[[VAL_184]] : South)
// CHECK:           AIE.wire(%[[VAL_179]] : East, %[[VAL_185:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_41]] : Core, %[[VAL_185]] : Core)
// CHECK:           AIE.wire(%[[VAL_41]] : DMA, %[[VAL_185]] : DMA)
// CHECK:           AIE.wire(%[[VAL_184]] : North, %[[VAL_185]] : South)
// CHECK:           AIE.wire(%[[VAL_180]] : East, %[[VAL_186:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_187:.*]] : North, %[[VAL_186]] : South)
// CHECK:           AIE.wire(%[[VAL_7]] : DMA, %[[VAL_187]] : DMA)
// CHECK:           AIE.wire(%[[VAL_182]] : East, %[[VAL_188:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_42]] : Core, %[[VAL_188]] : Core)
// CHECK:           AIE.wire(%[[VAL_42]] : DMA, %[[VAL_188]] : DMA)
// CHECK:           AIE.wire(%[[VAL_186]] : North, %[[VAL_188]] : South)
// CHECK:           AIE.wire(%[[VAL_183]] : East, %[[VAL_189:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_43]] : Core, %[[VAL_189]] : Core)
// CHECK:           AIE.wire(%[[VAL_43]] : DMA, %[[VAL_189]] : DMA)
// CHECK:           AIE.wire(%[[VAL_188]] : North, %[[VAL_189]] : South)
// CHECK:           AIE.wire(%[[VAL_184]] : East, %[[VAL_190:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_44]] : Core, %[[VAL_190]] : Core)
// CHECK:           AIE.wire(%[[VAL_44]] : DMA, %[[VAL_190]] : DMA)
// CHECK:           AIE.wire(%[[VAL_189]] : North, %[[VAL_190]] : South)
// CHECK:           AIE.wire(%[[VAL_185]] : East, %[[VAL_191:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_45]] : Core, %[[VAL_191]] : Core)
// CHECK:           AIE.wire(%[[VAL_45]] : DMA, %[[VAL_191]] : DMA)
// CHECK:           AIE.wire(%[[VAL_190]] : North, %[[VAL_191]] : South)
// CHECK:           AIE.wire(%[[VAL_186]] : East, %[[VAL_192:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_188]] : East, %[[VAL_193:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_46]] : Core, %[[VAL_193]] : Core)
// CHECK:           AIE.wire(%[[VAL_46]] : DMA, %[[VAL_193]] : DMA)
// CHECK:           AIE.wire(%[[VAL_192]] : North, %[[VAL_193]] : South)
// CHECK:           AIE.wire(%[[VAL_189]] : East, %[[VAL_194:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_47]] : Core, %[[VAL_194]] : Core)
// CHECK:           AIE.wire(%[[VAL_47]] : DMA, %[[VAL_194]] : DMA)
// CHECK:           AIE.wire(%[[VAL_193]] : North, %[[VAL_194]] : South)
// CHECK:           AIE.wire(%[[VAL_190]] : East, %[[VAL_195:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_48]] : Core, %[[VAL_195]] : Core)
// CHECK:           AIE.wire(%[[VAL_48]] : DMA, %[[VAL_195]] : DMA)
// CHECK:           AIE.wire(%[[VAL_194]] : North, %[[VAL_195]] : South)
// CHECK:           AIE.wire(%[[VAL_191]] : East, %[[VAL_196:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_49]] : Core, %[[VAL_196]] : Core)
// CHECK:           AIE.wire(%[[VAL_49]] : DMA, %[[VAL_196]] : DMA)
// CHECK:           AIE.wire(%[[VAL_195]] : North, %[[VAL_196]] : South)
// CHECK:           AIE.wire(%[[VAL_192]] : East, %[[VAL_197:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_193]] : East, %[[VAL_198:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_50]] : Core, %[[VAL_198]] : Core)
// CHECK:           AIE.wire(%[[VAL_50]] : DMA, %[[VAL_198]] : DMA)
// CHECK:           AIE.wire(%[[VAL_197]] : North, %[[VAL_198]] : South)
// CHECK:           AIE.wire(%[[VAL_194]] : East, %[[VAL_199:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_51]] : Core, %[[VAL_199]] : Core)
// CHECK:           AIE.wire(%[[VAL_51]] : DMA, %[[VAL_199]] : DMA)
// CHECK:           AIE.wire(%[[VAL_198]] : North, %[[VAL_199]] : South)
// CHECK:           AIE.wire(%[[VAL_195]] : East, %[[VAL_200:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_52]] : Core, %[[VAL_200]] : Core)
// CHECK:           AIE.wire(%[[VAL_52]] : DMA, %[[VAL_200]] : DMA)
// CHECK:           AIE.wire(%[[VAL_199]] : North, %[[VAL_200]] : South)
// CHECK:           AIE.wire(%[[VAL_196]] : East, %[[VAL_201:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_53]] : Core, %[[VAL_201]] : Core)
// CHECK:           AIE.wire(%[[VAL_53]] : DMA, %[[VAL_201]] : DMA)
// CHECK:           AIE.wire(%[[VAL_200]] : North, %[[VAL_201]] : South)
// CHECK:           AIE.wire(%[[VAL_197]] : East, %[[VAL_202:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_203:.*]] : North, %[[VAL_202]] : South)
// CHECK:           AIE.wire(%[[VAL_10]] : DMA, %[[VAL_203]] : DMA)
// CHECK:           AIE.wire(%[[VAL_198]] : East, %[[VAL_204:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_54]] : Core, %[[VAL_204]] : Core)
// CHECK:           AIE.wire(%[[VAL_54]] : DMA, %[[VAL_204]] : DMA)
// CHECK:           AIE.wire(%[[VAL_202]] : North, %[[VAL_204]] : South)
// CHECK:           AIE.wire(%[[VAL_199]] : East, %[[VAL_205:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_55]] : Core, %[[VAL_205]] : Core)
// CHECK:           AIE.wire(%[[VAL_55]] : DMA, %[[VAL_205]] : DMA)
// CHECK:           AIE.wire(%[[VAL_204]] : North, %[[VAL_205]] : South)
// CHECK:           AIE.wire(%[[VAL_200]] : East, %[[VAL_206:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_56]] : Core, %[[VAL_206]] : Core)
// CHECK:           AIE.wire(%[[VAL_56]] : DMA, %[[VAL_206]] : DMA)
// CHECK:           AIE.wire(%[[VAL_205]] : North, %[[VAL_206]] : South)
// CHECK:           AIE.wire(%[[VAL_201]] : East, %[[VAL_207:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_57]] : Core, %[[VAL_207]] : Core)
// CHECK:           AIE.wire(%[[VAL_57]] : DMA, %[[VAL_207]] : DMA)
// CHECK:           AIE.wire(%[[VAL_206]] : North, %[[VAL_207]] : South)
// CHECK:           AIE.wire(%[[VAL_202]] : East, %[[VAL_208:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_209:.*]] : North, %[[VAL_208]] : South)
// CHECK:           AIE.wire(%[[VAL_11]] : DMA, %[[VAL_209]] : DMA)
// CHECK:           AIE.wire(%[[VAL_204]] : East, %[[VAL_210:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_58]] : Core, %[[VAL_210]] : Core)
// CHECK:           AIE.wire(%[[VAL_58]] : DMA, %[[VAL_210]] : DMA)
// CHECK:           AIE.wire(%[[VAL_208]] : North, %[[VAL_210]] : South)
// CHECK:           AIE.wire(%[[VAL_205]] : East, %[[VAL_211:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_59]] : Core, %[[VAL_211]] : Core)
// CHECK:           AIE.wire(%[[VAL_59]] : DMA, %[[VAL_211]] : DMA)
// CHECK:           AIE.wire(%[[VAL_210]] : North, %[[VAL_211]] : South)
// CHECK:           AIE.wire(%[[VAL_206]] : East, %[[VAL_212:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_60]] : Core, %[[VAL_212]] : Core)
// CHECK:           AIE.wire(%[[VAL_60]] : DMA, %[[VAL_212]] : DMA)
// CHECK:           AIE.wire(%[[VAL_211]] : North, %[[VAL_212]] : South)
// CHECK:           AIE.wire(%[[VAL_207]] : East, %[[VAL_213:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_61]] : Core, %[[VAL_213]] : Core)
// CHECK:           AIE.wire(%[[VAL_61]] : DMA, %[[VAL_213]] : DMA)
// CHECK:           AIE.wire(%[[VAL_212]] : North, %[[VAL_213]] : South)
// CHECK:           AIE.wire(%[[VAL_208]] : East, %[[VAL_214:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_214]] : East, %[[VAL_215:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_215]] : East, %[[VAL_216:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_216]] : East, %[[VAL_217:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_217]] : East, %[[VAL_218:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_218]] : East, %[[VAL_219:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_219]] : East, %[[VAL_220:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_221:.*]] : North, %[[VAL_220]] : South)
// CHECK:           AIE.wire(%[[VAL_12]] : DMA, %[[VAL_221]] : DMA)
// CHECK:           AIE.wire(%[[VAL_220]] : East, %[[VAL_222:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_223:.*]] : North, %[[VAL_222]] : South)
// CHECK:           AIE.wire(%[[VAL_13]] : DMA, %[[VAL_223]] : DMA)
// CHECK:         }

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
