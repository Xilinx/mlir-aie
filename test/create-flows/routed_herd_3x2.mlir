//===- routed_herd_3x2.mlir ------------------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_18:.*]] = AIE.tile(0, 5)
// CHECK:           %[[VAL_19:.*]] = AIE.tile(0, 6)
// CHECK:           %[[VAL_20:.*]] = AIE.tile(0, 7)
// CHECK:           %[[VAL_21:.*]] = AIE.tile(0, 8)
// CHECK:           %[[VAL_22:.*]] = AIE.tile(1, 1)
// CHECK:           %[[VAL_23:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_24:.*]] = AIE.tile(1, 3)
// CHECK:           %[[VAL_25:.*]] = AIE.tile(1, 4)
// CHECK:           %[[VAL_26:.*]] = AIE.tile(1, 5)
// CHECK:           %[[VAL_27:.*]] = AIE.tile(1, 6)
// CHECK:           %[[VAL_28:.*]] = AIE.tile(1, 7)
// CHECK:           %[[VAL_29:.*]] = AIE.tile(1, 8)
// CHECK:           %[[VAL_30:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_31:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_32:.*]] = AIE.tile(2, 3)
// CHECK:           %[[VAL_33:.*]] = AIE.tile(2, 4)
// CHECK:           %[[VAL_34:.*]] = AIE.tile(2, 5)
// CHECK:           %[[VAL_35:.*]] = AIE.tile(2, 6)
// CHECK:           %[[VAL_36:.*]] = AIE.tile(2, 7)
// CHECK:           %[[VAL_37:.*]] = AIE.tile(2, 8)
// CHECK:           %[[VAL_38:.*]] = AIE.tile(3, 1)
// CHECK:           %[[VAL_39:.*]] = AIE.tile(3, 2)
// CHECK:           %[[VAL_40:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_41:.*]] = AIE.tile(3, 4)
// CHECK:           %[[VAL_42:.*]] = AIE.tile(3, 5)
// CHECK:           %[[VAL_43:.*]] = AIE.tile(3, 6)
// CHECK:           %[[VAL_44:.*]] = AIE.tile(3, 7)
// CHECK:           %[[VAL_45:.*]] = AIE.tile(3, 8)
// CHECK:           %[[VAL_46:.*]] = AIE.tile(4, 1)
// CHECK:           %[[VAL_47:.*]] = AIE.tile(4, 2)
// CHECK:           %[[VAL_48:.*]] = AIE.tile(4, 3)
// CHECK:           %[[VAL_49:.*]] = AIE.tile(4, 4)
// CHECK:           %[[VAL_50:.*]] = AIE.tile(4, 5)
// CHECK:           %[[VAL_51:.*]] = AIE.tile(4, 6)
// CHECK:           %[[VAL_52:.*]] = AIE.tile(4, 7)
// CHECK:           %[[VAL_53:.*]] = AIE.tile(4, 8)
// CHECK:           %[[VAL_54:.*]] = AIE.tile(5, 1)
// CHECK:           %[[VAL_55:.*]] = AIE.tile(5, 2)
// CHECK:           %[[VAL_56:.*]] = AIE.tile(5, 3)
// CHECK:           %[[VAL_57:.*]] = AIE.tile(5, 4)
// CHECK:           %[[VAL_58:.*]] = AIE.tile(5, 5)
// CHECK:           %[[VAL_59:.*]] = AIE.tile(5, 6)
// CHECK:           %[[VAL_60:.*]] = AIE.tile(5, 7)
// CHECK:           %[[VAL_61:.*]] = AIE.tile(5, 8)
// CHECK:           %[[VAL_62:.*]] = AIE.tile(6, 1)
// CHECK:           %[[VAL_63:.*]] = AIE.tile(6, 2)
// CHECK:           %[[VAL_64:.*]] = AIE.tile(6, 3)
// CHECK:           %[[VAL_65:.*]] = AIE.tile(6, 4)
// CHECK:           %[[VAL_66:.*]] = AIE.tile(6, 5)
// CHECK:           %[[VAL_67:.*]] = AIE.tile(6, 6)
// CHECK:           %[[VAL_68:.*]] = AIE.tile(6, 7)
// CHECK:           %[[VAL_69:.*]] = AIE.tile(6, 8)
// CHECK:           %[[VAL_70:.*]] = AIE.tile(7, 1)
// CHECK:           %[[VAL_71:.*]] = AIE.tile(7, 2)
// CHECK:           %[[VAL_72:.*]] = AIE.tile(7, 3)
// CHECK:           %[[VAL_73:.*]] = AIE.tile(7, 4)
// CHECK:           %[[VAL_74:.*]] = AIE.tile(7, 5)
// CHECK:           %[[VAL_75:.*]] = AIE.tile(7, 6)
// CHECK:           %[[VAL_76:.*]] = AIE.tile(7, 7)
// CHECK:           %[[VAL_77:.*]] = AIE.tile(7, 8)
// CHECK:           %[[VAL_78:.*]] = AIE.tile(8, 1)
// CHECK:           %[[VAL_79:.*]] = AIE.tile(8, 2)
// CHECK:           %[[VAL_80:.*]] = AIE.tile(8, 3)
// CHECK:           %[[VAL_81:.*]] = AIE.tile(8, 4)
// CHECK:           %[[VAL_82:.*]] = AIE.tile(8, 5)
// CHECK:           %[[VAL_83:.*]] = AIE.tile(8, 6)
// CHECK:           %[[VAL_84:.*]] = AIE.tile(8, 7)
// CHECK:           %[[VAL_85:.*]] = AIE.tile(8, 8)
// CHECK:           %[[VAL_86:.*]] = AIE.tile(9, 1)
// CHECK:           %[[VAL_87:.*]] = AIE.tile(9, 2)
// CHECK:           %[[VAL_88:.*]] = AIE.tile(9, 3)
// CHECK:           %[[VAL_89:.*]] = AIE.tile(9, 4)
// CHECK:           %[[VAL_90:.*]] = AIE.tile(9, 5)
// CHECK:           %[[VAL_91:.*]] = AIE.tile(9, 6)
// CHECK:           %[[VAL_92:.*]] = AIE.tile(9, 7)
// CHECK:           %[[VAL_93:.*]] = AIE.tile(9, 8)
// CHECK:           %[[VAL_94:.*]] = AIE.tile(10, 1)
// CHECK:           %[[VAL_95:.*]] = AIE.tile(10, 2)
// CHECK:           %[[VAL_96:.*]] = AIE.tile(10, 3)
// CHECK:           %[[VAL_97:.*]] = AIE.tile(10, 4)
// CHECK:           %[[VAL_98:.*]] = AIE.tile(10, 5)
// CHECK:           %[[VAL_99:.*]] = AIE.tile(10, 6)
// CHECK:           %[[VAL_100:.*]] = AIE.tile(10, 7)
// CHECK:           %[[VAL_101:.*]] = AIE.tile(10, 8)
// CHECK:           %[[VAL_102:.*]] = AIE.tile(11, 1)
// CHECK:           %[[VAL_103:.*]] = AIE.tile(11, 2)
// CHECK:           %[[VAL_104:.*]] = AIE.tile(11, 3)
// CHECK:           %[[VAL_105:.*]] = AIE.tile(11, 4)
// CHECK:           %[[VAL_106:.*]] = AIE.tile(11, 5)
// CHECK:           %[[VAL_107:.*]] = AIE.tile(11, 6)
// CHECK:           %[[VAL_108:.*]] = AIE.tile(11, 7)
// CHECK:           %[[VAL_109:.*]] = AIE.tile(11, 8)
// CHECK:           %[[VAL_110:.*]] = AIE.tile(12, 1)
// CHECK:           %[[VAL_111:.*]] = AIE.tile(12, 2)
// CHECK:           %[[VAL_112:.*]] = AIE.tile(12, 3)
// CHECK:           %[[VAL_113:.*]] = AIE.tile(12, 4)
// CHECK:           %[[VAL_114:.*]] = AIE.tile(12, 5)
// CHECK:           %[[VAL_115:.*]] = AIE.tile(12, 6)
// CHECK:           %[[VAL_116:.*]] = AIE.tile(12, 7)
// CHECK:           %[[VAL_117:.*]] = AIE.tile(12, 8)
// CHECK:           %[[VAL_118:.*]] = AIE.tile(13, 0)
// CHECK:           %[[VAL_119:.*]] = AIE.tile(13, 1)
// CHECK:           %[[VAL_120:.*]] = AIE.tile(13, 2)
// CHECK:           %[[VAL_121:.*]] = AIE.tile(13, 3)
// CHECK:           %[[VAL_122:.*]] = AIE.tile(13, 4)
// CHECK:           %[[VAL_123:.*]] = AIE.tile(13, 5)
// CHECK:           %[[VAL_124:.*]] = AIE.tile(13, 6)
// CHECK:           %[[VAL_125:.*]] = AIE.tile(13, 7)
// CHECK:           %[[VAL_126:.*]] = AIE.tile(13, 8)
// CHECK:           %[[VAL_127:.*]] = AIE.tile(14, 1)
// CHECK:           %[[VAL_128:.*]] = AIE.tile(14, 2)
// CHECK:           %[[VAL_129:.*]] = AIE.tile(14, 3)
// CHECK:           %[[VAL_130:.*]] = AIE.tile(14, 4)
// CHECK:           %[[VAL_131:.*]] = AIE.tile(14, 5)
// CHECK:           %[[VAL_132:.*]] = AIE.tile(14, 6)
// CHECK:           %[[VAL_133:.*]] = AIE.tile(14, 7)
// CHECK:           %[[VAL_134:.*]] = AIE.tile(14, 8)
// CHECK:           %[[VAL_135:.*]] = AIE.switchbox(%[[VAL_14]]) {
// CHECK:           }
// CHECK:           %[[VAL_136:.*]] = AIE.switchbox(%[[VAL_15]]) {
// CHECK:           }
// CHECK:           %[[VAL_137:.*]] = AIE.switchbox(%[[VAL_16]]) {
// CHECK:           }
// CHECK:           %[[VAL_138:.*]] = AIE.switchbox(%[[VAL_17]]) {
// CHECK:           }
// CHECK:           %[[VAL_139:.*]] = AIE.switchbox(%[[VAL_22]]) {
// CHECK:           }
// CHECK:           %[[VAL_140:.*]] = AIE.switchbox(%[[VAL_23]]) {
// CHECK:           }
// CHECK:           %[[VAL_141:.*]] = AIE.switchbox(%[[VAL_24]]) {
// CHECK:           }
// CHECK:           %[[VAL_142:.*]] = AIE.switchbox(%[[VAL_25]]) {
// CHECK:           }
// CHECK:           %[[VAL_143:.*]] = AIE.switchbox(%[[VAL_30]]) {
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_144:.*]] = AIE.switchbox(%[[VAL_31]]) {
// CHECK:           }
// CHECK:           %[[VAL_145:.*]] = AIE.switchbox(%[[VAL_32]]) {
// CHECK:           }
// CHECK:           %[[VAL_146:.*]] = AIE.switchbox(%[[VAL_33]]) {
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_147:.*]] = AIE.switchbox(%[[VAL_34]]) {
// CHECK:             AIE.connect<South : 0, Core : 0>
// CHECK:             AIE.connect<DMA : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_148:.*]] = AIE.switchbox(%[[VAL_38]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<Core : 0, North : 0>
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_149:.*]] = AIE.switchbox(%[[VAL_39]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_150:.*]] = AIE.switchbox(%[[VAL_40]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_151:.*]] = AIE.switchbox(%[[VAL_41]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_152:.*]] = AIE.switchbox(%[[VAL_42]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_153:.*]] = AIE.switchbox(%[[VAL_46]]) {
// CHECK:           }
// CHECK:           %[[VAL_154:.*]] = AIE.switchbox(%[[VAL_47]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_155:.*]] = AIE.switchbox(%[[VAL_48]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_156:.*]] = AIE.switchbox(%[[VAL_49]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<North : 1, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_157:.*]] = AIE.switchbox(%[[VAL_54]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_158:.*]] = AIE.switchbox(%[[VAL_55]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_159:.*]] = AIE.switchbox(%[[VAL_56]]) {
// CHECK:           }
// CHECK:           %[[VAL_160:.*]] = AIE.switchbox(%[[VAL_57]]) {
// CHECK:           }
// CHECK:           %[[VAL_161:.*]] = AIE.switchbox(%[[VAL_58]]) {
// CHECK:           }
// CHECK:           %[[VAL_162:.*]] = AIE.switchbox(%[[VAL_59]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_163:.*]] = AIE.switchbox(%[[VAL_62]]) {
// CHECK:           }
// CHECK:           %[[VAL_164:.*]] = AIE.switchbox(%[[VAL_63]]) {
// CHECK:           }
// CHECK:           %[[VAL_165:.*]] = AIE.switchbox(%[[VAL_64]]) {
// CHECK:           }
// CHECK:           %[[VAL_166:.*]] = AIE.switchbox(%[[VAL_65]]) {
// CHECK:           }
// CHECK:           %[[VAL_167:.*]] = AIE.switchbox(%[[VAL_66]]) {
// CHECK:           }
// CHECK:           %[[VAL_168:.*]] = AIE.switchbox(%[[VAL_67]]) {
// CHECK:             AIE.connect<East : 0, Core : 0>
// CHECK:             AIE.connect<DMA : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_169:.*]] = AIE.switchbox(%[[VAL_70]]) {
// CHECK:           }
// CHECK:           %[[VAL_170:.*]] = AIE.switchbox(%[[VAL_71]]) {
// CHECK:           }
// CHECK:           %[[VAL_171:.*]] = AIE.switchbox(%[[VAL_72]]) {
// CHECK:             AIE.connect<East : 0, DMA : 0>
// CHECK:             AIE.connect<Core : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_172:.*]] = AIE.switchbox(%[[VAL_73]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_173:.*]] = AIE.switchbox(%[[VAL_74]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_174:.*]] = AIE.switchbox(%[[VAL_75]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_175:.*]] = AIE.switchbox(%[[VAL_78]]) {
// CHECK:           }
// CHECK:           %[[VAL_176:.*]] = AIE.switchbox(%[[VAL_79]]) {
// CHECK:           }
// CHECK:           %[[VAL_177:.*]] = AIE.switchbox(%[[VAL_80]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_178:.*]] = AIE.switchbox(%[[VAL_81]]) {
// CHECK:           }
// CHECK:           %[[VAL_179:.*]] = AIE.switchbox(%[[VAL_86]]) {
// CHECK:           }
// CHECK:           %[[VAL_180:.*]] = AIE.switchbox(%[[VAL_87]]) {
// CHECK:           }
// CHECK:           %[[VAL_181:.*]] = AIE.switchbox(%[[VAL_88]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_182:.*]] = AIE.switchbox(%[[VAL_89]]) {
// CHECK:           }
// CHECK:           %[[VAL_183:.*]] = AIE.switchbox(%[[VAL_94]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_184:.*]] = AIE.switchbox(%[[VAL_95]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_185:.*]] = AIE.switchbox(%[[VAL_96]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_186:.*]] = AIE.switchbox(%[[VAL_97]]) {
// CHECK:           }
// CHECK:           %[[VAL_187:.*]] = AIE.switchbox(%[[VAL_102]]) {
// CHECK:           }
// CHECK:           %[[VAL_188:.*]] = AIE.switchbox(%[[VAL_103]]) {
// CHECK:           }
// CHECK:           %[[VAL_189:.*]] = AIE.switchbox(%[[VAL_104]]) {
// CHECK:           }
// CHECK:           %[[VAL_190:.*]] = AIE.switchbox(%[[VAL_105]]) {
// CHECK:           }
// CHECK:           %[[VAL_191:.*]] = AIE.switchbox(%[[VAL_110]]) {
// CHECK:           }
// CHECK:           %[[VAL_192:.*]] = AIE.switchbox(%[[VAL_111]]) {
// CHECK:           }
// CHECK:           %[[VAL_193:.*]] = AIE.switchbox(%[[VAL_112]]) {
// CHECK:           }
// CHECK:           %[[VAL_194:.*]] = AIE.switchbox(%[[VAL_113]]) {
// CHECK:           }
// CHECK:           %[[VAL_195:.*]] = AIE.switchbox(%[[VAL_114]]) {
// CHECK:             AIE.connect<East : 0, Core : 0>
// CHECK:             AIE.connect<DMA : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_196:.*]] = AIE.switchbox(%[[VAL_119]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_197:.*]] = AIE.switchbox(%[[VAL_120]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_198:.*]] = AIE.switchbox(%[[VAL_121]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<Core : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_199:.*]] = AIE.switchbox(%[[VAL_122]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_200:.*]] = AIE.switchbox(%[[VAL_123]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_201:.*]] = AIE.switchbox(%[[VAL_3]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_202:.*]] = AIE.shimmux(%[[VAL_3]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_203:.*]] = AIE.switchbox(%[[VAL_5]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_204:.*]] = AIE.switchbox(%[[VAL_6]]) {
// CHECK:             AIE.connect<West : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_205:.*]] = AIE.shimmux(%[[VAL_6]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_206:.*]] = AIE.switchbox(%[[VAL_50]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_207:.*]] = AIE.switchbox(%[[VAL_10]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_208:.*]] = AIE.shimmux(%[[VAL_10]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_209:.*]] = AIE.switchbox(%[[VAL_2]]) {
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_210:.*]] = AIE.shimmux(%[[VAL_2]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_211:.*]] = AIE.switchbox(%[[VAL_51]]) {
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_212:.*]] = AIE.switchbox(%[[VAL_11]]) {
// CHECK:             AIE.connect<South : 3, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_213:.*]] = AIE.shimmux(%[[VAL_11]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_214:.*]] = AIE.tile(12, 0)
// CHECK:           %[[VAL_215:.*]] = AIE.switchbox(%[[VAL_214]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_216:.*]] = AIE.switchbox(%[[VAL_118]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_217:.*]] = AIE.tile(17, 0)
// CHECK:           %[[VAL_218:.*]] = AIE.switchbox(%[[VAL_217]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_219:.*]] = AIE.switchbox(%[[VAL_12]]) {
// CHECK:             AIE.connect<West : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_220:.*]] = AIE.shimmux(%[[VAL_12]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_221:.*]] = AIE.tile(17, 1)
// CHECK:           %[[VAL_222:.*]] = AIE.switchbox(%[[VAL_221]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_223:.*]] = AIE.switchbox(%[[VAL_128]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_224:.*]] = AIE.tile(15, 2)
// CHECK:           %[[VAL_225:.*]] = AIE.switchbox(%[[VAL_224]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_226:.*]] = AIE.tile(16, 2)
// CHECK:           %[[VAL_227:.*]] = AIE.switchbox(%[[VAL_226]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_228:.*]] = AIE.tile(17, 2)
// CHECK:           %[[VAL_229:.*]] = AIE.switchbox(%[[VAL_228]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_230:.*]] = AIE.switchbox(%[[VAL_129]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_231:.*]] = AIE.switchbox(%[[VAL_130]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_232:.*]] = AIE.switchbox(%[[VAL_131]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_14]] : Core, %[[VAL_135]] : Core)
// CHECK:           AIE.wire(%[[VAL_14]] : DMA, %[[VAL_135]] : DMA)
// CHECK:           AIE.wire(%[[VAL_15]] : Core, %[[VAL_136]] : Core)
// CHECK:           AIE.wire(%[[VAL_15]] : DMA, %[[VAL_136]] : DMA)
// CHECK:           AIE.wire(%[[VAL_135]] : North, %[[VAL_136]] : South)
// CHECK:           AIE.wire(%[[VAL_16]] : Core, %[[VAL_137]] : Core)
// CHECK:           AIE.wire(%[[VAL_16]] : DMA, %[[VAL_137]] : DMA)
// CHECK:           AIE.wire(%[[VAL_136]] : North, %[[VAL_137]] : South)
// CHECK:           AIE.wire(%[[VAL_17]] : Core, %[[VAL_138]] : Core)
// CHECK:           AIE.wire(%[[VAL_17]] : DMA, %[[VAL_138]] : DMA)
// CHECK:           AIE.wire(%[[VAL_137]] : North, %[[VAL_138]] : South)
// CHECK:           AIE.wire(%[[VAL_135]] : East, %[[VAL_139]] : West)
// CHECK:           AIE.wire(%[[VAL_22]] : Core, %[[VAL_139]] : Core)
// CHECK:           AIE.wire(%[[VAL_22]] : DMA, %[[VAL_139]] : DMA)
// CHECK:           AIE.wire(%[[VAL_136]] : East, %[[VAL_140]] : West)
// CHECK:           AIE.wire(%[[VAL_23]] : Core, %[[VAL_140]] : Core)
// CHECK:           AIE.wire(%[[VAL_23]] : DMA, %[[VAL_140]] : DMA)
// CHECK:           AIE.wire(%[[VAL_139]] : North, %[[VAL_140]] : South)
// CHECK:           AIE.wire(%[[VAL_137]] : East, %[[VAL_141]] : West)
// CHECK:           AIE.wire(%[[VAL_24]] : Core, %[[VAL_141]] : Core)
// CHECK:           AIE.wire(%[[VAL_24]] : DMA, %[[VAL_141]] : DMA)
// CHECK:           AIE.wire(%[[VAL_140]] : North, %[[VAL_141]] : South)
// CHECK:           AIE.wire(%[[VAL_138]] : East, %[[VAL_142]] : West)
// CHECK:           AIE.wire(%[[VAL_25]] : Core, %[[VAL_142]] : Core)
// CHECK:           AIE.wire(%[[VAL_25]] : DMA, %[[VAL_142]] : DMA)
// CHECK:           AIE.wire(%[[VAL_141]] : North, %[[VAL_142]] : South)
// CHECK:           AIE.wire(%[[VAL_210]] : North, %[[VAL_209]] : South)
// CHECK:           AIE.wire(%[[VAL_2]] : DMA, %[[VAL_210]] : DMA)
// CHECK:           AIE.wire(%[[VAL_139]] : East, %[[VAL_143]] : West)
// CHECK:           AIE.wire(%[[VAL_30]] : Core, %[[VAL_143]] : Core)
// CHECK:           AIE.wire(%[[VAL_30]] : DMA, %[[VAL_143]] : DMA)
// CHECK:           AIE.wire(%[[VAL_209]] : North, %[[VAL_143]] : South)
// CHECK:           AIE.wire(%[[VAL_140]] : East, %[[VAL_144]] : West)
// CHECK:           AIE.wire(%[[VAL_31]] : Core, %[[VAL_144]] : Core)
// CHECK:           AIE.wire(%[[VAL_31]] : DMA, %[[VAL_144]] : DMA)
// CHECK:           AIE.wire(%[[VAL_143]] : North, %[[VAL_144]] : South)
// CHECK:           AIE.wire(%[[VAL_141]] : East, %[[VAL_145]] : West)
// CHECK:           AIE.wire(%[[VAL_32]] : Core, %[[VAL_145]] : Core)
// CHECK:           AIE.wire(%[[VAL_32]] : DMA, %[[VAL_145]] : DMA)
// CHECK:           AIE.wire(%[[VAL_144]] : North, %[[VAL_145]] : South)
// CHECK:           AIE.wire(%[[VAL_142]] : East, %[[VAL_146]] : West)
// CHECK:           AIE.wire(%[[VAL_33]] : Core, %[[VAL_146]] : Core)
// CHECK:           AIE.wire(%[[VAL_33]] : DMA, %[[VAL_146]] : DMA)
// CHECK:           AIE.wire(%[[VAL_145]] : North, %[[VAL_146]] : South)
// CHECK:           AIE.wire(%[[VAL_34]] : Core, %[[VAL_147]] : Core)
// CHECK:           AIE.wire(%[[VAL_34]] : DMA, %[[VAL_147]] : DMA)
// CHECK:           AIE.wire(%[[VAL_146]] : North, %[[VAL_147]] : South)
// CHECK:           AIE.wire(%[[VAL_209]] : East, %[[VAL_201]] : West)
// CHECK:           AIE.wire(%[[VAL_202]] : North, %[[VAL_201]] : South)
// CHECK:           AIE.wire(%[[VAL_3]] : DMA, %[[VAL_202]] : DMA)
// CHECK:           AIE.wire(%[[VAL_143]] : East, %[[VAL_148]] : West)
// CHECK:           AIE.wire(%[[VAL_38]] : Core, %[[VAL_148]] : Core)
// CHECK:           AIE.wire(%[[VAL_38]] : DMA, %[[VAL_148]] : DMA)
// CHECK:           AIE.wire(%[[VAL_201]] : North, %[[VAL_148]] : South)
// CHECK:           AIE.wire(%[[VAL_144]] : East, %[[VAL_149]] : West)
// CHECK:           AIE.wire(%[[VAL_39]] : Core, %[[VAL_149]] : Core)
// CHECK:           AIE.wire(%[[VAL_39]] : DMA, %[[VAL_149]] : DMA)
// CHECK:           AIE.wire(%[[VAL_148]] : North, %[[VAL_149]] : South)
// CHECK:           AIE.wire(%[[VAL_145]] : East, %[[VAL_150]] : West)
// CHECK:           AIE.wire(%[[VAL_40]] : Core, %[[VAL_150]] : Core)
// CHECK:           AIE.wire(%[[VAL_40]] : DMA, %[[VAL_150]] : DMA)
// CHECK:           AIE.wire(%[[VAL_149]] : North, %[[VAL_150]] : South)
// CHECK:           AIE.wire(%[[VAL_146]] : East, %[[VAL_151]] : West)
// CHECK:           AIE.wire(%[[VAL_41]] : Core, %[[VAL_151]] : Core)
// CHECK:           AIE.wire(%[[VAL_41]] : DMA, %[[VAL_151]] : DMA)
// CHECK:           AIE.wire(%[[VAL_150]] : North, %[[VAL_151]] : South)
// CHECK:           AIE.wire(%[[VAL_147]] : East, %[[VAL_152]] : West)
// CHECK:           AIE.wire(%[[VAL_42]] : Core, %[[VAL_152]] : Core)
// CHECK:           AIE.wire(%[[VAL_42]] : DMA, %[[VAL_152]] : DMA)
// CHECK:           AIE.wire(%[[VAL_151]] : North, %[[VAL_152]] : South)
// CHECK:           AIE.wire(%[[VAL_148]] : East, %[[VAL_153]] : West)
// CHECK:           AIE.wire(%[[VAL_46]] : Core, %[[VAL_153]] : Core)
// CHECK:           AIE.wire(%[[VAL_46]] : DMA, %[[VAL_153]] : DMA)
// CHECK:           AIE.wire(%[[VAL_149]] : East, %[[VAL_154]] : West)
// CHECK:           AIE.wire(%[[VAL_47]] : Core, %[[VAL_154]] : Core)
// CHECK:           AIE.wire(%[[VAL_47]] : DMA, %[[VAL_154]] : DMA)
// CHECK:           AIE.wire(%[[VAL_153]] : North, %[[VAL_154]] : South)
// CHECK:           AIE.wire(%[[VAL_150]] : East, %[[VAL_155]] : West)
// CHECK:           AIE.wire(%[[VAL_48]] : Core, %[[VAL_155]] : Core)
// CHECK:           AIE.wire(%[[VAL_48]] : DMA, %[[VAL_155]] : DMA)
// CHECK:           AIE.wire(%[[VAL_154]] : North, %[[VAL_155]] : South)
// CHECK:           AIE.wire(%[[VAL_151]] : East, %[[VAL_156]] : West)
// CHECK:           AIE.wire(%[[VAL_49]] : Core, %[[VAL_156]] : Core)
// CHECK:           AIE.wire(%[[VAL_49]] : DMA, %[[VAL_156]] : DMA)
// CHECK:           AIE.wire(%[[VAL_155]] : North, %[[VAL_156]] : South)
// CHECK:           AIE.wire(%[[VAL_152]] : East, %[[VAL_206]] : West)
// CHECK:           AIE.wire(%[[VAL_50]] : Core, %[[VAL_206]] : Core)
// CHECK:           AIE.wire(%[[VAL_50]] : DMA, %[[VAL_206]] : DMA)
// CHECK:           AIE.wire(%[[VAL_156]] : North, %[[VAL_206]] : South)
// CHECK:           AIE.wire(%[[VAL_51]] : Core, %[[VAL_211]] : Core)
// CHECK:           AIE.wire(%[[VAL_51]] : DMA, %[[VAL_211]] : DMA)
// CHECK:           AIE.wire(%[[VAL_206]] : North, %[[VAL_211]] : South)
// CHECK:           AIE.wire(%[[VAL_153]] : East, %[[VAL_157]] : West)
// CHECK:           AIE.wire(%[[VAL_54]] : Core, %[[VAL_157]] : Core)
// CHECK:           AIE.wire(%[[VAL_54]] : DMA, %[[VAL_157]] : DMA)
// CHECK:           AIE.wire(%[[VAL_203]] : North, %[[VAL_157]] : South)
// CHECK:           AIE.wire(%[[VAL_154]] : East, %[[VAL_158]] : West)
// CHECK:           AIE.wire(%[[VAL_55]] : Core, %[[VAL_158]] : Core)
// CHECK:           AIE.wire(%[[VAL_55]] : DMA, %[[VAL_158]] : DMA)
// CHECK:           AIE.wire(%[[VAL_157]] : North, %[[VAL_158]] : South)
// CHECK:           AIE.wire(%[[VAL_155]] : East, %[[VAL_159]] : West)
// CHECK:           AIE.wire(%[[VAL_56]] : Core, %[[VAL_159]] : Core)
// CHECK:           AIE.wire(%[[VAL_56]] : DMA, %[[VAL_159]] : DMA)
// CHECK:           AIE.wire(%[[VAL_158]] : North, %[[VAL_159]] : South)
// CHECK:           AIE.wire(%[[VAL_156]] : East, %[[VAL_160]] : West)
// CHECK:           AIE.wire(%[[VAL_57]] : Core, %[[VAL_160]] : Core)
// CHECK:           AIE.wire(%[[VAL_57]] : DMA, %[[VAL_160]] : DMA)
// CHECK:           AIE.wire(%[[VAL_159]] : North, %[[VAL_160]] : South)
// CHECK:           AIE.wire(%[[VAL_206]] : East, %[[VAL_161]] : West)
// CHECK:           AIE.wire(%[[VAL_58]] : Core, %[[VAL_161]] : Core)
// CHECK:           AIE.wire(%[[VAL_58]] : DMA, %[[VAL_161]] : DMA)
// CHECK:           AIE.wire(%[[VAL_160]] : North, %[[VAL_161]] : South)
// CHECK:           AIE.wire(%[[VAL_211]] : East, %[[VAL_162]] : West)
// CHECK:           AIE.wire(%[[VAL_59]] : Core, %[[VAL_162]] : Core)
// CHECK:           AIE.wire(%[[VAL_59]] : DMA, %[[VAL_162]] : DMA)
// CHECK:           AIE.wire(%[[VAL_161]] : North, %[[VAL_162]] : South)
// CHECK:           AIE.wire(%[[VAL_203]] : East, %[[VAL_204]] : West)
// CHECK:           AIE.wire(%[[VAL_205]] : North, %[[VAL_204]] : South)
// CHECK:           AIE.wire(%[[VAL_6]] : DMA, %[[VAL_205]] : DMA)
// CHECK:           AIE.wire(%[[VAL_157]] : East, %[[VAL_163]] : West)
// CHECK:           AIE.wire(%[[VAL_62]] : Core, %[[VAL_163]] : Core)
// CHECK:           AIE.wire(%[[VAL_62]] : DMA, %[[VAL_163]] : DMA)
// CHECK:           AIE.wire(%[[VAL_204]] : North, %[[VAL_163]] : South)
// CHECK:           AIE.wire(%[[VAL_158]] : East, %[[VAL_164]] : West)
// CHECK:           AIE.wire(%[[VAL_63]] : Core, %[[VAL_164]] : Core)
// CHECK:           AIE.wire(%[[VAL_63]] : DMA, %[[VAL_164]] : DMA)
// CHECK:           AIE.wire(%[[VAL_163]] : North, %[[VAL_164]] : South)
// CHECK:           AIE.wire(%[[VAL_159]] : East, %[[VAL_165]] : West)
// CHECK:           AIE.wire(%[[VAL_64]] : Core, %[[VAL_165]] : Core)
// CHECK:           AIE.wire(%[[VAL_64]] : DMA, %[[VAL_165]] : DMA)
// CHECK:           AIE.wire(%[[VAL_164]] : North, %[[VAL_165]] : South)
// CHECK:           AIE.wire(%[[VAL_160]] : East, %[[VAL_166]] : West)
// CHECK:           AIE.wire(%[[VAL_65]] : Core, %[[VAL_166]] : Core)
// CHECK:           AIE.wire(%[[VAL_65]] : DMA, %[[VAL_166]] : DMA)
// CHECK:           AIE.wire(%[[VAL_165]] : North, %[[VAL_166]] : South)
// CHECK:           AIE.wire(%[[VAL_161]] : East, %[[VAL_167]] : West)
// CHECK:           AIE.wire(%[[VAL_66]] : Core, %[[VAL_167]] : Core)
// CHECK:           AIE.wire(%[[VAL_66]] : DMA, %[[VAL_167]] : DMA)
// CHECK:           AIE.wire(%[[VAL_166]] : North, %[[VAL_167]] : South)
// CHECK:           AIE.wire(%[[VAL_162]] : East, %[[VAL_168]] : West)
// CHECK:           AIE.wire(%[[VAL_67]] : Core, %[[VAL_168]] : Core)
// CHECK:           AIE.wire(%[[VAL_67]] : DMA, %[[VAL_168]] : DMA)
// CHECK:           AIE.wire(%[[VAL_167]] : North, %[[VAL_168]] : South)
// CHECK:           AIE.wire(%[[VAL_163]] : East, %[[VAL_169]] : West)
// CHECK:           AIE.wire(%[[VAL_70]] : Core, %[[VAL_169]] : Core)
// CHECK:           AIE.wire(%[[VAL_70]] : DMA, %[[VAL_169]] : DMA)
// CHECK:           AIE.wire(%[[VAL_164]] : East, %[[VAL_170]] : West)
// CHECK:           AIE.wire(%[[VAL_71]] : Core, %[[VAL_170]] : Core)
// CHECK:           AIE.wire(%[[VAL_71]] : DMA, %[[VAL_170]] : DMA)
// CHECK:           AIE.wire(%[[VAL_169]] : North, %[[VAL_170]] : South)
// CHECK:           AIE.wire(%[[VAL_165]] : East, %[[VAL_171]] : West)
// CHECK:           AIE.wire(%[[VAL_72]] : Core, %[[VAL_171]] : Core)
// CHECK:           AIE.wire(%[[VAL_72]] : DMA, %[[VAL_171]] : DMA)
// CHECK:           AIE.wire(%[[VAL_170]] : North, %[[VAL_171]] : South)
// CHECK:           AIE.wire(%[[VAL_166]] : East, %[[VAL_172]] : West)
// CHECK:           AIE.wire(%[[VAL_73]] : Core, %[[VAL_172]] : Core)
// CHECK:           AIE.wire(%[[VAL_73]] : DMA, %[[VAL_172]] : DMA)
// CHECK:           AIE.wire(%[[VAL_171]] : North, %[[VAL_172]] : South)
// CHECK:           AIE.wire(%[[VAL_167]] : East, %[[VAL_173]] : West)
// CHECK:           AIE.wire(%[[VAL_74]] : Core, %[[VAL_173]] : Core)
// CHECK:           AIE.wire(%[[VAL_74]] : DMA, %[[VAL_173]] : DMA)
// CHECK:           AIE.wire(%[[VAL_172]] : North, %[[VAL_173]] : South)
// CHECK:           AIE.wire(%[[VAL_168]] : East, %[[VAL_174]] : West)
// CHECK:           AIE.wire(%[[VAL_75]] : Core, %[[VAL_174]] : Core)
// CHECK:           AIE.wire(%[[VAL_75]] : DMA, %[[VAL_174]] : DMA)
// CHECK:           AIE.wire(%[[VAL_173]] : North, %[[VAL_174]] : South)
// CHECK:           AIE.wire(%[[VAL_169]] : East, %[[VAL_175]] : West)
// CHECK:           AIE.wire(%[[VAL_78]] : Core, %[[VAL_175]] : Core)
// CHECK:           AIE.wire(%[[VAL_78]] : DMA, %[[VAL_175]] : DMA)
// CHECK:           AIE.wire(%[[VAL_170]] : East, %[[VAL_176]] : West)
// CHECK:           AIE.wire(%[[VAL_79]] : Core, %[[VAL_176]] : Core)
// CHECK:           AIE.wire(%[[VAL_79]] : DMA, %[[VAL_176]] : DMA)
// CHECK:           AIE.wire(%[[VAL_175]] : North, %[[VAL_176]] : South)
// CHECK:           AIE.wire(%[[VAL_171]] : East, %[[VAL_177]] : West)
// CHECK:           AIE.wire(%[[VAL_80]] : Core, %[[VAL_177]] : Core)
// CHECK:           AIE.wire(%[[VAL_80]] : DMA, %[[VAL_177]] : DMA)
// CHECK:           AIE.wire(%[[VAL_176]] : North, %[[VAL_177]] : South)
// CHECK:           AIE.wire(%[[VAL_172]] : East, %[[VAL_178]] : West)
// CHECK:           AIE.wire(%[[VAL_81]] : Core, %[[VAL_178]] : Core)
// CHECK:           AIE.wire(%[[VAL_81]] : DMA, %[[VAL_178]] : DMA)
// CHECK:           AIE.wire(%[[VAL_177]] : North, %[[VAL_178]] : South)
// CHECK:           AIE.wire(%[[VAL_175]] : East, %[[VAL_179]] : West)
// CHECK:           AIE.wire(%[[VAL_86]] : Core, %[[VAL_179]] : Core)
// CHECK:           AIE.wire(%[[VAL_86]] : DMA, %[[VAL_179]] : DMA)
// CHECK:           AIE.wire(%[[VAL_176]] : East, %[[VAL_180]] : West)
// CHECK:           AIE.wire(%[[VAL_87]] : Core, %[[VAL_180]] : Core)
// CHECK:           AIE.wire(%[[VAL_87]] : DMA, %[[VAL_180]] : DMA)
// CHECK:           AIE.wire(%[[VAL_179]] : North, %[[VAL_180]] : South)
// CHECK:           AIE.wire(%[[VAL_177]] : East, %[[VAL_181]] : West)
// CHECK:           AIE.wire(%[[VAL_88]] : Core, %[[VAL_181]] : Core)
// CHECK:           AIE.wire(%[[VAL_88]] : DMA, %[[VAL_181]] : DMA)
// CHECK:           AIE.wire(%[[VAL_180]] : North, %[[VAL_181]] : South)
// CHECK:           AIE.wire(%[[VAL_178]] : East, %[[VAL_182]] : West)
// CHECK:           AIE.wire(%[[VAL_89]] : Core, %[[VAL_182]] : Core)
// CHECK:           AIE.wire(%[[VAL_89]] : DMA, %[[VAL_182]] : DMA)
// CHECK:           AIE.wire(%[[VAL_181]] : North, %[[VAL_182]] : South)
// CHECK:           AIE.wire(%[[VAL_208]] : North, %[[VAL_207]] : South)
// CHECK:           AIE.wire(%[[VAL_10]] : DMA, %[[VAL_208]] : DMA)
// CHECK:           AIE.wire(%[[VAL_179]] : East, %[[VAL_183]] : West)
// CHECK:           AIE.wire(%[[VAL_94]] : Core, %[[VAL_183]] : Core)
// CHECK:           AIE.wire(%[[VAL_94]] : DMA, %[[VAL_183]] : DMA)
// CHECK:           AIE.wire(%[[VAL_207]] : North, %[[VAL_183]] : South)
// CHECK:           AIE.wire(%[[VAL_180]] : East, %[[VAL_184]] : West)
// CHECK:           AIE.wire(%[[VAL_95]] : Core, %[[VAL_184]] : Core)
// CHECK:           AIE.wire(%[[VAL_95]] : DMA, %[[VAL_184]] : DMA)
// CHECK:           AIE.wire(%[[VAL_183]] : North, %[[VAL_184]] : South)
// CHECK:           AIE.wire(%[[VAL_181]] : East, %[[VAL_185]] : West)
// CHECK:           AIE.wire(%[[VAL_96]] : Core, %[[VAL_185]] : Core)
// CHECK:           AIE.wire(%[[VAL_96]] : DMA, %[[VAL_185]] : DMA)
// CHECK:           AIE.wire(%[[VAL_184]] : North, %[[VAL_185]] : South)
// CHECK:           AIE.wire(%[[VAL_182]] : East, %[[VAL_186]] : West)
// CHECK:           AIE.wire(%[[VAL_97]] : Core, %[[VAL_186]] : Core)
// CHECK:           AIE.wire(%[[VAL_97]] : DMA, %[[VAL_186]] : DMA)
// CHECK:           AIE.wire(%[[VAL_185]] : North, %[[VAL_186]] : South)
// CHECK:           AIE.wire(%[[VAL_207]] : East, %[[VAL_212]] : West)
// CHECK:           AIE.wire(%[[VAL_213]] : North, %[[VAL_212]] : South)
// CHECK:           AIE.wire(%[[VAL_11]] : DMA, %[[VAL_213]] : DMA)
// CHECK:           AIE.wire(%[[VAL_183]] : East, %[[VAL_187]] : West)
// CHECK:           AIE.wire(%[[VAL_102]] : Core, %[[VAL_187]] : Core)
// CHECK:           AIE.wire(%[[VAL_102]] : DMA, %[[VAL_187]] : DMA)
// CHECK:           AIE.wire(%[[VAL_212]] : North, %[[VAL_187]] : South)
// CHECK:           AIE.wire(%[[VAL_184]] : East, %[[VAL_188]] : West)
// CHECK:           AIE.wire(%[[VAL_103]] : Core, %[[VAL_188]] : Core)
// CHECK:           AIE.wire(%[[VAL_103]] : DMA, %[[VAL_188]] : DMA)
// CHECK:           AIE.wire(%[[VAL_187]] : North, %[[VAL_188]] : South)
// CHECK:           AIE.wire(%[[VAL_185]] : East, %[[VAL_189]] : West)
// CHECK:           AIE.wire(%[[VAL_104]] : Core, %[[VAL_189]] : Core)
// CHECK:           AIE.wire(%[[VAL_104]] : DMA, %[[VAL_189]] : DMA)
// CHECK:           AIE.wire(%[[VAL_188]] : North, %[[VAL_189]] : South)
// CHECK:           AIE.wire(%[[VAL_186]] : East, %[[VAL_190]] : West)
// CHECK:           AIE.wire(%[[VAL_105]] : Core, %[[VAL_190]] : Core)
// CHECK:           AIE.wire(%[[VAL_105]] : DMA, %[[VAL_190]] : DMA)
// CHECK:           AIE.wire(%[[VAL_189]] : North, %[[VAL_190]] : South)
// CHECK:           AIE.wire(%[[VAL_212]] : East, %[[VAL_215]] : West)
// CHECK:           AIE.wire(%[[VAL_187]] : East, %[[VAL_191]] : West)
// CHECK:           AIE.wire(%[[VAL_110]] : Core, %[[VAL_191]] : Core)
// CHECK:           AIE.wire(%[[VAL_110]] : DMA, %[[VAL_191]] : DMA)
// CHECK:           AIE.wire(%[[VAL_215]] : North, %[[VAL_191]] : South)
// CHECK:           AIE.wire(%[[VAL_188]] : East, %[[VAL_192]] : West)
// CHECK:           AIE.wire(%[[VAL_111]] : Core, %[[VAL_192]] : Core)
// CHECK:           AIE.wire(%[[VAL_111]] : DMA, %[[VAL_192]] : DMA)
// CHECK:           AIE.wire(%[[VAL_191]] : North, %[[VAL_192]] : South)
// CHECK:           AIE.wire(%[[VAL_189]] : East, %[[VAL_193]] : West)
// CHECK:           AIE.wire(%[[VAL_112]] : Core, %[[VAL_193]] : Core)
// CHECK:           AIE.wire(%[[VAL_112]] : DMA, %[[VAL_193]] : DMA)
// CHECK:           AIE.wire(%[[VAL_192]] : North, %[[VAL_193]] : South)
// CHECK:           AIE.wire(%[[VAL_190]] : East, %[[VAL_194]] : West)
// CHECK:           AIE.wire(%[[VAL_113]] : Core, %[[VAL_194]] : Core)
// CHECK:           AIE.wire(%[[VAL_113]] : DMA, %[[VAL_194]] : DMA)
// CHECK:           AIE.wire(%[[VAL_193]] : North, %[[VAL_194]] : South)
// CHECK:           AIE.wire(%[[VAL_114]] : Core, %[[VAL_195]] : Core)
// CHECK:           AIE.wire(%[[VAL_114]] : DMA, %[[VAL_195]] : DMA)
// CHECK:           AIE.wire(%[[VAL_194]] : North, %[[VAL_195]] : South)
// CHECK:           AIE.wire(%[[VAL_215]] : East, %[[VAL_216]] : West)
// CHECK:           AIE.wire(%[[VAL_191]] : East, %[[VAL_196]] : West)
// CHECK:           AIE.wire(%[[VAL_119]] : Core, %[[VAL_196]] : Core)
// CHECK:           AIE.wire(%[[VAL_119]] : DMA, %[[VAL_196]] : DMA)
// CHECK:           AIE.wire(%[[VAL_216]] : North, %[[VAL_196]] : South)
// CHECK:           AIE.wire(%[[VAL_192]] : East, %[[VAL_197]] : West)
// CHECK:           AIE.wire(%[[VAL_120]] : Core, %[[VAL_197]] : Core)
// CHECK:           AIE.wire(%[[VAL_120]] : DMA, %[[VAL_197]] : DMA)
// CHECK:           AIE.wire(%[[VAL_196]] : North, %[[VAL_197]] : South)
// CHECK:           AIE.wire(%[[VAL_193]] : East, %[[VAL_198]] : West)
// CHECK:           AIE.wire(%[[VAL_121]] : Core, %[[VAL_198]] : Core)
// CHECK:           AIE.wire(%[[VAL_121]] : DMA, %[[VAL_198]] : DMA)
// CHECK:           AIE.wire(%[[VAL_197]] : North, %[[VAL_198]] : South)
// CHECK:           AIE.wire(%[[VAL_194]] : East, %[[VAL_199]] : West)
// CHECK:           AIE.wire(%[[VAL_122]] : Core, %[[VAL_199]] : Core)
// CHECK:           AIE.wire(%[[VAL_122]] : DMA, %[[VAL_199]] : DMA)
// CHECK:           AIE.wire(%[[VAL_198]] : North, %[[VAL_199]] : South)
// CHECK:           AIE.wire(%[[VAL_195]] : East, %[[VAL_200]] : West)
// CHECK:           AIE.wire(%[[VAL_123]] : Core, %[[VAL_200]] : Core)
// CHECK:           AIE.wire(%[[VAL_123]] : DMA, %[[VAL_200]] : DMA)
// CHECK:           AIE.wire(%[[VAL_199]] : North, %[[VAL_200]] : South)
// CHECK:           AIE.wire(%[[VAL_197]] : East, %[[VAL_223]] : West)
// CHECK:           AIE.wire(%[[VAL_128]] : Core, %[[VAL_223]] : Core)
// CHECK:           AIE.wire(%[[VAL_128]] : DMA, %[[VAL_223]] : DMA)
// CHECK:           AIE.wire(%[[VAL_198]] : East, %[[VAL_230]] : West)
// CHECK:           AIE.wire(%[[VAL_129]] : Core, %[[VAL_230]] : Core)
// CHECK:           AIE.wire(%[[VAL_129]] : DMA, %[[VAL_230]] : DMA)
// CHECK:           AIE.wire(%[[VAL_223]] : North, %[[VAL_230]] : South)
// CHECK:           AIE.wire(%[[VAL_199]] : East, %[[VAL_231]] : West)
// CHECK:           AIE.wire(%[[VAL_130]] : Core, %[[VAL_231]] : Core)
// CHECK:           AIE.wire(%[[VAL_130]] : DMA, %[[VAL_231]] : DMA)
// CHECK:           AIE.wire(%[[VAL_230]] : North, %[[VAL_231]] : South)
// CHECK:           AIE.wire(%[[VAL_200]] : East, %[[VAL_232]] : West)
// CHECK:           AIE.wire(%[[VAL_131]] : Core, %[[VAL_232]] : Core)
// CHECK:           AIE.wire(%[[VAL_131]] : DMA, %[[VAL_232]] : DMA)
// CHECK:           AIE.wire(%[[VAL_231]] : North, %[[VAL_232]] : South)
// CHECK:           AIE.wire(%[[VAL_223]] : East, %[[VAL_225]] : West)
// CHECK:           AIE.wire(%[[VAL_224]] : Core, %[[VAL_225]] : Core)
// CHECK:           AIE.wire(%[[VAL_224]] : DMA, %[[VAL_225]] : DMA)
// CHECK:           AIE.wire(%[[VAL_225]] : East, %[[VAL_227]] : West)
// CHECK:           AIE.wire(%[[VAL_226]] : Core, %[[VAL_227]] : Core)
// CHECK:           AIE.wire(%[[VAL_226]] : DMA, %[[VAL_227]] : DMA)
// CHECK:           AIE.wire(%[[VAL_221]] : Core, %[[VAL_222]] : Core)
// CHECK:           AIE.wire(%[[VAL_221]] : DMA, %[[VAL_222]] : DMA)
// CHECK:           AIE.wire(%[[VAL_218]] : North, %[[VAL_222]] : South)
// CHECK:           AIE.wire(%[[VAL_227]] : East, %[[VAL_229]] : West)
// CHECK:           AIE.wire(%[[VAL_228]] : Core, %[[VAL_229]] : Core)
// CHECK:           AIE.wire(%[[VAL_228]] : DMA, %[[VAL_229]] : DMA)
// CHECK:           AIE.wire(%[[VAL_222]] : North, %[[VAL_229]] : South)
// CHECK:           AIE.wire(%[[VAL_218]] : East, %[[VAL_219]] : West)
// CHECK:           AIE.wire(%[[VAL_220]] : North, %[[VAL_219]] : South)
// CHECK:           AIE.wire(%[[VAL_12]] : DMA, %[[VAL_220]] : DMA)
// CHECK:         }

//
//
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
		%t05 = AIE.tile(0, 5)
		%t06 = AIE.tile(0, 6)
		%t07 = AIE.tile(0, 7)
		%t08 = AIE.tile(0, 8)
		%t11 = AIE.tile(1, 1)
		%t12 = AIE.tile(1, 2)
		%t13 = AIE.tile(1, 3)
		%t14 = AIE.tile(1, 4)
		%t15 = AIE.tile(1, 5)
		%t16 = AIE.tile(1, 6)
		%t17 = AIE.tile(1, 7)
		%t18 = AIE.tile(1, 8)
		%t21 = AIE.tile(2, 1)
		%t22 = AIE.tile(2, 2)
		%t23 = AIE.tile(2, 3)
		%t24 = AIE.tile(2, 4)
		%t25 = AIE.tile(2, 5)
		%t26 = AIE.tile(2, 6)
		%t27 = AIE.tile(2, 7)
		%t28 = AIE.tile(2, 8)
		%t31 = AIE.tile(3, 1)
		%t32 = AIE.tile(3, 2)
		%t33 = AIE.tile(3, 3)
		%t34 = AIE.tile(3, 4)
		%t35 = AIE.tile(3, 5)
		%t36 = AIE.tile(3, 6)
		%t37 = AIE.tile(3, 7)
		%t38 = AIE.tile(3, 8)
		%t41 = AIE.tile(4, 1)
		%t42 = AIE.tile(4, 2)
		%t43 = AIE.tile(4, 3)
		%t44 = AIE.tile(4, 4)
		%t45 = AIE.tile(4, 5)
		%t46 = AIE.tile(4, 6)
		%t47 = AIE.tile(4, 7)
		%t48 = AIE.tile(4, 8)
		%t51 = AIE.tile(5, 1)
		%t52 = AIE.tile(5, 2)
		%t53 = AIE.tile(5, 3)
		%t54 = AIE.tile(5, 4)
		%t55 = AIE.tile(5, 5)
		%t56 = AIE.tile(5, 6)
		%t57 = AIE.tile(5, 7)
		%t58 = AIE.tile(5, 8)
		%t61 = AIE.tile(6, 1)
		%t62 = AIE.tile(6, 2)
		%t63 = AIE.tile(6, 3)
		%t64 = AIE.tile(6, 4)
		%t65 = AIE.tile(6, 5)
		%t66 = AIE.tile(6, 6)
		%t67 = AIE.tile(6, 7)
		%t68 = AIE.tile(6, 8)
		%t71 = AIE.tile(7, 1)
		%t72 = AIE.tile(7, 2)
		%t73 = AIE.tile(7, 3)
		%t74 = AIE.tile(7, 4)
		%t75 = AIE.tile(7, 5)
		%t76 = AIE.tile(7, 6)
		%t77 = AIE.tile(7, 7)
		%t78 = AIE.tile(7, 8)
		%t81 = AIE.tile(8, 1)
		%t82 = AIE.tile(8, 2)
		%t83 = AIE.tile(8, 3)
		%t84 = AIE.tile(8, 4)
		%t85 = AIE.tile(8, 5)
		%t86 = AIE.tile(8, 6)
		%t87 = AIE.tile(8, 7)
		%t88 = AIE.tile(8, 8)
		%t91 = AIE.tile(9, 1)
		%t92 = AIE.tile(9, 2)
		%t93 = AIE.tile(9, 3)
		%t94 = AIE.tile(9, 4)
		%t95 = AIE.tile(9, 5)
		%t96 = AIE.tile(9, 6)
		%t97 = AIE.tile(9, 7)
		%t98 = AIE.tile(9, 8)
		%t101 = AIE.tile(10, 1)
		%t102 = AIE.tile(10, 2)
		%t103 = AIE.tile(10, 3)
		%t104 = AIE.tile(10, 4)
		%t105 = AIE.tile(10, 5)
		%t106 = AIE.tile(10, 6)
		%t107 = AIE.tile(10, 7)
		%t108 = AIE.tile(10, 8)
		%t111 = AIE.tile(11, 1)
		%t112 = AIE.tile(11, 2)
		%t113 = AIE.tile(11, 3)
		%t114 = AIE.tile(11, 4)
		%t115 = AIE.tile(11, 5)
		%t116 = AIE.tile(11, 6)
		%t117 = AIE.tile(11, 7)
		%t118 = AIE.tile(11, 8)
		%t121 = AIE.tile(12, 1)
		%t122 = AIE.tile(12, 2)
		%t123 = AIE.tile(12, 3)
		%t124 = AIE.tile(12, 4)
		%t125 = AIE.tile(12, 5)
		%t126 = AIE.tile(12, 6)
		%t127 = AIE.tile(12, 7)
		%t128 = AIE.tile(12, 8)
		%t130 = AIE.tile(13, 0)
		%t131 = AIE.tile(13, 1)
		%t132 = AIE.tile(13, 2)
		%t133 = AIE.tile(13, 3)
		%t134 = AIE.tile(13, 4)
		%t135 = AIE.tile(13, 5)
		%t136 = AIE.tile(13, 6)
		%t137 = AIE.tile(13, 7)
		%t138 = AIE.tile(13, 8)
		%t141 = AIE.tile(14, 1)
		%t142 = AIE.tile(14, 2)
		%t143 = AIE.tile(14, 3)
		%t144 = AIE.tile(14, 4)
		%t145 = AIE.tile(14, 5)
		%t146 = AIE.tile(14, 6)
		%t147 = AIE.tile(14, 7)
		%t148 = AIE.tile(14, 8)

		%sb01 = AIE.switchbox(%t01) {
		}
		%sb02 = AIE.switchbox(%t02) {
		}
		%sb03 = AIE.switchbox(%t03) {
		}
		%sb04 = AIE.switchbox(%t04) {
		}
		%sb11 = AIE.switchbox(%t11) {
		}
		%sb12 = AIE.switchbox(%t12) {
		}
		%sb13 = AIE.switchbox(%t13) {
		}
		%sb14 = AIE.switchbox(%t14) {
		}
		%sb21 = AIE.switchbox(%t21) {
		}
		%sb22 = AIE.switchbox(%t22) {
		}
		%sb23 = AIE.switchbox(%t23) {
		}
		%sb24 = AIE.switchbox(%t24) {
			AIE.connect<East : 0, North : 0>
		}
		%sb25 = AIE.switchbox(%t25) {
			AIE.connect<South: 0, Core : 0>
			AIE.connect<DMA : 0, East : 0>
		}
		%sb31 = AIE.switchbox(%t31) {
			AIE.connect<South : 0, DMA : 0>
			AIE.connect<Core : 0, North: 0>
		}
		%sb32 = AIE.switchbox(%t32) {
			AIE.connect<South : 0, North : 0>
		}
		%sb33 = AIE.switchbox(%t33) {
			AIE.connect<South : 0, North : 0>
		}
		%sb34 = AIE.switchbox(%t34) {
			AIE.connect<South : 0, West : 0>
		}
		%sb35 = AIE.switchbox(%t35) {
			AIE.connect<West : 0, East : 0>
		}
		%sb41 = AIE.switchbox(%t41) {
		}
		%sb42 = AIE.switchbox(%t42) {
		}
		%sb43 = AIE.switchbox(%t43) {
		}
		%sb44 = AIE.switchbox(%t44) {
		}
		%sb51 = AIE.switchbox(%t51) {
		}
		%sb52 = AIE.switchbox(%t52) {
		}
		%sb53 = AIE.switchbox(%t53) {
		}
		%sb54 = AIE.switchbox(%t54) {
		}
		%sb55 = AIE.switchbox(%t55) {
		}
		%sb56 = AIE.switchbox(%t56) {
			AIE.connect<East : 0, West : 0>
		}
		%sb61 = AIE.switchbox(%t61) {
		}
		%sb62 = AIE.switchbox(%t62) {
		}
		%sb63 = AIE.switchbox(%t63) {
		}
		%sb64 = AIE.switchbox(%t64) {
		}
		%sb65 = AIE.switchbox(%t65) {
		}
		%sb66 = AIE.switchbox(%t66) {
			AIE.connect<East : 0, Core : 0>
			AIE.connect<DMA : 0, West : 0>
		}
		%sb71 = AIE.switchbox(%t71) {
		}
		%sb72 = AIE.switchbox(%t72) {
		}
		%sb73 = AIE.switchbox(%t73) {
			AIE.connect<East : 0, DMA : 0>
			AIE.connect<Core : 0, North : 0>
		}
		%sb74 = AIE.switchbox(%t74) {
			AIE.connect<South : 0, North : 0>
		}
		%sb75 = AIE.switchbox(%t75) {
			AIE.connect<South : 0, North : 0>
		}
		%sb76 = AIE.switchbox(%t76) {
			AIE.connect<South : 0, West: 0>
		}
		%sb81 = AIE.switchbox(%t81) {
		}
		%sb82 = AIE.switchbox(%t82) {
		}
		%sb83 = AIE.switchbox(%t83) {
			AIE.connect<East : 0, West : 0>
		}
		%sb84 = AIE.switchbox(%t84) {
		}
		%sb91 = AIE.switchbox(%t91) {
		}
		%sb92 = AIE.switchbox(%t92) {
		}
		%sb93 = AIE.switchbox(%t93) {
		}
		%sb94 = AIE.switchbox(%t94) {
		}
		%sb101 = AIE.switchbox(%t101) {
		}
		%sb102 = AIE.switchbox(%t102) {
		}
		%sb103 = AIE.switchbox(%t103) {
		}
		%sb104 = AIE.switchbox(%t104) {
		}
		%sb111 = AIE.switchbox(%t111) {
		}
		%sb112 = AIE.switchbox(%t112) {
		}
		%sb113 = AIE.switchbox(%t113) {
		}
		%sb114 = AIE.switchbox(%t114) {
		}
		%sb121 = AIE.switchbox(%t121) {
		}
		%sb122 = AIE.switchbox(%t122) {
		}
		%sb123 = AIE.switchbox(%t123) {
		}
		%sb124 = AIE.switchbox(%t124) {
		}
		%sb125 = AIE.switchbox(%t125) {
			AIE.connect<East : 0, Core : 0>
			AIE.connect<DMA : 0, East : 0>
		}
		%sb131 = AIE.switchbox(%t131) {
			AIE.connect<South : 0, North : 0>
		}
		%sb132 = AIE.switchbox(%t132) {
			AIE.connect<South : 0, North : 0>
		}
		%sb133 = AIE.switchbox(%t133) {
			AIE.connect<South : 0, DMA : 0>
			AIE.connect<Core : 0, North: 0>
		}
		%sb134 = AIE.switchbox(%t134) {
			AIE.connect<South : 0, North : 0>
		}
		%sb135 = AIE.switchbox(%t135) {
			AIE.connect<South : 0, West : 0>
			AIE.connect<West : 0, East : 0>
		}

		AIE.flow(%t30, DMA : 0, %t30, North: 0)
		AIE.flow(%t45, West: 0, %t60, DMA : 0)

		AIE.flow(%t100, DMA : 0, %t93, West: 0)
		AIE.flow(%t46, East: 0, %t20, DMA : 0)

		AIE.flow(%t110, DMA : 0, %t130, North: 0)
		AIE.flow(%t145, West: 0, %t180, DMA : 0)
	}
}
