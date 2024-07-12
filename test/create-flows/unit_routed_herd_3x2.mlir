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

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:           }
// CHECK:           %[[VAL_2:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_3:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK:           }
// CHECK:           %[[VAL_4:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_5:.*]] = aie.tile(3, 0)
// CHECK:           %[[VAL_6:.*]] = aie.tile(4, 0)
// CHECK:           %[[VAL_7:.*]] = aie.switchbox(%[[VAL_6]]) {
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = aie.tile(5, 0)
// CHECK:           %[[VAL_9:.*]] = aie.tile(6, 0)
// CHECK:           %[[VAL_10:.*]] = aie.tile(7, 0)
// CHECK:           %[[VAL_11:.*]] = aie.switchbox(%[[VAL_10]]) {
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = aie.tile(8, 0)
// CHECK:           %[[VAL_13:.*]] = aie.switchbox(%[[VAL_12]]) {
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = aie.tile(9, 0)
// CHECK:           %[[VAL_15:.*]] = aie.switchbox(%[[VAL_14]]) {
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = aie.tile(10, 0)
// CHECK:           %[[VAL_17:.*]] = aie.tile(11, 0)
// CHECK:           %[[VAL_18:.*]] = aie.tile(18, 0)
// CHECK:           %[[VAL_19:.*]] = aie.tile(19, 0)
// CHECK:           %[[VAL_20:.*]] = aie.switchbox(%[[VAL_19]]) {
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_22:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_23:.*]] = aie.tile(0, 3)
// CHECK:           %[[VAL_24:.*]] = aie.tile(0, 4)
// CHECK:           %[[VAL_25:.*]] = aie.tile(0, 5)
// CHECK:           %[[VAL_26:.*]] = aie.switchbox(%[[VAL_25]]) {
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = aie.tile(0, 6)
// CHECK:           %[[VAL_28:.*]] = aie.switchbox(%[[VAL_27]]) {
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = aie.tile(0, 7)
// CHECK:           %[[VAL_30:.*]] = aie.switchbox(%[[VAL_29]]) {
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = aie.tile(0, 8)
// CHECK:           %[[VAL_32:.*]] = aie.switchbox(%[[VAL_31]]) {
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_34:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_35:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_36:.*]] = aie.tile(1, 4)
// CHECK:           %[[VAL_37:.*]] = aie.tile(1, 5)
// CHECK:           %[[VAL_38:.*]] = aie.switchbox(%[[VAL_37]]) {
// CHECK:           }
// CHECK:           %[[VAL_39:.*]] = aie.tile(1, 6)
// CHECK:           %[[VAL_40:.*]] = aie.switchbox(%[[VAL_39]]) {
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = aie.tile(1, 7)
// CHECK:           %[[VAL_42:.*]] = aie.switchbox(%[[VAL_41]]) {
// CHECK:           }
// CHECK:           %[[VAL_43:.*]] = aie.tile(1, 8)
// CHECK:           %[[VAL_44:.*]] = aie.switchbox(%[[VAL_43]]) {
// CHECK:           }
// CHECK:           %[[VAL_45:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_46:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_47:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_48:.*]] = aie.tile(2, 4)
// CHECK:           %[[VAL_49:.*]] = aie.tile(2, 5)
// CHECK:           %[[VAL_50:.*]] = aie.tile(2, 6)
// CHECK:           %[[VAL_51:.*]] = aie.switchbox(%[[VAL_50]]) {
// CHECK:           }
// CHECK:           %[[VAL_52:.*]] = aie.tile(2, 7)
// CHECK:           %[[VAL_53:.*]] = aie.switchbox(%[[VAL_52]]) {
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = aie.tile(2, 8)
// CHECK:           %[[VAL_55:.*]] = aie.switchbox(%[[VAL_54]]) {
// CHECK:           }
// CHECK:           %[[VAL_56:.*]] = aie.tile(3, 1)
// CHECK:           %[[VAL_57:.*]] = aie.tile(3, 2)
// CHECK:           %[[VAL_58:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_59:.*]] = aie.tile(3, 4)
// CHECK:           %[[VAL_60:.*]] = aie.tile(3, 5)
// CHECK:           %[[VAL_61:.*]] = aie.tile(3, 6)
// CHECK:           %[[VAL_62:.*]] = aie.switchbox(%[[VAL_61]]) {
// CHECK:           }
// CHECK:           %[[VAL_63:.*]] = aie.tile(3, 7)
// CHECK:           %[[VAL_64:.*]] = aie.switchbox(%[[VAL_63]]) {
// CHECK:           }
// CHECK:           %[[VAL_65:.*]] = aie.tile(3, 8)
// CHECK:           %[[VAL_66:.*]] = aie.switchbox(%[[VAL_65]]) {
// CHECK:           }
// CHECK:           %[[VAL_67:.*]] = aie.tile(4, 1)
// CHECK:           %[[VAL_68:.*]] = aie.tile(4, 2)
// CHECK:           %[[VAL_69:.*]] = aie.tile(4, 3)
// CHECK:           %[[VAL_70:.*]] = aie.tile(4, 4)
// CHECK:           %[[VAL_71:.*]] = aie.tile(4, 5)
// CHECK:           %[[VAL_72:.*]] = aie.tile(4, 6)
// CHECK:           %[[VAL_73:.*]] = aie.tile(4, 7)
// CHECK:           %[[VAL_74:.*]] = aie.switchbox(%[[VAL_73]]) {
// CHECK:           }
// CHECK:           %[[VAL_75:.*]] = aie.tile(4, 8)
// CHECK:           %[[VAL_76:.*]] = aie.switchbox(%[[VAL_75]]) {
// CHECK:           }
// CHECK:           %[[VAL_77:.*]] = aie.tile(5, 1)
// CHECK:           %[[VAL_78:.*]] = aie.tile(5, 2)
// CHECK:           %[[VAL_79:.*]] = aie.tile(5, 3)
// CHECK:           %[[VAL_80:.*]] = aie.tile(5, 4)
// CHECK:           %[[VAL_81:.*]] = aie.tile(5, 5)
// CHECK:           %[[VAL_82:.*]] = aie.tile(5, 6)
// CHECK:           %[[VAL_83:.*]] = aie.tile(5, 7)
// CHECK:           %[[VAL_84:.*]] = aie.switchbox(%[[VAL_83]]) {
// CHECK:           }
// CHECK:           %[[VAL_85:.*]] = aie.tile(5, 8)
// CHECK:           %[[VAL_86:.*]] = aie.switchbox(%[[VAL_85]]) {
// CHECK:           }
// CHECK:           %[[VAL_87:.*]] = aie.tile(6, 1)
// CHECK:           %[[VAL_88:.*]] = aie.tile(6, 2)
// CHECK:           %[[VAL_89:.*]] = aie.tile(6, 3)
// CHECK:           %[[VAL_90:.*]] = aie.tile(6, 4)
// CHECK:           %[[VAL_91:.*]] = aie.tile(6, 5)
// CHECK:           %[[VAL_92:.*]] = aie.tile(6, 6)
// CHECK:           %[[VAL_93:.*]] = aie.tile(6, 7)
// CHECK:           %[[VAL_94:.*]] = aie.switchbox(%[[VAL_93]]) {
// CHECK:           }
// CHECK:           %[[VAL_95:.*]] = aie.tile(6, 8)
// CHECK:           %[[VAL_96:.*]] = aie.switchbox(%[[VAL_95]]) {
// CHECK:           }
// CHECK:           %[[VAL_97:.*]] = aie.tile(7, 1)
// CHECK:           %[[VAL_98:.*]] = aie.tile(7, 2)
// CHECK:           %[[VAL_99:.*]] = aie.tile(7, 3)
// CHECK:           %[[VAL_100:.*]] = aie.tile(7, 4)
// CHECK:           %[[VAL_101:.*]] = aie.tile(7, 5)
// CHECK:           %[[VAL_102:.*]] = aie.tile(7, 6)
// CHECK:           %[[VAL_103:.*]] = aie.tile(7, 7)
// CHECK:           %[[VAL_104:.*]] = aie.switchbox(%[[VAL_103]]) {
// CHECK:           }
// CHECK:           %[[VAL_105:.*]] = aie.tile(7, 8)
// CHECK:           %[[VAL_106:.*]] = aie.switchbox(%[[VAL_105]]) {
// CHECK:           }
// CHECK:           %[[VAL_107:.*]] = aie.tile(8, 1)
// CHECK:           %[[VAL_108:.*]] = aie.tile(8, 2)
// CHECK:           %[[VAL_109:.*]] = aie.tile(8, 3)
// CHECK:           %[[VAL_110:.*]] = aie.tile(8, 4)
// CHECK:           %[[VAL_111:.*]] = aie.tile(8, 5)
// CHECK:           %[[VAL_112:.*]] = aie.switchbox(%[[VAL_111]]) {
// CHECK:           }
// CHECK:           %[[VAL_113:.*]] = aie.tile(8, 6)
// CHECK:           %[[VAL_114:.*]] = aie.switchbox(%[[VAL_113]]) {
// CHECK:           }
// CHECK:           %[[VAL_115:.*]] = aie.tile(8, 7)
// CHECK:           %[[VAL_116:.*]] = aie.switchbox(%[[VAL_115]]) {
// CHECK:           }
// CHECK:           %[[VAL_117:.*]] = aie.tile(8, 8)
// CHECK:           %[[VAL_118:.*]] = aie.switchbox(%[[VAL_117]]) {
// CHECK:           }
// CHECK:           %[[VAL_119:.*]] = aie.tile(9, 1)
// CHECK:           %[[VAL_120:.*]] = aie.tile(9, 2)
// CHECK:           %[[VAL_121:.*]] = aie.tile(9, 3)
// CHECK:           %[[VAL_122:.*]] = aie.tile(9, 4)
// CHECK:           %[[VAL_123:.*]] = aie.tile(9, 5)
// CHECK:           %[[VAL_124:.*]] = aie.switchbox(%[[VAL_123]]) {
// CHECK:           }
// CHECK:           %[[VAL_125:.*]] = aie.tile(9, 6)
// CHECK:           %[[VAL_126:.*]] = aie.switchbox(%[[VAL_125]]) {
// CHECK:           }
// CHECK:           %[[VAL_127:.*]] = aie.tile(9, 7)
// CHECK:           %[[VAL_128:.*]] = aie.switchbox(%[[VAL_127]]) {
// CHECK:           }
// CHECK:           %[[VAL_129:.*]] = aie.tile(9, 8)
// CHECK:           %[[VAL_130:.*]] = aie.switchbox(%[[VAL_129]]) {
// CHECK:           }
// CHECK:           %[[VAL_131:.*]] = aie.tile(10, 1)
// CHECK:           %[[VAL_132:.*]] = aie.tile(10, 2)
// CHECK:           %[[VAL_133:.*]] = aie.tile(10, 3)
// CHECK:           %[[VAL_134:.*]] = aie.tile(10, 4)
// CHECK:           %[[VAL_135:.*]] = aie.tile(10, 5)
// CHECK:           %[[VAL_136:.*]] = aie.switchbox(%[[VAL_135]]) {
// CHECK:           }
// CHECK:           %[[VAL_137:.*]] = aie.tile(10, 6)
// CHECK:           %[[VAL_138:.*]] = aie.switchbox(%[[VAL_137]]) {
// CHECK:           }
// CHECK:           %[[VAL_139:.*]] = aie.tile(10, 7)
// CHECK:           %[[VAL_140:.*]] = aie.switchbox(%[[VAL_139]]) {
// CHECK:           }
// CHECK:           %[[VAL_141:.*]] = aie.tile(10, 8)
// CHECK:           %[[VAL_142:.*]] = aie.switchbox(%[[VAL_141]]) {
// CHECK:           }
// CHECK:           %[[VAL_143:.*]] = aie.tile(11, 1)
// CHECK:           %[[VAL_144:.*]] = aie.tile(11, 2)
// CHECK:           %[[VAL_145:.*]] = aie.tile(11, 3)
// CHECK:           %[[VAL_146:.*]] = aie.tile(11, 4)
// CHECK:           %[[VAL_147:.*]] = aie.tile(11, 5)
// CHECK:           %[[VAL_148:.*]] = aie.switchbox(%[[VAL_147]]) {
// CHECK:           }
// CHECK:           %[[VAL_149:.*]] = aie.tile(11, 6)
// CHECK:           %[[VAL_150:.*]] = aie.switchbox(%[[VAL_149]]) {
// CHECK:           }
// CHECK:           %[[VAL_151:.*]] = aie.tile(11, 7)
// CHECK:           %[[VAL_152:.*]] = aie.switchbox(%[[VAL_151]]) {
// CHECK:           }
// CHECK:           %[[VAL_153:.*]] = aie.tile(11, 8)
// CHECK:           %[[VAL_154:.*]] = aie.switchbox(%[[VAL_153]]) {
// CHECK:           }
// CHECK:           %[[VAL_155:.*]] = aie.tile(12, 1)
// CHECK:           %[[VAL_156:.*]] = aie.tile(12, 2)
// CHECK:           %[[VAL_157:.*]] = aie.tile(12, 3)
// CHECK:           %[[VAL_158:.*]] = aie.tile(12, 4)
// CHECK:           %[[VAL_159:.*]] = aie.tile(12, 5)
// CHECK:           %[[VAL_160:.*]] = aie.tile(12, 6)
// CHECK:           %[[VAL_161:.*]] = aie.switchbox(%[[VAL_160]]) {
// CHECK:           }
// CHECK:           %[[VAL_162:.*]] = aie.tile(12, 7)
// CHECK:           %[[VAL_163:.*]] = aie.switchbox(%[[VAL_162]]) {
// CHECK:           }
// CHECK:           %[[VAL_164:.*]] = aie.tile(12, 8)
// CHECK:           %[[VAL_165:.*]] = aie.switchbox(%[[VAL_164]]) {
// CHECK:           }
// CHECK:           %[[VAL_166:.*]] = aie.tile(13, 0)
// CHECK:           %[[VAL_167:.*]] = aie.tile(13, 1)
// CHECK:           %[[VAL_168:.*]] = aie.tile(13, 2)
// CHECK:           %[[VAL_169:.*]] = aie.tile(13, 3)
// CHECK:           %[[VAL_170:.*]] = aie.tile(13, 4)
// CHECK:           %[[VAL_171:.*]] = aie.tile(13, 5)
// CHECK:           %[[VAL_172:.*]] = aie.tile(13, 6)
// CHECK:           %[[VAL_173:.*]] = aie.switchbox(%[[VAL_172]]) {
// CHECK:           }
// CHECK:           %[[VAL_174:.*]] = aie.tile(13, 7)
// CHECK:           %[[VAL_175:.*]] = aie.switchbox(%[[VAL_174]]) {
// CHECK:           }
// CHECK:           %[[VAL_176:.*]] = aie.tile(13, 8)
// CHECK:           %[[VAL_177:.*]] = aie.switchbox(%[[VAL_176]]) {
// CHECK:           }
// CHECK:           %[[VAL_178:.*]] = aie.tile(14, 1)
// CHECK:           %[[VAL_179:.*]] = aie.switchbox(%[[VAL_178]]) {
// CHECK:           }
// CHECK:           %[[VAL_180:.*]] = aie.tile(14, 2)
// CHECK:           %[[VAL_181:.*]] = aie.tile(14, 3)
// CHECK:           %[[VAL_182:.*]] = aie.tile(14, 4)
// CHECK:           %[[VAL_183:.*]] = aie.tile(14, 5)
// CHECK:           %[[VAL_184:.*]] = aie.tile(14, 6)
// CHECK:           %[[VAL_185:.*]] = aie.switchbox(%[[VAL_184]]) {
// CHECK:           }
// CHECK:           %[[VAL_186:.*]] = aie.tile(14, 7)
// CHECK:           %[[VAL_187:.*]] = aie.switchbox(%[[VAL_186]]) {
// CHECK:           }
// CHECK:           %[[VAL_188:.*]] = aie.tile(14, 8)
// CHECK:           %[[VAL_189:.*]] = aie.switchbox(%[[VAL_188]]) {
// CHECK:           }
// CHECK:           %[[VAL_190:.*]] = aie.switchbox(%[[VAL_21]]) {
// CHECK:           }
// CHECK:           %[[VAL_191:.*]] = aie.switchbox(%[[VAL_22]]) {
// CHECK:           }
// CHECK:           %[[VAL_192:.*]] = aie.switchbox(%[[VAL_23]]) {
// CHECK:           }
// CHECK:           %[[VAL_193:.*]] = aie.switchbox(%[[VAL_24]]) {
// CHECK:           }
// CHECK:           %[[VAL_194:.*]] = aie.switchbox(%[[VAL_33]]) {
// CHECK:           }
// CHECK:           %[[VAL_195:.*]] = aie.switchbox(%[[VAL_34]]) {
// CHECK:           }
// CHECK:           %[[VAL_196:.*]] = aie.switchbox(%[[VAL_35]]) {
// CHECK:           }
// CHECK:           %[[VAL_197:.*]] = aie.switchbox(%[[VAL_36]]) {
// CHECK:           }
// CHECK:           %[[VAL_198:.*]] = aie.switchbox(%[[VAL_45]]) {
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_199:.*]] = aie.switchbox(%[[VAL_46]]) {
// CHECK:           }
// CHECK:           %[[VAL_200:.*]] = aie.switchbox(%[[VAL_47]]) {
// CHECK:           }
// CHECK:           %[[VAL_201:.*]] = aie.switchbox(%[[VAL_48]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_202:.*]] = aie.switchbox(%[[VAL_49]]) {
// CHECK:             aie.connect<South : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_203:.*]] = aie.switchbox(%[[VAL_56]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_204:.*]] = aie.switchbox(%[[VAL_57]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_205:.*]] = aie.switchbox(%[[VAL_58]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_206:.*]] = aie.switchbox(%[[VAL_59]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_207:.*]] = aie.switchbox(%[[VAL_60]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_208:.*]] = aie.switchbox(%[[VAL_67]]) {
// CHECK:           }
// CHECK:           %[[VAL_209:.*]] = aie.switchbox(%[[VAL_68]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_210:.*]] = aie.switchbox(%[[VAL_69]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_211:.*]] = aie.switchbox(%[[VAL_70]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<North : 1, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_212:.*]] = aie.switchbox(%[[VAL_77]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_213:.*]] = aie.switchbox(%[[VAL_78]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_214:.*]] = aie.switchbox(%[[VAL_79]]) {
// CHECK:           }
// CHECK:           %[[VAL_215:.*]] = aie.switchbox(%[[VAL_80]]) {
// CHECK:           }
// CHECK:           %[[VAL_216:.*]] = aie.switchbox(%[[VAL_81]]) {
// CHECK:           }
// CHECK:           %[[VAL_217:.*]] = aie.switchbox(%[[VAL_82]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_218:.*]] = aie.switchbox(%[[VAL_87]]) {
// CHECK:           }
// CHECK:           %[[VAL_219:.*]] = aie.switchbox(%[[VAL_88]]) {
// CHECK:           }
// CHECK:           %[[VAL_220:.*]] = aie.switchbox(%[[VAL_89]]) {
// CHECK:           }
// CHECK:           %[[VAL_221:.*]] = aie.switchbox(%[[VAL_90]]) {
// CHECK:           }
// CHECK:           %[[VAL_222:.*]] = aie.switchbox(%[[VAL_91]]) {
// CHECK:           }
// CHECK:           %[[VAL_223:.*]] = aie.switchbox(%[[VAL_92]]) {
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_224:.*]] = aie.switchbox(%[[VAL_97]]) {
// CHECK:           }
// CHECK:           %[[VAL_225:.*]] = aie.switchbox(%[[VAL_98]]) {
// CHECK:           }
// CHECK:           %[[VAL_226:.*]] = aie.switchbox(%[[VAL_99]]) {
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_227:.*]] = aie.switchbox(%[[VAL_100]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_228:.*]] = aie.switchbox(%[[VAL_101]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_229:.*]] = aie.switchbox(%[[VAL_102]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_230:.*]] = aie.switchbox(%[[VAL_107]]) {
// CHECK:           }
// CHECK:           %[[VAL_231:.*]] = aie.switchbox(%[[VAL_108]]) {
// CHECK:           }
// CHECK:           %[[VAL_232:.*]] = aie.switchbox(%[[VAL_109]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_233:.*]] = aie.switchbox(%[[VAL_110]]) {
// CHECK:           }
// CHECK:           %[[VAL_234:.*]] = aie.switchbox(%[[VAL_119]]) {
// CHECK:           }
// CHECK:           %[[VAL_235:.*]] = aie.switchbox(%[[VAL_120]]) {
// CHECK:           }
// CHECK:           %[[VAL_236:.*]] = aie.switchbox(%[[VAL_121]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_237:.*]] = aie.switchbox(%[[VAL_122]]) {
// CHECK:           }
// CHECK:           %[[VAL_238:.*]] = aie.switchbox(%[[VAL_131]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_239:.*]] = aie.switchbox(%[[VAL_132]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_240:.*]] = aie.switchbox(%[[VAL_133]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_241:.*]] = aie.switchbox(%[[VAL_134]]) {
// CHECK:           }
// CHECK:           %[[VAL_242:.*]] = aie.switchbox(%[[VAL_143]]) {
// CHECK:           }
// CHECK:           %[[VAL_243:.*]] = aie.switchbox(%[[VAL_144]]) {
// CHECK:           }
// CHECK:           %[[VAL_244:.*]] = aie.switchbox(%[[VAL_145]]) {
// CHECK:           }
// CHECK:           %[[VAL_245:.*]] = aie.switchbox(%[[VAL_146]]) {
// CHECK:           }
// CHECK:           %[[VAL_246:.*]] = aie.switchbox(%[[VAL_155]]) {
// CHECK:           }
// CHECK:           %[[VAL_247:.*]] = aie.switchbox(%[[VAL_156]]) {
// CHECK:           }
// CHECK:           %[[VAL_248:.*]] = aie.switchbox(%[[VAL_157]]) {
// CHECK:           }
// CHECK:           %[[VAL_249:.*]] = aie.switchbox(%[[VAL_158]]) {
// CHECK:           }
// CHECK:           %[[VAL_250:.*]] = aie.switchbox(%[[VAL_159]]) {
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_251:.*]] = aie.switchbox(%[[VAL_167]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_252:.*]] = aie.switchbox(%[[VAL_168]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_253:.*]] = aie.switchbox(%[[VAL_169]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_254:.*]] = aie.switchbox(%[[VAL_170]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_255:.*]] = aie.switchbox(%[[VAL_171]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_256:.*]] = aie.switchbox(%[[VAL_5]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_257:.*]] = aie.shim_mux(%[[VAL_5]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_258:.*]] = aie.switchbox(%[[VAL_8]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_259:.*]] = aie.switchbox(%[[VAL_9]]) {
// CHECK:             aie.connect<West : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_260:.*]] = aie.shim_mux(%[[VAL_9]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_261:.*]] = aie.switchbox(%[[VAL_71]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_262:.*]] = aie.switchbox(%[[VAL_16]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_263:.*]] = aie.shim_mux(%[[VAL_16]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_264:.*]] = aie.switchbox(%[[VAL_4]]) {
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_265:.*]] = aie.shim_mux(%[[VAL_4]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_266:.*]] = aie.switchbox(%[[VAL_72]]) {
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_267:.*]] = aie.switchbox(%[[VAL_17]]) {
// CHECK:             aie.connect<South : 3, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_268:.*]] = aie.shim_mux(%[[VAL_17]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_269:.*]] = aie.tile(12, 0)
// CHECK:           %[[VAL_270:.*]] = aie.switchbox(%[[VAL_269]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_271:.*]] = aie.switchbox(%[[VAL_166]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_272:.*]] = aie.tile(17, 0)
// CHECK:           %[[VAL_273:.*]] = aie.switchbox(%[[VAL_272]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_274:.*]] = aie.switchbox(%[[VAL_18]]) {
// CHECK:             aie.connect<West : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_275:.*]] = aie.shim_mux(%[[VAL_18]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_276:.*]] = aie.tile(17, 1)
// CHECK:           %[[VAL_277:.*]] = aie.switchbox(%[[VAL_276]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_278:.*]] = aie.switchbox(%[[VAL_180]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_279:.*]] = aie.tile(15, 2)
// CHECK:           %[[VAL_280:.*]] = aie.switchbox(%[[VAL_279]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_281:.*]] = aie.tile(16, 2)
// CHECK:           %[[VAL_282:.*]] = aie.switchbox(%[[VAL_281]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_283:.*]] = aie.tile(17, 2)
// CHECK:           %[[VAL_284:.*]] = aie.switchbox(%[[VAL_283]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_285:.*]] = aie.switchbox(%[[VAL_181]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_286:.*]] = aie.switchbox(%[[VAL_182]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_287:.*]] = aie.switchbox(%[[VAL_183]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[VAL_21]] : Core, %[[VAL_288:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_21]] : DMA, %[[VAL_288]] : DMA)
// CHECK:           aie.wire(%[[VAL_22]] : Core, %[[VAL_289:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_22]] : DMA, %[[VAL_289]] : DMA)
// CHECK:           aie.wire(%[[VAL_288]] : North, %[[VAL_289]] : South)
// CHECK:           aie.wire(%[[VAL_23]] : Core, %[[VAL_290:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_23]] : DMA, %[[VAL_290]] : DMA)
// CHECK:           aie.wire(%[[VAL_289]] : North, %[[VAL_290]] : South)
// CHECK:           aie.wire(%[[VAL_24]] : Core, %[[VAL_291:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_24]] : DMA, %[[VAL_291]] : DMA)
// CHECK:           aie.wire(%[[VAL_290]] : North, %[[VAL_291]] : South)
// CHECK:           aie.wire(%[[VAL_288]] : East, %[[VAL_292:.*]] : West)
// CHECK:           aie.wire(%[[VAL_33]] : Core, %[[VAL_292]] : Core)
// CHECK:           aie.wire(%[[VAL_33]] : DMA, %[[VAL_292]] : DMA)
// CHECK:           aie.wire(%[[VAL_289]] : East, %[[VAL_293:.*]] : West)
// CHECK:           aie.wire(%[[VAL_34]] : Core, %[[VAL_293]] : Core)
// CHECK:           aie.wire(%[[VAL_34]] : DMA, %[[VAL_293]] : DMA)
// CHECK:           aie.wire(%[[VAL_292]] : North, %[[VAL_293]] : South)
// CHECK:           aie.wire(%[[VAL_290]] : East, %[[VAL_294:.*]] : West)
// CHECK:           aie.wire(%[[VAL_35]] : Core, %[[VAL_294]] : Core)
// CHECK:           aie.wire(%[[VAL_35]] : DMA, %[[VAL_294]] : DMA)
// CHECK:           aie.wire(%[[VAL_293]] : North, %[[VAL_294]] : South)
// CHECK:           aie.wire(%[[VAL_291]] : East, %[[VAL_295:.*]] : West)
// CHECK:           aie.wire(%[[VAL_36]] : Core, %[[VAL_295]] : Core)
// CHECK:           aie.wire(%[[VAL_36]] : DMA, %[[VAL_295]] : DMA)
// CHECK:           aie.wire(%[[VAL_294]] : North, %[[VAL_295]] : South)
// CHECK:           aie.wire(%[[VAL_296:.*]] : North, %[[VAL_297:.*]] : South)
// CHECK:           aie.wire(%[[VAL_4]] : DMA, %[[VAL_296]] : DMA)
// CHECK:           aie.wire(%[[VAL_292]] : East, %[[VAL_298:.*]] : West)
// CHECK:           aie.wire(%[[VAL_45]] : Core, %[[VAL_298]] : Core)
// CHECK:           aie.wire(%[[VAL_45]] : DMA, %[[VAL_298]] : DMA)
// CHECK:           aie.wire(%[[VAL_297]] : North, %[[VAL_298]] : South)
// CHECK:           aie.wire(%[[VAL_293]] : East, %[[VAL_299:.*]] : West)
// CHECK:           aie.wire(%[[VAL_46]] : Core, %[[VAL_299]] : Core)
// CHECK:           aie.wire(%[[VAL_46]] : DMA, %[[VAL_299]] : DMA)
// CHECK:           aie.wire(%[[VAL_298]] : North, %[[VAL_299]] : South)
// CHECK:           aie.wire(%[[VAL_294]] : East, %[[VAL_300:.*]] : West)
// CHECK:           aie.wire(%[[VAL_47]] : Core, %[[VAL_300]] : Core)
// CHECK:           aie.wire(%[[VAL_47]] : DMA, %[[VAL_300]] : DMA)
// CHECK:           aie.wire(%[[VAL_299]] : North, %[[VAL_300]] : South)
// CHECK:           aie.wire(%[[VAL_295]] : East, %[[VAL_301:.*]] : West)
// CHECK:           aie.wire(%[[VAL_48]] : Core, %[[VAL_301]] : Core)
// CHECK:           aie.wire(%[[VAL_48]] : DMA, %[[VAL_301]] : DMA)
// CHECK:           aie.wire(%[[VAL_300]] : North, %[[VAL_301]] : South)
// CHECK:           aie.wire(%[[VAL_49]] : Core, %[[VAL_302:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_49]] : DMA, %[[VAL_302]] : DMA)
// CHECK:           aie.wire(%[[VAL_301]] : North, %[[VAL_302]] : South)
// CHECK:           aie.wire(%[[VAL_297]] : East, %[[VAL_303:.*]] : West)
// CHECK:           aie.wire(%[[VAL_304:.*]] : North, %[[VAL_303]] : South)
// CHECK:           aie.wire(%[[VAL_5]] : DMA, %[[VAL_304]] : DMA)
// CHECK:           aie.wire(%[[VAL_298]] : East, %[[VAL_305:.*]] : West)
// CHECK:           aie.wire(%[[VAL_56]] : Core, %[[VAL_305]] : Core)
// CHECK:           aie.wire(%[[VAL_56]] : DMA, %[[VAL_305]] : DMA)
// CHECK:           aie.wire(%[[VAL_303]] : North, %[[VAL_305]] : South)
// CHECK:           aie.wire(%[[VAL_299]] : East, %[[VAL_306:.*]] : West)
// CHECK:           aie.wire(%[[VAL_57]] : Core, %[[VAL_306]] : Core)
// CHECK:           aie.wire(%[[VAL_57]] : DMA, %[[VAL_306]] : DMA)
// CHECK:           aie.wire(%[[VAL_305]] : North, %[[VAL_306]] : South)
// CHECK:           aie.wire(%[[VAL_300]] : East, %[[VAL_307:.*]] : West)
// CHECK:           aie.wire(%[[VAL_58]] : Core, %[[VAL_307]] : Core)
// CHECK:           aie.wire(%[[VAL_58]] : DMA, %[[VAL_307]] : DMA)
// CHECK:           aie.wire(%[[VAL_306]] : North, %[[VAL_307]] : South)
// CHECK:           aie.wire(%[[VAL_301]] : East, %[[VAL_308:.*]] : West)
// CHECK:           aie.wire(%[[VAL_59]] : Core, %[[VAL_308]] : Core)
// CHECK:           aie.wire(%[[VAL_59]] : DMA, %[[VAL_308]] : DMA)
// CHECK:           aie.wire(%[[VAL_307]] : North, %[[VAL_308]] : South)
// CHECK:           aie.wire(%[[VAL_302]] : East, %[[VAL_309:.*]] : West)
// CHECK:           aie.wire(%[[VAL_60]] : Core, %[[VAL_309]] : Core)
// CHECK:           aie.wire(%[[VAL_60]] : DMA, %[[VAL_309]] : DMA)
// CHECK:           aie.wire(%[[VAL_308]] : North, %[[VAL_309]] : South)
// CHECK:           aie.wire(%[[VAL_305]] : East, %[[VAL_310:.*]] : West)
// CHECK:           aie.wire(%[[VAL_67]] : Core, %[[VAL_310]] : Core)
// CHECK:           aie.wire(%[[VAL_67]] : DMA, %[[VAL_310]] : DMA)
// CHECK:           aie.wire(%[[VAL_306]] : East, %[[VAL_311:.*]] : West)
// CHECK:           aie.wire(%[[VAL_68]] : Core, %[[VAL_311]] : Core)
// CHECK:           aie.wire(%[[VAL_68]] : DMA, %[[VAL_311]] : DMA)
// CHECK:           aie.wire(%[[VAL_310]] : North, %[[VAL_311]] : South)
// CHECK:           aie.wire(%[[VAL_307]] : East, %[[VAL_312:.*]] : West)
// CHECK:           aie.wire(%[[VAL_69]] : Core, %[[VAL_312]] : Core)
// CHECK:           aie.wire(%[[VAL_69]] : DMA, %[[VAL_312]] : DMA)
// CHECK:           aie.wire(%[[VAL_311]] : North, %[[VAL_312]] : South)
// CHECK:           aie.wire(%[[VAL_308]] : East, %[[VAL_313:.*]] : West)
// CHECK:           aie.wire(%[[VAL_70]] : Core, %[[VAL_313]] : Core)
// CHECK:           aie.wire(%[[VAL_70]] : DMA, %[[VAL_313]] : DMA)
// CHECK:           aie.wire(%[[VAL_312]] : North, %[[VAL_313]] : South)
// CHECK:           aie.wire(%[[VAL_309]] : East, %[[VAL_314:.*]] : West)
// CHECK:           aie.wire(%[[VAL_71]] : Core, %[[VAL_314]] : Core)
// CHECK:           aie.wire(%[[VAL_71]] : DMA, %[[VAL_314]] : DMA)
// CHECK:           aie.wire(%[[VAL_313]] : North, %[[VAL_314]] : South)
// CHECK:           aie.wire(%[[VAL_72]] : Core, %[[VAL_315:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_72]] : DMA, %[[VAL_315]] : DMA)
// CHECK:           aie.wire(%[[VAL_314]] : North, %[[VAL_315]] : South)
// CHECK:           aie.wire(%[[VAL_310]] : East, %[[VAL_316:.*]] : West)
// CHECK:           aie.wire(%[[VAL_77]] : Core, %[[VAL_316]] : Core)
// CHECK:           aie.wire(%[[VAL_77]] : DMA, %[[VAL_316]] : DMA)
// CHECK:           aie.wire(%[[VAL_317:.*]] : North, %[[VAL_316]] : South)
// CHECK:           aie.wire(%[[VAL_311]] : East, %[[VAL_318:.*]] : West)
// CHECK:           aie.wire(%[[VAL_78]] : Core, %[[VAL_318]] : Core)
// CHECK:           aie.wire(%[[VAL_78]] : DMA, %[[VAL_318]] : DMA)
// CHECK:           aie.wire(%[[VAL_316]] : North, %[[VAL_318]] : South)
// CHECK:           aie.wire(%[[VAL_312]] : East, %[[VAL_319:.*]] : West)
// CHECK:           aie.wire(%[[VAL_79]] : Core, %[[VAL_319]] : Core)
// CHECK:           aie.wire(%[[VAL_79]] : DMA, %[[VAL_319]] : DMA)
// CHECK:           aie.wire(%[[VAL_318]] : North, %[[VAL_319]] : South)
// CHECK:           aie.wire(%[[VAL_313]] : East, %[[VAL_320:.*]] : West)
// CHECK:           aie.wire(%[[VAL_80]] : Core, %[[VAL_320]] : Core)
// CHECK:           aie.wire(%[[VAL_80]] : DMA, %[[VAL_320]] : DMA)
// CHECK:           aie.wire(%[[VAL_319]] : North, %[[VAL_320]] : South)
// CHECK:           aie.wire(%[[VAL_314]] : East, %[[VAL_321:.*]] : West)
// CHECK:           aie.wire(%[[VAL_81]] : Core, %[[VAL_321]] : Core)
// CHECK:           aie.wire(%[[VAL_81]] : DMA, %[[VAL_321]] : DMA)
// CHECK:           aie.wire(%[[VAL_320]] : North, %[[VAL_321]] : South)
// CHECK:           aie.wire(%[[VAL_315]] : East, %[[VAL_322:.*]] : West)
// CHECK:           aie.wire(%[[VAL_82]] : Core, %[[VAL_322]] : Core)
// CHECK:           aie.wire(%[[VAL_82]] : DMA, %[[VAL_322]] : DMA)
// CHECK:           aie.wire(%[[VAL_321]] : North, %[[VAL_322]] : South)
// CHECK:           aie.wire(%[[VAL_317]] : East, %[[VAL_323:.*]] : West)
// CHECK:           aie.wire(%[[VAL_324:.*]] : North, %[[VAL_323]] : South)
// CHECK:           aie.wire(%[[VAL_9]] : DMA, %[[VAL_324]] : DMA)
// CHECK:           aie.wire(%[[VAL_316]] : East, %[[VAL_325:.*]] : West)
// CHECK:           aie.wire(%[[VAL_87]] : Core, %[[VAL_325]] : Core)
// CHECK:           aie.wire(%[[VAL_87]] : DMA, %[[VAL_325]] : DMA)
// CHECK:           aie.wire(%[[VAL_323]] : North, %[[VAL_325]] : South)
// CHECK:           aie.wire(%[[VAL_318]] : East, %[[VAL_326:.*]] : West)
// CHECK:           aie.wire(%[[VAL_88]] : Core, %[[VAL_326]] : Core)
// CHECK:           aie.wire(%[[VAL_88]] : DMA, %[[VAL_326]] : DMA)
// CHECK:           aie.wire(%[[VAL_325]] : North, %[[VAL_326]] : South)
// CHECK:           aie.wire(%[[VAL_319]] : East, %[[VAL_327:.*]] : West)
// CHECK:           aie.wire(%[[VAL_89]] : Core, %[[VAL_327]] : Core)
// CHECK:           aie.wire(%[[VAL_89]] : DMA, %[[VAL_327]] : DMA)
// CHECK:           aie.wire(%[[VAL_326]] : North, %[[VAL_327]] : South)
// CHECK:           aie.wire(%[[VAL_320]] : East, %[[VAL_328:.*]] : West)
// CHECK:           aie.wire(%[[VAL_90]] : Core, %[[VAL_328]] : Core)
// CHECK:           aie.wire(%[[VAL_90]] : DMA, %[[VAL_328]] : DMA)
// CHECK:           aie.wire(%[[VAL_327]] : North, %[[VAL_328]] : South)
// CHECK:           aie.wire(%[[VAL_321]] : East, %[[VAL_329:.*]] : West)
// CHECK:           aie.wire(%[[VAL_91]] : Core, %[[VAL_329]] : Core)
// CHECK:           aie.wire(%[[VAL_91]] : DMA, %[[VAL_329]] : DMA)
// CHECK:           aie.wire(%[[VAL_328]] : North, %[[VAL_329]] : South)
// CHECK:           aie.wire(%[[VAL_322]] : East, %[[VAL_330:.*]] : West)
// CHECK:           aie.wire(%[[VAL_92]] : Core, %[[VAL_330]] : Core)
// CHECK:           aie.wire(%[[VAL_92]] : DMA, %[[VAL_330]] : DMA)
// CHECK:           aie.wire(%[[VAL_329]] : North, %[[VAL_330]] : South)
// CHECK:           aie.wire(%[[VAL_325]] : East, %[[VAL_331:.*]] : West)
// CHECK:           aie.wire(%[[VAL_97]] : Core, %[[VAL_331]] : Core)
// CHECK:           aie.wire(%[[VAL_97]] : DMA, %[[VAL_331]] : DMA)
// CHECK:           aie.wire(%[[VAL_326]] : East, %[[VAL_332:.*]] : West)
// CHECK:           aie.wire(%[[VAL_98]] : Core, %[[VAL_332]] : Core)
// CHECK:           aie.wire(%[[VAL_98]] : DMA, %[[VAL_332]] : DMA)
// CHECK:           aie.wire(%[[VAL_331]] : North, %[[VAL_332]] : South)
// CHECK:           aie.wire(%[[VAL_327]] : East, %[[VAL_333:.*]] : West)
// CHECK:           aie.wire(%[[VAL_99]] : Core, %[[VAL_333]] : Core)
// CHECK:           aie.wire(%[[VAL_99]] : DMA, %[[VAL_333]] : DMA)
// CHECK:           aie.wire(%[[VAL_332]] : North, %[[VAL_333]] : South)
// CHECK:           aie.wire(%[[VAL_328]] : East, %[[VAL_334:.*]] : West)
// CHECK:           aie.wire(%[[VAL_100]] : Core, %[[VAL_334]] : Core)
// CHECK:           aie.wire(%[[VAL_100]] : DMA, %[[VAL_334]] : DMA)
// CHECK:           aie.wire(%[[VAL_333]] : North, %[[VAL_334]] : South)
// CHECK:           aie.wire(%[[VAL_329]] : East, %[[VAL_335:.*]] : West)
// CHECK:           aie.wire(%[[VAL_101]] : Core, %[[VAL_335]] : Core)
// CHECK:           aie.wire(%[[VAL_101]] : DMA, %[[VAL_335]] : DMA)
// CHECK:           aie.wire(%[[VAL_334]] : North, %[[VAL_335]] : South)
// CHECK:           aie.wire(%[[VAL_330]] : East, %[[VAL_336:.*]] : West)
// CHECK:           aie.wire(%[[VAL_102]] : Core, %[[VAL_336]] : Core)
// CHECK:           aie.wire(%[[VAL_102]] : DMA, %[[VAL_336]] : DMA)
// CHECK:           aie.wire(%[[VAL_335]] : North, %[[VAL_336]] : South)
// CHECK:           aie.wire(%[[VAL_331]] : East, %[[VAL_337:.*]] : West)
// CHECK:           aie.wire(%[[VAL_107]] : Core, %[[VAL_337]] : Core)
// CHECK:           aie.wire(%[[VAL_107]] : DMA, %[[VAL_337]] : DMA)
// CHECK:           aie.wire(%[[VAL_332]] : East, %[[VAL_338:.*]] : West)
// CHECK:           aie.wire(%[[VAL_108]] : Core, %[[VAL_338]] : Core)
// CHECK:           aie.wire(%[[VAL_108]] : DMA, %[[VAL_338]] : DMA)
// CHECK:           aie.wire(%[[VAL_337]] : North, %[[VAL_338]] : South)
// CHECK:           aie.wire(%[[VAL_333]] : East, %[[VAL_339:.*]] : West)
// CHECK:           aie.wire(%[[VAL_109]] : Core, %[[VAL_339]] : Core)
// CHECK:           aie.wire(%[[VAL_109]] : DMA, %[[VAL_339]] : DMA)
// CHECK:           aie.wire(%[[VAL_338]] : North, %[[VAL_339]] : South)
// CHECK:           aie.wire(%[[VAL_334]] : East, %[[VAL_340:.*]] : West)
// CHECK:           aie.wire(%[[VAL_110]] : Core, %[[VAL_340]] : Core)
// CHECK:           aie.wire(%[[VAL_110]] : DMA, %[[VAL_340]] : DMA)
// CHECK:           aie.wire(%[[VAL_339]] : North, %[[VAL_340]] : South)
// CHECK:           aie.wire(%[[VAL_337]] : East, %[[VAL_341:.*]] : West)
// CHECK:           aie.wire(%[[VAL_119]] : Core, %[[VAL_341]] : Core)
// CHECK:           aie.wire(%[[VAL_119]] : DMA, %[[VAL_341]] : DMA)
// CHECK:           aie.wire(%[[VAL_338]] : East, %[[VAL_342:.*]] : West)
// CHECK:           aie.wire(%[[VAL_120]] : Core, %[[VAL_342]] : Core)
// CHECK:           aie.wire(%[[VAL_120]] : DMA, %[[VAL_342]] : DMA)
// CHECK:           aie.wire(%[[VAL_341]] : North, %[[VAL_342]] : South)
// CHECK:           aie.wire(%[[VAL_339]] : East, %[[VAL_343:.*]] : West)
// CHECK:           aie.wire(%[[VAL_121]] : Core, %[[VAL_343]] : Core)
// CHECK:           aie.wire(%[[VAL_121]] : DMA, %[[VAL_343]] : DMA)
// CHECK:           aie.wire(%[[VAL_342]] : North, %[[VAL_343]] : South)
// CHECK:           aie.wire(%[[VAL_340]] : East, %[[VAL_344:.*]] : West)
// CHECK:           aie.wire(%[[VAL_122]] : Core, %[[VAL_344]] : Core)
// CHECK:           aie.wire(%[[VAL_122]] : DMA, %[[VAL_344]] : DMA)
// CHECK:           aie.wire(%[[VAL_343]] : North, %[[VAL_344]] : South)
// CHECK:           aie.wire(%[[VAL_345:.*]] : North, %[[VAL_346:.*]] : South)
// CHECK:           aie.wire(%[[VAL_16]] : DMA, %[[VAL_345]] : DMA)
// CHECK:           aie.wire(%[[VAL_341]] : East, %[[VAL_347:.*]] : West)
// CHECK:           aie.wire(%[[VAL_131]] : Core, %[[VAL_347]] : Core)
// CHECK:           aie.wire(%[[VAL_131]] : DMA, %[[VAL_347]] : DMA)
// CHECK:           aie.wire(%[[VAL_346]] : North, %[[VAL_347]] : South)
// CHECK:           aie.wire(%[[VAL_342]] : East, %[[VAL_348:.*]] : West)
// CHECK:           aie.wire(%[[VAL_132]] : Core, %[[VAL_348]] : Core)
// CHECK:           aie.wire(%[[VAL_132]] : DMA, %[[VAL_348]] : DMA)
// CHECK:           aie.wire(%[[VAL_347]] : North, %[[VAL_348]] : South)
// CHECK:           aie.wire(%[[VAL_343]] : East, %[[VAL_349:.*]] : West)
// CHECK:           aie.wire(%[[VAL_133]] : Core, %[[VAL_349]] : Core)
// CHECK:           aie.wire(%[[VAL_133]] : DMA, %[[VAL_349]] : DMA)
// CHECK:           aie.wire(%[[VAL_348]] : North, %[[VAL_349]] : South)
// CHECK:           aie.wire(%[[VAL_344]] : East, %[[VAL_350:.*]] : West)
// CHECK:           aie.wire(%[[VAL_134]] : Core, %[[VAL_350]] : Core)
// CHECK:           aie.wire(%[[VAL_134]] : DMA, %[[VAL_350]] : DMA)
// CHECK:           aie.wire(%[[VAL_349]] : North, %[[VAL_350]] : South)
// CHECK:           aie.wire(%[[VAL_346]] : East, %[[VAL_351:.*]] : West)
// CHECK:           aie.wire(%[[VAL_352:.*]] : North, %[[VAL_351]] : South)
// CHECK:           aie.wire(%[[VAL_17]] : DMA, %[[VAL_352]] : DMA)
// CHECK:           aie.wire(%[[VAL_347]] : East, %[[VAL_353:.*]] : West)
// CHECK:           aie.wire(%[[VAL_143]] : Core, %[[VAL_353]] : Core)
// CHECK:           aie.wire(%[[VAL_143]] : DMA, %[[VAL_353]] : DMA)
// CHECK:           aie.wire(%[[VAL_351]] : North, %[[VAL_353]] : South)
// CHECK:           aie.wire(%[[VAL_348]] : East, %[[VAL_354:.*]] : West)
// CHECK:           aie.wire(%[[VAL_144]] : Core, %[[VAL_354]] : Core)
// CHECK:           aie.wire(%[[VAL_144]] : DMA, %[[VAL_354]] : DMA)
// CHECK:           aie.wire(%[[VAL_353]] : North, %[[VAL_354]] : South)
// CHECK:           aie.wire(%[[VAL_349]] : East, %[[VAL_355:.*]] : West)
// CHECK:           aie.wire(%[[VAL_145]] : Core, %[[VAL_355]] : Core)
// CHECK:           aie.wire(%[[VAL_145]] : DMA, %[[VAL_355]] : DMA)
// CHECK:           aie.wire(%[[VAL_354]] : North, %[[VAL_355]] : South)
// CHECK:           aie.wire(%[[VAL_350]] : East, %[[VAL_356:.*]] : West)
// CHECK:           aie.wire(%[[VAL_146]] : Core, %[[VAL_356]] : Core)
// CHECK:           aie.wire(%[[VAL_146]] : DMA, %[[VAL_356]] : DMA)
// CHECK:           aie.wire(%[[VAL_355]] : North, %[[VAL_356]] : South)
// CHECK:           aie.wire(%[[VAL_351]] : East, %[[VAL_357:.*]] : West)
// CHECK:           aie.wire(%[[VAL_353]] : East, %[[VAL_358:.*]] : West)
// CHECK:           aie.wire(%[[VAL_155]] : Core, %[[VAL_358]] : Core)
// CHECK:           aie.wire(%[[VAL_155]] : DMA, %[[VAL_358]] : DMA)
// CHECK:           aie.wire(%[[VAL_357]] : North, %[[VAL_358]] : South)
// CHECK:           aie.wire(%[[VAL_354]] : East, %[[VAL_359:.*]] : West)
// CHECK:           aie.wire(%[[VAL_156]] : Core, %[[VAL_359]] : Core)
// CHECK:           aie.wire(%[[VAL_156]] : DMA, %[[VAL_359]] : DMA)
// CHECK:           aie.wire(%[[VAL_358]] : North, %[[VAL_359]] : South)
// CHECK:           aie.wire(%[[VAL_355]] : East, %[[VAL_360:.*]] : West)
// CHECK:           aie.wire(%[[VAL_157]] : Core, %[[VAL_360]] : Core)
// CHECK:           aie.wire(%[[VAL_157]] : DMA, %[[VAL_360]] : DMA)
// CHECK:           aie.wire(%[[VAL_359]] : North, %[[VAL_360]] : South)
// CHECK:           aie.wire(%[[VAL_356]] : East, %[[VAL_361:.*]] : West)
// CHECK:           aie.wire(%[[VAL_158]] : Core, %[[VAL_361]] : Core)
// CHECK:           aie.wire(%[[VAL_158]] : DMA, %[[VAL_361]] : DMA)
// CHECK:           aie.wire(%[[VAL_360]] : North, %[[VAL_361]] : South)
// CHECK:           aie.wire(%[[VAL_159]] : Core, %[[VAL_362:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_159]] : DMA, %[[VAL_362]] : DMA)
// CHECK:           aie.wire(%[[VAL_361]] : North, %[[VAL_362]] : South)
// CHECK:           aie.wire(%[[VAL_357]] : East, %[[VAL_363:.*]] : West)
// CHECK:           aie.wire(%[[VAL_358]] : East, %[[VAL_364:.*]] : West)
// CHECK:           aie.wire(%[[VAL_167]] : Core, %[[VAL_364]] : Core)
// CHECK:           aie.wire(%[[VAL_167]] : DMA, %[[VAL_364]] : DMA)
// CHECK:           aie.wire(%[[VAL_363]] : North, %[[VAL_364]] : South)
// CHECK:           aie.wire(%[[VAL_359]] : East, %[[VAL_365:.*]] : West)
// CHECK:           aie.wire(%[[VAL_168]] : Core, %[[VAL_365]] : Core)
// CHECK:           aie.wire(%[[VAL_168]] : DMA, %[[VAL_365]] : DMA)
// CHECK:           aie.wire(%[[VAL_364]] : North, %[[VAL_365]] : South)
// CHECK:           aie.wire(%[[VAL_360]] : East, %[[VAL_366:.*]] : West)
// CHECK:           aie.wire(%[[VAL_169]] : Core, %[[VAL_366]] : Core)
// CHECK:           aie.wire(%[[VAL_169]] : DMA, %[[VAL_366]] : DMA)
// CHECK:           aie.wire(%[[VAL_365]] : North, %[[VAL_366]] : South)
// CHECK:           aie.wire(%[[VAL_361]] : East, %[[VAL_367:.*]] : West)
// CHECK:           aie.wire(%[[VAL_170]] : Core, %[[VAL_367]] : Core)
// CHECK:           aie.wire(%[[VAL_170]] : DMA, %[[VAL_367]] : DMA)
// CHECK:           aie.wire(%[[VAL_366]] : North, %[[VAL_367]] : South)
// CHECK:           aie.wire(%[[VAL_362]] : East, %[[VAL_368:.*]] : West)
// CHECK:           aie.wire(%[[VAL_171]] : Core, %[[VAL_368]] : Core)
// CHECK:           aie.wire(%[[VAL_171]] : DMA, %[[VAL_368]] : DMA)
// CHECK:           aie.wire(%[[VAL_367]] : North, %[[VAL_368]] : South)
// CHECK:           aie.wire(%[[VAL_365]] : East, %[[VAL_369:.*]] : West)
// CHECK:           aie.wire(%[[VAL_180]] : Core, %[[VAL_369]] : Core)
// CHECK:           aie.wire(%[[VAL_180]] : DMA, %[[VAL_369]] : DMA)
// CHECK:           aie.wire(%[[VAL_366]] : East, %[[VAL_370:.*]] : West)
// CHECK:           aie.wire(%[[VAL_181]] : Core, %[[VAL_370]] : Core)
// CHECK:           aie.wire(%[[VAL_181]] : DMA, %[[VAL_370]] : DMA)
// CHECK:           aie.wire(%[[VAL_369]] : North, %[[VAL_370]] : South)
// CHECK:           aie.wire(%[[VAL_367]] : East, %[[VAL_371:.*]] : West)
// CHECK:           aie.wire(%[[VAL_182]] : Core, %[[VAL_371]] : Core)
// CHECK:           aie.wire(%[[VAL_182]] : DMA, %[[VAL_371]] : DMA)
// CHECK:           aie.wire(%[[VAL_370]] : North, %[[VAL_371]] : South)
// CHECK:           aie.wire(%[[VAL_368]] : East, %[[VAL_372:.*]] : West)
// CHECK:           aie.wire(%[[VAL_183]] : Core, %[[VAL_372]] : Core)
// CHECK:           aie.wire(%[[VAL_183]] : DMA, %[[VAL_372]] : DMA)
// CHECK:           aie.wire(%[[VAL_371]] : North, %[[VAL_372]] : South)
// CHECK:           aie.wire(%[[VAL_369]] : East, %[[VAL_373:.*]] : West)
// CHECK:           aie.wire(%[[VAL_279]] : Core, %[[VAL_373]] : Core)
// CHECK:           aie.wire(%[[VAL_279]] : DMA, %[[VAL_373]] : DMA)
// CHECK:           aie.wire(%[[VAL_373]] : East, %[[VAL_374:.*]] : West)
// CHECK:           aie.wire(%[[VAL_281]] : Core, %[[VAL_374]] : Core)
// CHECK:           aie.wire(%[[VAL_281]] : DMA, %[[VAL_374]] : DMA)
// CHECK:           aie.wire(%[[VAL_276]] : Core, %[[VAL_375:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_276]] : DMA, %[[VAL_375]] : DMA)
// CHECK:           aie.wire(%[[VAL_376:.*]] : North, %[[VAL_375]] : South)
// CHECK:           aie.wire(%[[VAL_374]] : East, %[[VAL_377:.*]] : West)
// CHECK:           aie.wire(%[[VAL_283]] : Core, %[[VAL_377]] : Core)
// CHECK:           aie.wire(%[[VAL_283]] : DMA, %[[VAL_377]] : DMA)
// CHECK:           aie.wire(%[[VAL_375]] : North, %[[VAL_377]] : South)
// CHECK:           aie.wire(%[[VAL_376]] : East, %[[VAL_378:.*]] : West)
// CHECK:           aie.wire(%[[VAL_379:.*]] : North, %[[VAL_378]] : South)
// CHECK:           aie.wire(%[[VAL_18]] : DMA, %[[VAL_379]] : DMA)
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_2_0 = aie.tile(2, 0)
    %tile_3_0 = aie.tile(3, 0)
    %tile_4_0 = aie.tile(4, 0)
    %tile_5_0 = aie.tile(5, 0)
    %tile_6_0 = aie.tile(6, 0)
    %tile_7_0 = aie.tile(7, 0)
    %tile_8_0 = aie.tile(8, 0)
    %tile_9_0 = aie.tile(9, 0)
    %tile_10_0 = aie.tile(10, 0)
    %tile_11_0 = aie.tile(11, 0)
    %tile_18_0 = aie.tile(18, 0)
    %tile_19_0 = aie.tile(19, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_0_6 = aie.tile(0, 6)
    %tile_0_7 = aie.tile(0, 7)
    %tile_0_8 = aie.tile(0, 8)
    %tile_1_1 = aie.tile(1, 1)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    %tile_1_6 = aie.tile(1, 6)
    %tile_1_7 = aie.tile(1, 7)
    %tile_1_8 = aie.tile(1, 8)
    %tile_2_1 = aie.tile(2, 1)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %tile_2_5 = aie.tile(2, 5)
    %tile_2_6 = aie.tile(2, 6)
    %tile_2_7 = aie.tile(2, 7)
    %tile_2_8 = aie.tile(2, 8)
    %tile_3_1 = aie.tile(3, 1)
    %tile_3_2 = aie.tile(3, 2)
    %tile_3_3 = aie.tile(3, 3)
    %tile_3_4 = aie.tile(3, 4)
    %tile_3_5 = aie.tile(3, 5)
    %tile_3_6 = aie.tile(3, 6)
    %tile_3_7 = aie.tile(3, 7)
    %tile_3_8 = aie.tile(3, 8)
    %tile_4_1 = aie.tile(4, 1)
    %tile_4_2 = aie.tile(4, 2)
    %tile_4_3 = aie.tile(4, 3)
    %tile_4_4 = aie.tile(4, 4)
    %tile_4_5 = aie.tile(4, 5)
    %tile_4_6 = aie.tile(4, 6)
    %tile_4_7 = aie.tile(4, 7)
    %tile_4_8 = aie.tile(4, 8)
    %tile_5_1 = aie.tile(5, 1)
    %tile_5_2 = aie.tile(5, 2)
    %tile_5_3 = aie.tile(5, 3)
    %tile_5_4 = aie.tile(5, 4)
    %tile_5_5 = aie.tile(5, 5)
    %tile_5_6 = aie.tile(5, 6)
    %tile_5_7 = aie.tile(5, 7)
    %tile_5_8 = aie.tile(5, 8)
    %tile_6_1 = aie.tile(6, 1)
    %tile_6_2 = aie.tile(6, 2)
    %tile_6_3 = aie.tile(6, 3)
    %tile_6_4 = aie.tile(6, 4)
    %tile_6_5 = aie.tile(6, 5)
    %tile_6_6 = aie.tile(6, 6)
    %tile_6_7 = aie.tile(6, 7)
    %tile_6_8 = aie.tile(6, 8)
    %tile_7_1 = aie.tile(7, 1)
    %tile_7_2 = aie.tile(7, 2)
    %tile_7_3 = aie.tile(7, 3)
    %tile_7_4 = aie.tile(7, 4)
    %tile_7_5 = aie.tile(7, 5)
    %tile_7_6 = aie.tile(7, 6)
    %tile_7_7 = aie.tile(7, 7)
    %tile_7_8 = aie.tile(7, 8)
    %tile_8_1 = aie.tile(8, 1)
    %tile_8_2 = aie.tile(8, 2)
    %tile_8_3 = aie.tile(8, 3)
    %tile_8_4 = aie.tile(8, 4)
    %tile_8_5 = aie.tile(8, 5)
    %tile_8_6 = aie.tile(8, 6)
    %tile_8_7 = aie.tile(8, 7)
    %tile_8_8 = aie.tile(8, 8)
    %tile_9_1 = aie.tile(9, 1)
    %tile_9_2 = aie.tile(9, 2)
    %tile_9_3 = aie.tile(9, 3)
    %tile_9_4 = aie.tile(9, 4)
    %tile_9_5 = aie.tile(9, 5)
    %tile_9_6 = aie.tile(9, 6)
    %tile_9_7 = aie.tile(9, 7)
    %tile_9_8 = aie.tile(9, 8)
    %tile_10_1 = aie.tile(10, 1)
    %tile_10_2 = aie.tile(10, 2)
    %tile_10_3 = aie.tile(10, 3)
    %tile_10_4 = aie.tile(10, 4)
    %tile_10_5 = aie.tile(10, 5)
    %tile_10_6 = aie.tile(10, 6)
    %tile_10_7 = aie.tile(10, 7)
    %tile_10_8 = aie.tile(10, 8)
    %tile_11_1 = aie.tile(11, 1)
    %tile_11_2 = aie.tile(11, 2)
    %tile_11_3 = aie.tile(11, 3)
    %tile_11_4 = aie.tile(11, 4)
    %tile_11_5 = aie.tile(11, 5)
    %tile_11_6 = aie.tile(11, 6)
    %tile_11_7 = aie.tile(11, 7)
    %tile_11_8 = aie.tile(11, 8)
    %tile_12_1 = aie.tile(12, 1)
    %tile_12_2 = aie.tile(12, 2)
    %tile_12_3 = aie.tile(12, 3)
    %tile_12_4 = aie.tile(12, 4)
    %tile_12_5 = aie.tile(12, 5)
    %tile_12_6 = aie.tile(12, 6)
    %tile_12_7 = aie.tile(12, 7)
    %tile_12_8 = aie.tile(12, 8)
    %tile_13_0 = aie.tile(13, 0)
    %tile_13_1 = aie.tile(13, 1)
    %tile_13_2 = aie.tile(13, 2)
    %tile_13_3 = aie.tile(13, 3)
    %tile_13_4 = aie.tile(13, 4)
    %tile_13_5 = aie.tile(13, 5)
    %tile_13_6 = aie.tile(13, 6)
    %tile_13_7 = aie.tile(13, 7)
    %tile_13_8 = aie.tile(13, 8)
    %tile_14_1 = aie.tile(14, 1)
    %tile_14_2 = aie.tile(14, 2)
    %tile_14_3 = aie.tile(14, 3)
    %tile_14_4 = aie.tile(14, 4)
    %tile_14_5 = aie.tile(14, 5)
    %tile_14_6 = aie.tile(14, 6)
    %tile_14_7 = aie.tile(14, 7)
    %tile_14_8 = aie.tile(14, 8)
    %switchbox_0_1 = aie.switchbox(%tile_0_1) {
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
    }
    %switchbox_1_1 = aie.switchbox(%tile_1_1) {
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
    }
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
    }
    %switchbox_2_1 = aie.switchbox(%tile_2_1) {
    }
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
    }
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
    }
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      aie.connect<East : 0, North : 0>
    }
    %switchbox_2_5 = aie.switchbox(%tile_2_5) {
      aie.connect<South : 0, Core : 0>
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_3_1 = aie.switchbox(%tile_3_1) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<Core : 0, North : 0>
    }
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_3_4 = aie.switchbox(%tile_3_4) {
      aie.connect<South : 0, West : 0>
    }
    %switchbox_3_5 = aie.switchbox(%tile_3_5) {
      aie.connect<West : 0, East : 0>
    }
    %switchbox_4_1 = aie.switchbox(%tile_4_1) {
    }
    %switchbox_4_2 = aie.switchbox(%tile_4_2) {
    }
    %switchbox_4_3 = aie.switchbox(%tile_4_3) {
    }
    %switchbox_4_4 = aie.switchbox(%tile_4_4) {
    }
    %switchbox_5_1 = aie.switchbox(%tile_5_1) {
    }
    %switchbox_5_2 = aie.switchbox(%tile_5_2) {
    }
    %switchbox_5_3 = aie.switchbox(%tile_5_3) {
    }
    %switchbox_5_4 = aie.switchbox(%tile_5_4) {
    }
    %switchbox_5_5 = aie.switchbox(%tile_5_5) {
    }
    %switchbox_5_6 = aie.switchbox(%tile_5_6) {
      aie.connect<East : 0, West : 0>
    }
    %switchbox_6_1 = aie.switchbox(%tile_6_1) {
    }
    %switchbox_6_2 = aie.switchbox(%tile_6_2) {
    }
    %switchbox_6_3 = aie.switchbox(%tile_6_3) {
    }
    %switchbox_6_4 = aie.switchbox(%tile_6_4) {
    }
    %switchbox_6_5 = aie.switchbox(%tile_6_5) {
    }
    %switchbox_6_6 = aie.switchbox(%tile_6_6) {
      aie.connect<East : 0, Core : 0>
      aie.connect<DMA : 0, West : 0>
    }
    %switchbox_7_1 = aie.switchbox(%tile_7_1) {
    }
    %switchbox_7_2 = aie.switchbox(%tile_7_2) {
    }
    %switchbox_7_3 = aie.switchbox(%tile_7_3) {
      aie.connect<East : 0, DMA : 0>
      aie.connect<Core : 0, North : 0>
    }
    %switchbox_7_4 = aie.switchbox(%tile_7_4) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_7_5 = aie.switchbox(%tile_7_5) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_7_6 = aie.switchbox(%tile_7_6) {
      aie.connect<South : 0, West : 0>
    }
    %switchbox_8_1 = aie.switchbox(%tile_8_1) {
    }
    %switchbox_8_2 = aie.switchbox(%tile_8_2) {
    }
    %switchbox_8_3 = aie.switchbox(%tile_8_3) {
      aie.connect<East : 0, West : 0>
    }
    %switchbox_8_4 = aie.switchbox(%tile_8_4) {
    }
    %switchbox_9_1 = aie.switchbox(%tile_9_1) {
    }
    %switchbox_9_2 = aie.switchbox(%tile_9_2) {
    }
    %switchbox_9_3 = aie.switchbox(%tile_9_3) {
    }
    %switchbox_9_4 = aie.switchbox(%tile_9_4) {
    }
    %switchbox_10_1 = aie.switchbox(%tile_10_1) {
    }
    %switchbox_10_2 = aie.switchbox(%tile_10_2) {
    }
    %switchbox_10_3 = aie.switchbox(%tile_10_3) {
    }
    %switchbox_10_4 = aie.switchbox(%tile_10_4) {
    }
    %switchbox_11_1 = aie.switchbox(%tile_11_1) {
    }
    %switchbox_11_2 = aie.switchbox(%tile_11_2) {
    }
    %switchbox_11_3 = aie.switchbox(%tile_11_3) {
    }
    %switchbox_11_4 = aie.switchbox(%tile_11_4) {
    }
    %switchbox_12_1 = aie.switchbox(%tile_12_1) {
    }
    %switchbox_12_2 = aie.switchbox(%tile_12_2) {
    }
    %switchbox_12_3 = aie.switchbox(%tile_12_3) {
    }
    %switchbox_12_4 = aie.switchbox(%tile_12_4) {
    }
    %switchbox_12_5 = aie.switchbox(%tile_12_5) {
      aie.connect<East : 0, Core : 0>
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_13_1 = aie.switchbox(%tile_13_1) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_13_2 = aie.switchbox(%tile_13_2) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_13_3 = aie.switchbox(%tile_13_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<Core : 0, North : 0>
    }
    %switchbox_13_4 = aie.switchbox(%tile_13_4) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_13_5 = aie.switchbox(%tile_13_5) {
      aie.connect<South : 0, West : 0>
      aie.connect<West : 0, East : 0>
    }
    aie.flow(%tile_3_0, DMA : 0, %tile_3_0, North : 0)
    aie.flow(%tile_4_5, West : 0, %tile_6_0, DMA : 0)
    aie.flow(%tile_10_0, DMA : 0, %tile_9_3, West : 0)
    aie.flow(%tile_4_6, East : 0, %tile_2_0, DMA : 0)
    aie.flow(%tile_11_0, DMA : 0, %tile_13_0, North : 0)
    aie.flow(%tile_14_5, West : 0, %tile_18_0, DMA : 0)
  }
}
