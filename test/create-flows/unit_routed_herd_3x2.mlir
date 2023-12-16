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
// CHECK:           %[[VAL_202:.*]] = AIE.shim_mux(%[[VAL_3]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_203:.*]] = AIE.switchbox(%[[VAL_50]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_204:.*]] = AIE.switchbox(%[[VAL_5]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_205:.*]] = AIE.switchbox(%[[VAL_6]]) {
// CHECK:             AIE.connect<West : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_206:.*]] = AIE.shim_mux(%[[VAL_6]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_207:.*]] = AIE.switchbox(%[[VAL_10]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_208:.*]] = AIE.shim_mux(%[[VAL_10]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_209:.*]] = AIE.switchbox(%[[VAL_2]]) {
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_210:.*]] = AIE.shim_mux(%[[VAL_2]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_211:.*]] = AIE.switchbox(%[[VAL_51]]) {
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_212:.*]] = AIE.switchbox(%[[VAL_11]]) {
// CHECK:             AIE.connect<South : 3, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_213:.*]] = AIE.shim_mux(%[[VAL_11]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_214:.*]] = AIE.tile(12, 0)
// CHECK:           %[[VAL_215:.*]] = AIE.switchbox(%[[VAL_214]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_216:.*]] = AIE.switchbox(%[[VAL_118]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_217:.*]] = AIE.switchbox(%[[VAL_128]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_218:.*]] = AIE.switchbox(%[[VAL_129]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_219:.*]] = AIE.switchbox(%[[VAL_130]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_220:.*]] = AIE.switchbox(%[[VAL_131]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_221:.*]] = AIE.tile(15, 2)
// CHECK:           %[[VAL_222:.*]] = AIE.switchbox(%[[VAL_221]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_223:.*]] = AIE.tile(16, 2)
// CHECK:           %[[VAL_224:.*]] = AIE.switchbox(%[[VAL_223]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_225:.*]] = AIE.tile(17, 0)
// CHECK:           %[[VAL_226:.*]] = AIE.switchbox(%[[VAL_225]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_227:.*]] = AIE.tile(17, 1)
// CHECK:           %[[VAL_228:.*]] = AIE.switchbox(%[[VAL_227]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_229:.*]] = AIE.tile(17, 2)
// CHECK:           %[[VAL_230:.*]] = AIE.switchbox(%[[VAL_229]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_231:.*]] = AIE.switchbox(%[[VAL_12]]) {
// CHECK:             AIE.connect<West : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_232:.*]] = AIE.shim_mux(%[[VAL_12]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_14]] : Core, %[[VAL_233:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_14]] : DMA, %[[VAL_233]] : DMA)
// CHECK:           AIE.wire(%[[VAL_15]] : Core, %[[VAL_234:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_15]] : DMA, %[[VAL_234]] : DMA)
// CHECK:           AIE.wire(%[[VAL_233]] : North, %[[VAL_234]] : South)
// CHECK:           AIE.wire(%[[VAL_16]] : Core, %[[VAL_235:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_16]] : DMA, %[[VAL_235]] : DMA)
// CHECK:           AIE.wire(%[[VAL_234]] : North, %[[VAL_235]] : South)
// CHECK:           AIE.wire(%[[VAL_17]] : Core, %[[VAL_236:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_17]] : DMA, %[[VAL_236]] : DMA)
// CHECK:           AIE.wire(%[[VAL_235]] : North, %[[VAL_236]] : South)
// CHECK:           AIE.wire(%[[VAL_233]] : East, %[[VAL_237:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_22]] : Core, %[[VAL_237]] : Core)
// CHECK:           AIE.wire(%[[VAL_22]] : DMA, %[[VAL_237]] : DMA)
// CHECK:           AIE.wire(%[[VAL_234]] : East, %[[VAL_238:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_23]] : Core, %[[VAL_238]] : Core)
// CHECK:           AIE.wire(%[[VAL_23]] : DMA, %[[VAL_238]] : DMA)
// CHECK:           AIE.wire(%[[VAL_237]] : North, %[[VAL_238]] : South)
// CHECK:           AIE.wire(%[[VAL_235]] : East, %[[VAL_239:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_24]] : Core, %[[VAL_239]] : Core)
// CHECK:           AIE.wire(%[[VAL_24]] : DMA, %[[VAL_239]] : DMA)
// CHECK:           AIE.wire(%[[VAL_238]] : North, %[[VAL_239]] : South)
// CHECK:           AIE.wire(%[[VAL_236]] : East, %[[VAL_240:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_25]] : Core, %[[VAL_240]] : Core)
// CHECK:           AIE.wire(%[[VAL_25]] : DMA, %[[VAL_240]] : DMA)
// CHECK:           AIE.wire(%[[VAL_239]] : North, %[[VAL_240]] : South)
// CHECK:           AIE.wire(%[[VAL_241:.*]] : North, %[[VAL_242:.*]] : South)
// CHECK:           AIE.wire(%[[VAL_2]] : DMA, %[[VAL_241]] : DMA)
// CHECK:           AIE.wire(%[[VAL_237]] : East, %[[VAL_243:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_30]] : Core, %[[VAL_243]] : Core)
// CHECK:           AIE.wire(%[[VAL_30]] : DMA, %[[VAL_243]] : DMA)
// CHECK:           AIE.wire(%[[VAL_242]] : North, %[[VAL_243]] : South)
// CHECK:           AIE.wire(%[[VAL_238]] : East, %[[VAL_244:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_31]] : Core, %[[VAL_244]] : Core)
// CHECK:           AIE.wire(%[[VAL_31]] : DMA, %[[VAL_244]] : DMA)
// CHECK:           AIE.wire(%[[VAL_243]] : North, %[[VAL_244]] : South)
// CHECK:           AIE.wire(%[[VAL_239]] : East, %[[VAL_245:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_32]] : Core, %[[VAL_245]] : Core)
// CHECK:           AIE.wire(%[[VAL_32]] : DMA, %[[VAL_245]] : DMA)
// CHECK:           AIE.wire(%[[VAL_244]] : North, %[[VAL_245]] : South)
// CHECK:           AIE.wire(%[[VAL_240]] : East, %[[VAL_246:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_33]] : Core, %[[VAL_246]] : Core)
// CHECK:           AIE.wire(%[[VAL_33]] : DMA, %[[VAL_246]] : DMA)
// CHECK:           AIE.wire(%[[VAL_245]] : North, %[[VAL_246]] : South)
// CHECK:           AIE.wire(%[[VAL_34]] : Core, %[[VAL_247:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_34]] : DMA, %[[VAL_247]] : DMA)
// CHECK:           AIE.wire(%[[VAL_246]] : North, %[[VAL_247]] : South)
// CHECK:           AIE.wire(%[[VAL_242]] : East, %[[VAL_248:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_249:.*]] : North, %[[VAL_248]] : South)
// CHECK:           AIE.wire(%[[VAL_3]] : DMA, %[[VAL_249]] : DMA)
// CHECK:           AIE.wire(%[[VAL_243]] : East, %[[VAL_250:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_38]] : Core, %[[VAL_250]] : Core)
// CHECK:           AIE.wire(%[[VAL_38]] : DMA, %[[VAL_250]] : DMA)
// CHECK:           AIE.wire(%[[VAL_248]] : North, %[[VAL_250]] : South)
// CHECK:           AIE.wire(%[[VAL_244]] : East, %[[VAL_251:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_39]] : Core, %[[VAL_251]] : Core)
// CHECK:           AIE.wire(%[[VAL_39]] : DMA, %[[VAL_251]] : DMA)
// CHECK:           AIE.wire(%[[VAL_250]] : North, %[[VAL_251]] : South)
// CHECK:           AIE.wire(%[[VAL_245]] : East, %[[VAL_252:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_40]] : Core, %[[VAL_252]] : Core)
// CHECK:           AIE.wire(%[[VAL_40]] : DMA, %[[VAL_252]] : DMA)
// CHECK:           AIE.wire(%[[VAL_251]] : North, %[[VAL_252]] : South)
// CHECK:           AIE.wire(%[[VAL_246]] : East, %[[VAL_253:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_41]] : Core, %[[VAL_253]] : Core)
// CHECK:           AIE.wire(%[[VAL_41]] : DMA, %[[VAL_253]] : DMA)
// CHECK:           AIE.wire(%[[VAL_252]] : North, %[[VAL_253]] : South)
// CHECK:           AIE.wire(%[[VAL_247]] : East, %[[VAL_254:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_42]] : Core, %[[VAL_254]] : Core)
// CHECK:           AIE.wire(%[[VAL_42]] : DMA, %[[VAL_254]] : DMA)
// CHECK:           AIE.wire(%[[VAL_253]] : North, %[[VAL_254]] : South)
// CHECK:           AIE.wire(%[[VAL_250]] : East, %[[VAL_255:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_46]] : Core, %[[VAL_255]] : Core)
// CHECK:           AIE.wire(%[[VAL_46]] : DMA, %[[VAL_255]] : DMA)
// CHECK:           AIE.wire(%[[VAL_251]] : East, %[[VAL_256:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_47]] : Core, %[[VAL_256]] : Core)
// CHECK:           AIE.wire(%[[VAL_47]] : DMA, %[[VAL_256]] : DMA)
// CHECK:           AIE.wire(%[[VAL_255]] : North, %[[VAL_256]] : South)
// CHECK:           AIE.wire(%[[VAL_252]] : East, %[[VAL_257:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_48]] : Core, %[[VAL_257]] : Core)
// CHECK:           AIE.wire(%[[VAL_48]] : DMA, %[[VAL_257]] : DMA)
// CHECK:           AIE.wire(%[[VAL_256]] : North, %[[VAL_257]] : South)
// CHECK:           AIE.wire(%[[VAL_253]] : East, %[[VAL_258:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_49]] : Core, %[[VAL_258]] : Core)
// CHECK:           AIE.wire(%[[VAL_49]] : DMA, %[[VAL_258]] : DMA)
// CHECK:           AIE.wire(%[[VAL_257]] : North, %[[VAL_258]] : South)
// CHECK:           AIE.wire(%[[VAL_254]] : East, %[[VAL_259:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_50]] : Core, %[[VAL_259]] : Core)
// CHECK:           AIE.wire(%[[VAL_50]] : DMA, %[[VAL_259]] : DMA)
// CHECK:           AIE.wire(%[[VAL_258]] : North, %[[VAL_259]] : South)
// CHECK:           AIE.wire(%[[VAL_51]] : Core, %[[VAL_260:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_51]] : DMA, %[[VAL_260]] : DMA)
// CHECK:           AIE.wire(%[[VAL_259]] : North, %[[VAL_260]] : South)
// CHECK:           AIE.wire(%[[VAL_255]] : East, %[[VAL_261:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_54]] : Core, %[[VAL_261]] : Core)
// CHECK:           AIE.wire(%[[VAL_54]] : DMA, %[[VAL_261]] : DMA)
// CHECK:           AIE.wire(%[[VAL_262:.*]] : North, %[[VAL_261]] : South)
// CHECK:           AIE.wire(%[[VAL_256]] : East, %[[VAL_263:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_55]] : Core, %[[VAL_263]] : Core)
// CHECK:           AIE.wire(%[[VAL_55]] : DMA, %[[VAL_263]] : DMA)
// CHECK:           AIE.wire(%[[VAL_261]] : North, %[[VAL_263]] : South)
// CHECK:           AIE.wire(%[[VAL_257]] : East, %[[VAL_264:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_56]] : Core, %[[VAL_264]] : Core)
// CHECK:           AIE.wire(%[[VAL_56]] : DMA, %[[VAL_264]] : DMA)
// CHECK:           AIE.wire(%[[VAL_263]] : North, %[[VAL_264]] : South)
// CHECK:           AIE.wire(%[[VAL_258]] : East, %[[VAL_265:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_57]] : Core, %[[VAL_265]] : Core)
// CHECK:           AIE.wire(%[[VAL_57]] : DMA, %[[VAL_265]] : DMA)
// CHECK:           AIE.wire(%[[VAL_264]] : North, %[[VAL_265]] : South)
// CHECK:           AIE.wire(%[[VAL_259]] : East, %[[VAL_266:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_58]] : Core, %[[VAL_266]] : Core)
// CHECK:           AIE.wire(%[[VAL_58]] : DMA, %[[VAL_266]] : DMA)
// CHECK:           AIE.wire(%[[VAL_265]] : North, %[[VAL_266]] : South)
// CHECK:           AIE.wire(%[[VAL_260]] : East, %[[VAL_267:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_59]] : Core, %[[VAL_267]] : Core)
// CHECK:           AIE.wire(%[[VAL_59]] : DMA, %[[VAL_267]] : DMA)
// CHECK:           AIE.wire(%[[VAL_266]] : North, %[[VAL_267]] : South)
// CHECK:           AIE.wire(%[[VAL_262]] : East, %[[VAL_268:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_269:.*]] : North, %[[VAL_268]] : South)
// CHECK:           AIE.wire(%[[VAL_6]] : DMA, %[[VAL_269]] : DMA)
// CHECK:           AIE.wire(%[[VAL_261]] : East, %[[VAL_270:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_62]] : Core, %[[VAL_270]] : Core)
// CHECK:           AIE.wire(%[[VAL_62]] : DMA, %[[VAL_270]] : DMA)
// CHECK:           AIE.wire(%[[VAL_268]] : North, %[[VAL_270]] : South)
// CHECK:           AIE.wire(%[[VAL_263]] : East, %[[VAL_271:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_63]] : Core, %[[VAL_271]] : Core)
// CHECK:           AIE.wire(%[[VAL_63]] : DMA, %[[VAL_271]] : DMA)
// CHECK:           AIE.wire(%[[VAL_270]] : North, %[[VAL_271]] : South)
// CHECK:           AIE.wire(%[[VAL_264]] : East, %[[VAL_272:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_64]] : Core, %[[VAL_272]] : Core)
// CHECK:           AIE.wire(%[[VAL_64]] : DMA, %[[VAL_272]] : DMA)
// CHECK:           AIE.wire(%[[VAL_271]] : North, %[[VAL_272]] : South)
// CHECK:           AIE.wire(%[[VAL_265]] : East, %[[VAL_273:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_65]] : Core, %[[VAL_273]] : Core)
// CHECK:           AIE.wire(%[[VAL_65]] : DMA, %[[VAL_273]] : DMA)
// CHECK:           AIE.wire(%[[VAL_272]] : North, %[[VAL_273]] : South)
// CHECK:           AIE.wire(%[[VAL_266]] : East, %[[VAL_274:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_66]] : Core, %[[VAL_274]] : Core)
// CHECK:           AIE.wire(%[[VAL_66]] : DMA, %[[VAL_274]] : DMA)
// CHECK:           AIE.wire(%[[VAL_273]] : North, %[[VAL_274]] : South)
// CHECK:           AIE.wire(%[[VAL_267]] : East, %[[VAL_275:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_67]] : Core, %[[VAL_275]] : Core)
// CHECK:           AIE.wire(%[[VAL_67]] : DMA, %[[VAL_275]] : DMA)
// CHECK:           AIE.wire(%[[VAL_274]] : North, %[[VAL_275]] : South)
// CHECK:           AIE.wire(%[[VAL_270]] : East, %[[VAL_276:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_70]] : Core, %[[VAL_276]] : Core)
// CHECK:           AIE.wire(%[[VAL_70]] : DMA, %[[VAL_276]] : DMA)
// CHECK:           AIE.wire(%[[VAL_271]] : East, %[[VAL_277:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_71]] : Core, %[[VAL_277]] : Core)
// CHECK:           AIE.wire(%[[VAL_71]] : DMA, %[[VAL_277]] : DMA)
// CHECK:           AIE.wire(%[[VAL_276]] : North, %[[VAL_277]] : South)
// CHECK:           AIE.wire(%[[VAL_272]] : East, %[[VAL_278:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_72]] : Core, %[[VAL_278]] : Core)
// CHECK:           AIE.wire(%[[VAL_72]] : DMA, %[[VAL_278]] : DMA)
// CHECK:           AIE.wire(%[[VAL_277]] : North, %[[VAL_278]] : South)
// CHECK:           AIE.wire(%[[VAL_273]] : East, %[[VAL_279:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_73]] : Core, %[[VAL_279]] : Core)
// CHECK:           AIE.wire(%[[VAL_73]] : DMA, %[[VAL_279]] : DMA)
// CHECK:           AIE.wire(%[[VAL_278]] : North, %[[VAL_279]] : South)
// CHECK:           AIE.wire(%[[VAL_274]] : East, %[[VAL_280:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_74]] : Core, %[[VAL_280]] : Core)
// CHECK:           AIE.wire(%[[VAL_74]] : DMA, %[[VAL_280]] : DMA)
// CHECK:           AIE.wire(%[[VAL_279]] : North, %[[VAL_280]] : South)
// CHECK:           AIE.wire(%[[VAL_275]] : East, %[[VAL_281:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_75]] : Core, %[[VAL_281]] : Core)
// CHECK:           AIE.wire(%[[VAL_75]] : DMA, %[[VAL_281]] : DMA)
// CHECK:           AIE.wire(%[[VAL_280]] : North, %[[VAL_281]] : South)
// CHECK:           AIE.wire(%[[VAL_276]] : East, %[[VAL_282:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_78]] : Core, %[[VAL_282]] : Core)
// CHECK:           AIE.wire(%[[VAL_78]] : DMA, %[[VAL_282]] : DMA)
// CHECK:           AIE.wire(%[[VAL_277]] : East, %[[VAL_283:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_79]] : Core, %[[VAL_283]] : Core)
// CHECK:           AIE.wire(%[[VAL_79]] : DMA, %[[VAL_283]] : DMA)
// CHECK:           AIE.wire(%[[VAL_282]] : North, %[[VAL_283]] : South)
// CHECK:           AIE.wire(%[[VAL_278]] : East, %[[VAL_284:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_80]] : Core, %[[VAL_284]] : Core)
// CHECK:           AIE.wire(%[[VAL_80]] : DMA, %[[VAL_284]] : DMA)
// CHECK:           AIE.wire(%[[VAL_283]] : North, %[[VAL_284]] : South)
// CHECK:           AIE.wire(%[[VAL_279]] : East, %[[VAL_285:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_81]] : Core, %[[VAL_285]] : Core)
// CHECK:           AIE.wire(%[[VAL_81]] : DMA, %[[VAL_285]] : DMA)
// CHECK:           AIE.wire(%[[VAL_284]] : North, %[[VAL_285]] : South)
// CHECK:           AIE.wire(%[[VAL_282]] : East, %[[VAL_286:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_86]] : Core, %[[VAL_286]] : Core)
// CHECK:           AIE.wire(%[[VAL_86]] : DMA, %[[VAL_286]] : DMA)
// CHECK:           AIE.wire(%[[VAL_283]] : East, %[[VAL_287:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_87]] : Core, %[[VAL_287]] : Core)
// CHECK:           AIE.wire(%[[VAL_87]] : DMA, %[[VAL_287]] : DMA)
// CHECK:           AIE.wire(%[[VAL_286]] : North, %[[VAL_287]] : South)
// CHECK:           AIE.wire(%[[VAL_284]] : East, %[[VAL_288:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_88]] : Core, %[[VAL_288]] : Core)
// CHECK:           AIE.wire(%[[VAL_88]] : DMA, %[[VAL_288]] : DMA)
// CHECK:           AIE.wire(%[[VAL_287]] : North, %[[VAL_288]] : South)
// CHECK:           AIE.wire(%[[VAL_285]] : East, %[[VAL_289:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_89]] : Core, %[[VAL_289]] : Core)
// CHECK:           AIE.wire(%[[VAL_89]] : DMA, %[[VAL_289]] : DMA)
// CHECK:           AIE.wire(%[[VAL_288]] : North, %[[VAL_289]] : South)
// CHECK:           AIE.wire(%[[VAL_290:.*]] : North, %[[VAL_291:.*]] : South)
// CHECK:           AIE.wire(%[[VAL_10]] : DMA, %[[VAL_290]] : DMA)
// CHECK:           AIE.wire(%[[VAL_286]] : East, %[[VAL_292:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_94]] : Core, %[[VAL_292]] : Core)
// CHECK:           AIE.wire(%[[VAL_94]] : DMA, %[[VAL_292]] : DMA)
// CHECK:           AIE.wire(%[[VAL_291]] : North, %[[VAL_292]] : South)
// CHECK:           AIE.wire(%[[VAL_287]] : East, %[[VAL_293:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_95]] : Core, %[[VAL_293]] : Core)
// CHECK:           AIE.wire(%[[VAL_95]] : DMA, %[[VAL_293]] : DMA)
// CHECK:           AIE.wire(%[[VAL_292]] : North, %[[VAL_293]] : South)
// CHECK:           AIE.wire(%[[VAL_288]] : East, %[[VAL_294:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_96]] : Core, %[[VAL_294]] : Core)
// CHECK:           AIE.wire(%[[VAL_96]] : DMA, %[[VAL_294]] : DMA)
// CHECK:           AIE.wire(%[[VAL_293]] : North, %[[VAL_294]] : South)
// CHECK:           AIE.wire(%[[VAL_289]] : East, %[[VAL_295:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_97]] : Core, %[[VAL_295]] : Core)
// CHECK:           AIE.wire(%[[VAL_97]] : DMA, %[[VAL_295]] : DMA)
// CHECK:           AIE.wire(%[[VAL_294]] : North, %[[VAL_295]] : South)
// CHECK:           AIE.wire(%[[VAL_291]] : East, %[[VAL_296:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_297:.*]] : North, %[[VAL_296]] : South)
// CHECK:           AIE.wire(%[[VAL_11]] : DMA, %[[VAL_297]] : DMA)
// CHECK:           AIE.wire(%[[VAL_292]] : East, %[[VAL_298:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_102]] : Core, %[[VAL_298]] : Core)
// CHECK:           AIE.wire(%[[VAL_102]] : DMA, %[[VAL_298]] : DMA)
// CHECK:           AIE.wire(%[[VAL_296]] : North, %[[VAL_298]] : South)
// CHECK:           AIE.wire(%[[VAL_293]] : East, %[[VAL_299:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_103]] : Core, %[[VAL_299]] : Core)
// CHECK:           AIE.wire(%[[VAL_103]] : DMA, %[[VAL_299]] : DMA)
// CHECK:           AIE.wire(%[[VAL_298]] : North, %[[VAL_299]] : South)
// CHECK:           AIE.wire(%[[VAL_294]] : East, %[[VAL_300:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_104]] : Core, %[[VAL_300]] : Core)
// CHECK:           AIE.wire(%[[VAL_104]] : DMA, %[[VAL_300]] : DMA)
// CHECK:           AIE.wire(%[[VAL_299]] : North, %[[VAL_300]] : South)
// CHECK:           AIE.wire(%[[VAL_295]] : East, %[[VAL_301:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_105]] : Core, %[[VAL_301]] : Core)
// CHECK:           AIE.wire(%[[VAL_105]] : DMA, %[[VAL_301]] : DMA)
// CHECK:           AIE.wire(%[[VAL_300]] : North, %[[VAL_301]] : South)
// CHECK:           AIE.wire(%[[VAL_296]] : East, %[[VAL_302:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_298]] : East, %[[VAL_303:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_110]] : Core, %[[VAL_303]] : Core)
// CHECK:           AIE.wire(%[[VAL_110]] : DMA, %[[VAL_303]] : DMA)
// CHECK:           AIE.wire(%[[VAL_302]] : North, %[[VAL_303]] : South)
// CHECK:           AIE.wire(%[[VAL_299]] : East, %[[VAL_304:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_111]] : Core, %[[VAL_304]] : Core)
// CHECK:           AIE.wire(%[[VAL_111]] : DMA, %[[VAL_304]] : DMA)
// CHECK:           AIE.wire(%[[VAL_303]] : North, %[[VAL_304]] : South)
// CHECK:           AIE.wire(%[[VAL_300]] : East, %[[VAL_305:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_112]] : Core, %[[VAL_305]] : Core)
// CHECK:           AIE.wire(%[[VAL_112]] : DMA, %[[VAL_305]] : DMA)
// CHECK:           AIE.wire(%[[VAL_304]] : North, %[[VAL_305]] : South)
// CHECK:           AIE.wire(%[[VAL_301]] : East, %[[VAL_306:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_113]] : Core, %[[VAL_306]] : Core)
// CHECK:           AIE.wire(%[[VAL_113]] : DMA, %[[VAL_306]] : DMA)
// CHECK:           AIE.wire(%[[VAL_305]] : North, %[[VAL_306]] : South)
// CHECK:           AIE.wire(%[[VAL_114]] : Core, %[[VAL_307:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_114]] : DMA, %[[VAL_307]] : DMA)
// CHECK:           AIE.wire(%[[VAL_306]] : North, %[[VAL_307]] : South)
// CHECK:           AIE.wire(%[[VAL_302]] : East, %[[VAL_308:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_303]] : East, %[[VAL_309:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_119]] : Core, %[[VAL_309]] : Core)
// CHECK:           AIE.wire(%[[VAL_119]] : DMA, %[[VAL_309]] : DMA)
// CHECK:           AIE.wire(%[[VAL_308]] : North, %[[VAL_309]] : South)
// CHECK:           AIE.wire(%[[VAL_304]] : East, %[[VAL_310:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_120]] : Core, %[[VAL_310]] : Core)
// CHECK:           AIE.wire(%[[VAL_120]] : DMA, %[[VAL_310]] : DMA)
// CHECK:           AIE.wire(%[[VAL_309]] : North, %[[VAL_310]] : South)
// CHECK:           AIE.wire(%[[VAL_305]] : East, %[[VAL_311:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_121]] : Core, %[[VAL_311]] : Core)
// CHECK:           AIE.wire(%[[VAL_121]] : DMA, %[[VAL_311]] : DMA)
// CHECK:           AIE.wire(%[[VAL_310]] : North, %[[VAL_311]] : South)
// CHECK:           AIE.wire(%[[VAL_306]] : East, %[[VAL_312:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_122]] : Core, %[[VAL_312]] : Core)
// CHECK:           AIE.wire(%[[VAL_122]] : DMA, %[[VAL_312]] : DMA)
// CHECK:           AIE.wire(%[[VAL_311]] : North, %[[VAL_312]] : South)
// CHECK:           AIE.wire(%[[VAL_307]] : East, %[[VAL_313:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_123]] : Core, %[[VAL_313]] : Core)
// CHECK:           AIE.wire(%[[VAL_123]] : DMA, %[[VAL_313]] : DMA)
// CHECK:           AIE.wire(%[[VAL_312]] : North, %[[VAL_313]] : South)
// CHECK:           AIE.wire(%[[VAL_310]] : East, %[[VAL_314:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_128]] : Core, %[[VAL_314]] : Core)
// CHECK:           AIE.wire(%[[VAL_128]] : DMA, %[[VAL_314]] : DMA)
// CHECK:           AIE.wire(%[[VAL_311]] : East, %[[VAL_315:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_129]] : Core, %[[VAL_315]] : Core)
// CHECK:           AIE.wire(%[[VAL_129]] : DMA, %[[VAL_315]] : DMA)
// CHECK:           AIE.wire(%[[VAL_314]] : North, %[[VAL_315]] : South)
// CHECK:           AIE.wire(%[[VAL_312]] : East, %[[VAL_316:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_130]] : Core, %[[VAL_316]] : Core)
// CHECK:           AIE.wire(%[[VAL_130]] : DMA, %[[VAL_316]] : DMA)
// CHECK:           AIE.wire(%[[VAL_315]] : North, %[[VAL_316]] : South)
// CHECK:           AIE.wire(%[[VAL_313]] : East, %[[VAL_317:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_131]] : Core, %[[VAL_317]] : Core)
// CHECK:           AIE.wire(%[[VAL_131]] : DMA, %[[VAL_317]] : DMA)
// CHECK:           AIE.wire(%[[VAL_316]] : North, %[[VAL_317]] : South)
// CHECK:           AIE.wire(%[[VAL_314]] : East, %[[VAL_318:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_221]] : Core, %[[VAL_318]] : Core)
// CHECK:           AIE.wire(%[[VAL_221]] : DMA, %[[VAL_318]] : DMA)
// CHECK:           AIE.wire(%[[VAL_318]] : East, %[[VAL_319:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_223]] : Core, %[[VAL_319]] : Core)
// CHECK:           AIE.wire(%[[VAL_223]] : DMA, %[[VAL_319]] : DMA)
// CHECK:           AIE.wire(%[[VAL_227]] : Core, %[[VAL_320:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_227]] : DMA, %[[VAL_320]] : DMA)
// CHECK:           AIE.wire(%[[VAL_321:.*]] : North, %[[VAL_320]] : South)
// CHECK:           AIE.wire(%[[VAL_319]] : East, %[[VAL_322:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_229]] : Core, %[[VAL_322]] : Core)
// CHECK:           AIE.wire(%[[VAL_229]] : DMA, %[[VAL_322]] : DMA)
// CHECK:           AIE.wire(%[[VAL_320]] : North, %[[VAL_322]] : South)
// CHECK:           AIE.wire(%[[VAL_321]] : East, %[[VAL_323:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_324:.*]] : North, %[[VAL_323]] : South)
// CHECK:           AIE.wire(%[[VAL_12]] : DMA, %[[VAL_324]] : DMA)
// CHECK:         }

module {
  AIE.device(xcvc1902) {
    %tile_0_0 = AIE.tile(0, 0)
    %tile_1_0 = AIE.tile(1, 0)
    %tile_2_0 = AIE.tile(2, 0)
    %tile_3_0 = AIE.tile(3, 0)
    %tile_4_0 = AIE.tile(4, 0)
    %tile_5_0 = AIE.tile(5, 0)
    %tile_6_0 = AIE.tile(6, 0)
    %tile_7_0 = AIE.tile(7, 0)
    %tile_8_0 = AIE.tile(8, 0)
    %tile_9_0 = AIE.tile(9, 0)
    %tile_10_0 = AIE.tile(10, 0)
    %tile_11_0 = AIE.tile(11, 0)
    %tile_18_0 = AIE.tile(18, 0)
    %tile_19_0 = AIE.tile(19, 0)
    %tile_0_1 = AIE.tile(0, 1)
    %tile_0_2 = AIE.tile(0, 2)
    %tile_0_3 = AIE.tile(0, 3)
    %tile_0_4 = AIE.tile(0, 4)
    %tile_0_5 = AIE.tile(0, 5)
    %tile_0_6 = AIE.tile(0, 6)
    %tile_0_7 = AIE.tile(0, 7)
    %tile_0_8 = AIE.tile(0, 8)
    %tile_1_1 = AIE.tile(1, 1)
    %tile_1_2 = AIE.tile(1, 2)
    %tile_1_3 = AIE.tile(1, 3)
    %tile_1_4 = AIE.tile(1, 4)
    %tile_1_5 = AIE.tile(1, 5)
    %tile_1_6 = AIE.tile(1, 6)
    %tile_1_7 = AIE.tile(1, 7)
    %tile_1_8 = AIE.tile(1, 8)
    %tile_2_1 = AIE.tile(2, 1)
    %tile_2_2 = AIE.tile(2, 2)
    %tile_2_3 = AIE.tile(2, 3)
    %tile_2_4 = AIE.tile(2, 4)
    %tile_2_5 = AIE.tile(2, 5)
    %tile_2_6 = AIE.tile(2, 6)
    %tile_2_7 = AIE.tile(2, 7)
    %tile_2_8 = AIE.tile(2, 8)
    %tile_3_1 = AIE.tile(3, 1)
    %tile_3_2 = AIE.tile(3, 2)
    %tile_3_3 = AIE.tile(3, 3)
    %tile_3_4 = AIE.tile(3, 4)
    %tile_3_5 = AIE.tile(3, 5)
    %tile_3_6 = AIE.tile(3, 6)
    %tile_3_7 = AIE.tile(3, 7)
    %tile_3_8 = AIE.tile(3, 8)
    %tile_4_1 = AIE.tile(4, 1)
    %tile_4_2 = AIE.tile(4, 2)
    %tile_4_3 = AIE.tile(4, 3)
    %tile_4_4 = AIE.tile(4, 4)
    %tile_4_5 = AIE.tile(4, 5)
    %tile_4_6 = AIE.tile(4, 6)
    %tile_4_7 = AIE.tile(4, 7)
    %tile_4_8 = AIE.tile(4, 8)
    %tile_5_1 = AIE.tile(5, 1)
    %tile_5_2 = AIE.tile(5, 2)
    %tile_5_3 = AIE.tile(5, 3)
    %tile_5_4 = AIE.tile(5, 4)
    %tile_5_5 = AIE.tile(5, 5)
    %tile_5_6 = AIE.tile(5, 6)
    %tile_5_7 = AIE.tile(5, 7)
    %tile_5_8 = AIE.tile(5, 8)
    %tile_6_1 = AIE.tile(6, 1)
    %tile_6_2 = AIE.tile(6, 2)
    %tile_6_3 = AIE.tile(6, 3)
    %tile_6_4 = AIE.tile(6, 4)
    %tile_6_5 = AIE.tile(6, 5)
    %tile_6_6 = AIE.tile(6, 6)
    %tile_6_7 = AIE.tile(6, 7)
    %tile_6_8 = AIE.tile(6, 8)
    %tile_7_1 = AIE.tile(7, 1)
    %tile_7_2 = AIE.tile(7, 2)
    %tile_7_3 = AIE.tile(7, 3)
    %tile_7_4 = AIE.tile(7, 4)
    %tile_7_5 = AIE.tile(7, 5)
    %tile_7_6 = AIE.tile(7, 6)
    %tile_7_7 = AIE.tile(7, 7)
    %tile_7_8 = AIE.tile(7, 8)
    %tile_8_1 = AIE.tile(8, 1)
    %tile_8_2 = AIE.tile(8, 2)
    %tile_8_3 = AIE.tile(8, 3)
    %tile_8_4 = AIE.tile(8, 4)
    %tile_8_5 = AIE.tile(8, 5)
    %tile_8_6 = AIE.tile(8, 6)
    %tile_8_7 = AIE.tile(8, 7)
    %tile_8_8 = AIE.tile(8, 8)
    %tile_9_1 = AIE.tile(9, 1)
    %tile_9_2 = AIE.tile(9, 2)
    %tile_9_3 = AIE.tile(9, 3)
    %tile_9_4 = AIE.tile(9, 4)
    %tile_9_5 = AIE.tile(9, 5)
    %tile_9_6 = AIE.tile(9, 6)
    %tile_9_7 = AIE.tile(9, 7)
    %tile_9_8 = AIE.tile(9, 8)
    %tile_10_1 = AIE.tile(10, 1)
    %tile_10_2 = AIE.tile(10, 2)
    %tile_10_3 = AIE.tile(10, 3)
    %tile_10_4 = AIE.tile(10, 4)
    %tile_10_5 = AIE.tile(10, 5)
    %tile_10_6 = AIE.tile(10, 6)
    %tile_10_7 = AIE.tile(10, 7)
    %tile_10_8 = AIE.tile(10, 8)
    %tile_11_1 = AIE.tile(11, 1)
    %tile_11_2 = AIE.tile(11, 2)
    %tile_11_3 = AIE.tile(11, 3)
    %tile_11_4 = AIE.tile(11, 4)
    %tile_11_5 = AIE.tile(11, 5)
    %tile_11_6 = AIE.tile(11, 6)
    %tile_11_7 = AIE.tile(11, 7)
    %tile_11_8 = AIE.tile(11, 8)
    %tile_12_1 = AIE.tile(12, 1)
    %tile_12_2 = AIE.tile(12, 2)
    %tile_12_3 = AIE.tile(12, 3)
    %tile_12_4 = AIE.tile(12, 4)
    %tile_12_5 = AIE.tile(12, 5)
    %tile_12_6 = AIE.tile(12, 6)
    %tile_12_7 = AIE.tile(12, 7)
    %tile_12_8 = AIE.tile(12, 8)
    %tile_13_0 = AIE.tile(13, 0)
    %tile_13_1 = AIE.tile(13, 1)
    %tile_13_2 = AIE.tile(13, 2)
    %tile_13_3 = AIE.tile(13, 3)
    %tile_13_4 = AIE.tile(13, 4)
    %tile_13_5 = AIE.tile(13, 5)
    %tile_13_6 = AIE.tile(13, 6)
    %tile_13_7 = AIE.tile(13, 7)
    %tile_13_8 = AIE.tile(13, 8)
    %tile_14_1 = AIE.tile(14, 1)
    %tile_14_2 = AIE.tile(14, 2)
    %tile_14_3 = AIE.tile(14, 3)
    %tile_14_4 = AIE.tile(14, 4)
    %tile_14_5 = AIE.tile(14, 5)
    %tile_14_6 = AIE.tile(14, 6)
    %tile_14_7 = AIE.tile(14, 7)
    %tile_14_8 = AIE.tile(14, 8)
    %switchbox_0_1 = AIE.switchbox(%tile_0_1) {
    }
    %switchbox_0_2 = AIE.switchbox(%tile_0_2) {
    }
    %switchbox_0_3 = AIE.switchbox(%tile_0_3) {
    }
    %switchbox_0_4 = AIE.switchbox(%tile_0_4) {
    }
    %switchbox_1_1 = AIE.switchbox(%tile_1_1) {
    }
    %switchbox_1_2 = AIE.switchbox(%tile_1_2) {
    }
    %switchbox_1_3 = AIE.switchbox(%tile_1_3) {
    }
    %switchbox_1_4 = AIE.switchbox(%tile_1_4) {
    }
    %switchbox_2_1 = AIE.switchbox(%tile_2_1) {
    }
    %switchbox_2_2 = AIE.switchbox(%tile_2_2) {
    }
    %switchbox_2_3 = AIE.switchbox(%tile_2_3) {
    }
    %switchbox_2_4 = AIE.switchbox(%tile_2_4) {
      AIE.connect<East : 0, North : 0>
    }
    %switchbox_2_5 = AIE.switchbox(%tile_2_5) {
      AIE.connect<South : 0, Core : 0>
      AIE.connect<DMA : 0, East : 0>
    }
    %switchbox_3_1 = AIE.switchbox(%tile_3_1) {
      AIE.connect<South : 0, DMA : 0>
      AIE.connect<Core : 0, North : 0>
    }
    %switchbox_3_2 = AIE.switchbox(%tile_3_2) {
      AIE.connect<South : 0, North : 0>
    }
    %switchbox_3_3 = AIE.switchbox(%tile_3_3) {
      AIE.connect<South : 0, North : 0>
    }
    %switchbox_3_4 = AIE.switchbox(%tile_3_4) {
      AIE.connect<South : 0, West : 0>
    }
    %switchbox_3_5 = AIE.switchbox(%tile_3_5) {
      AIE.connect<West : 0, East : 0>
    }
    %switchbox_4_1 = AIE.switchbox(%tile_4_1) {
    }
    %switchbox_4_2 = AIE.switchbox(%tile_4_2) {
    }
    %switchbox_4_3 = AIE.switchbox(%tile_4_3) {
    }
    %switchbox_4_4 = AIE.switchbox(%tile_4_4) {
    }
    %switchbox_5_1 = AIE.switchbox(%tile_5_1) {
    }
    %switchbox_5_2 = AIE.switchbox(%tile_5_2) {
    }
    %switchbox_5_3 = AIE.switchbox(%tile_5_3) {
    }
    %switchbox_5_4 = AIE.switchbox(%tile_5_4) {
    }
    %switchbox_5_5 = AIE.switchbox(%tile_5_5) {
    }
    %switchbox_5_6 = AIE.switchbox(%tile_5_6) {
      AIE.connect<East : 0, West : 0>
    }
    %switchbox_6_1 = AIE.switchbox(%tile_6_1) {
    }
    %switchbox_6_2 = AIE.switchbox(%tile_6_2) {
    }
    %switchbox_6_3 = AIE.switchbox(%tile_6_3) {
    }
    %switchbox_6_4 = AIE.switchbox(%tile_6_4) {
    }
    %switchbox_6_5 = AIE.switchbox(%tile_6_5) {
    }
    %switchbox_6_6 = AIE.switchbox(%tile_6_6) {
      AIE.connect<East : 0, Core : 0>
      AIE.connect<DMA : 0, West : 0>
    }
    %switchbox_7_1 = AIE.switchbox(%tile_7_1) {
    }
    %switchbox_7_2 = AIE.switchbox(%tile_7_2) {
    }
    %switchbox_7_3 = AIE.switchbox(%tile_7_3) {
      AIE.connect<East : 0, DMA : 0>
      AIE.connect<Core : 0, North : 0>
    }
    %switchbox_7_4 = AIE.switchbox(%tile_7_4) {
      AIE.connect<South : 0, North : 0>
    }
    %switchbox_7_5 = AIE.switchbox(%tile_7_5) {
      AIE.connect<South : 0, North : 0>
    }
    %switchbox_7_6 = AIE.switchbox(%tile_7_6) {
      AIE.connect<South : 0, West : 0>
    }
    %switchbox_8_1 = AIE.switchbox(%tile_8_1) {
    }
    %switchbox_8_2 = AIE.switchbox(%tile_8_2) {
    }
    %switchbox_8_3 = AIE.switchbox(%tile_8_3) {
      AIE.connect<East : 0, West : 0>
    }
    %switchbox_8_4 = AIE.switchbox(%tile_8_4) {
    }
    %switchbox_9_1 = AIE.switchbox(%tile_9_1) {
    }
    %switchbox_9_2 = AIE.switchbox(%tile_9_2) {
    }
    %switchbox_9_3 = AIE.switchbox(%tile_9_3) {
    }
    %switchbox_9_4 = AIE.switchbox(%tile_9_4) {
    }
    %switchbox_10_1 = AIE.switchbox(%tile_10_1) {
    }
    %switchbox_10_2 = AIE.switchbox(%tile_10_2) {
    }
    %switchbox_10_3 = AIE.switchbox(%tile_10_3) {
    }
    %switchbox_10_4 = AIE.switchbox(%tile_10_4) {
    }
    %switchbox_11_1 = AIE.switchbox(%tile_11_1) {
    }
    %switchbox_11_2 = AIE.switchbox(%tile_11_2) {
    }
    %switchbox_11_3 = AIE.switchbox(%tile_11_3) {
    }
    %switchbox_11_4 = AIE.switchbox(%tile_11_4) {
    }
    %switchbox_12_1 = AIE.switchbox(%tile_12_1) {
    }
    %switchbox_12_2 = AIE.switchbox(%tile_12_2) {
    }
    %switchbox_12_3 = AIE.switchbox(%tile_12_3) {
    }
    %switchbox_12_4 = AIE.switchbox(%tile_12_4) {
    }
    %switchbox_12_5 = AIE.switchbox(%tile_12_5) {
      AIE.connect<East : 0, Core : 0>
      AIE.connect<DMA : 0, East : 0>
    }
    %switchbox_13_1 = AIE.switchbox(%tile_13_1) {
      AIE.connect<South : 0, North : 0>
    }
    %switchbox_13_2 = AIE.switchbox(%tile_13_2) {
      AIE.connect<South : 0, North : 0>
    }
    %switchbox_13_3 = AIE.switchbox(%tile_13_3) {
      AIE.connect<South : 0, DMA : 0>
      AIE.connect<Core : 0, North : 0>
    }
    %switchbox_13_4 = AIE.switchbox(%tile_13_4) {
      AIE.connect<South : 0, North : 0>
    }
    %switchbox_13_5 = AIE.switchbox(%tile_13_5) {
      AIE.connect<South : 0, West : 0>
      AIE.connect<West : 0, East : 0>
    }
    AIE.flow(%tile_3_0, DMA : 0, %tile_3_0, North : 0)
    AIE.flow(%tile_4_5, West : 0, %tile_6_0, DMA : 0)
    AIE.flow(%tile_10_0, DMA : 0, %tile_9_3, West : 0)
    AIE.flow(%tile_4_6, East : 0, %tile_2_0, DMA : 0)
    AIE.flow(%tile_11_0, DMA : 0, %tile_13_0, North : 0)
    AIE.flow(%tile_14_5, West : 0, %tile_18_0, DMA : 0)
  }
}
