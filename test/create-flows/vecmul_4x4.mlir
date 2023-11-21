//===- vecmul_4x4.mlir -----------------------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_0:.*]] = AIE.tile(47, 2)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(47, 1)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(47, 0)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_4:.*]] = AIE.tile(10, 5)
// CHECK:           %[[VAL_5:.*]] = AIE.lock(%[[VAL_4]], 2)
// CHECK:           %[[VAL_6:.*]] = AIE.buffer(%[[VAL_4]]) {sym_name = "buf47"} : memref<64xi32, 2>
// CHECK:           %[[VAL_7:.*]] = AIE.lock(%[[VAL_4]], 1)
// CHECK:           %[[VAL_8:.*]] = AIE.buffer(%[[VAL_4]]) {sym_name = "buf46"} : memref<64xi32, 2>
// CHECK:           %[[VAL_9:.*]] = AIE.lock(%[[VAL_4]], 0)
// CHECK:           %[[VAL_10:.*]] = AIE.buffer(%[[VAL_4]]) {sym_name = "buf45"} : memref<64xi32, 2>
// CHECK:           %[[VAL_11:.*]] = AIE.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_12:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_9]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_10]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_9]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_13:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_7]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_8]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_14:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_5]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_6]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_5]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = AIE.core(%[[VAL_4]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_9]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_7]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_5]], Acquire, 0)
// CHECK:             affine.for %[[VAL_16:.*]] = 0 to 64 {
// CHECK:               %[[VAL_17:.*]] = affine.load %[[VAL_10]]{{\[}}%[[VAL_16]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_18:.*]] = affine.load %[[VAL_8]]{{\[}}%[[VAL_16]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_19:.*]] = arith.muli %[[VAL_17]], %[[VAL_18]] : i32
// CHECK:               affine.store %[[VAL_19]], %[[VAL_6]]{{\[}}%[[VAL_16]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_7]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_9]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = AIE.tile(46, 2)
// CHECK:           %[[VAL_21:.*]] = AIE.tile(46, 1)
// CHECK:           %[[VAL_22:.*]] = AIE.tile(46, 0)
// CHECK:           %[[VAL_23:.*]] = AIE.tile(2, 3)
// CHECK:           %[[VAL_24:.*]] = AIE.tile(9, 5)
// CHECK:           %[[VAL_25:.*]] = AIE.lock(%[[VAL_24]], 2)
// CHECK:           %[[VAL_26:.*]] = AIE.buffer(%[[VAL_24]]) {sym_name = "buf44"} : memref<64xi32, 2>
// CHECK:           %[[VAL_27:.*]] = AIE.lock(%[[VAL_24]], 1)
// CHECK:           %[[VAL_28:.*]] = AIE.buffer(%[[VAL_24]]) {sym_name = "buf43"} : memref<64xi32, 2>
// CHECK:           %[[VAL_29:.*]] = AIE.lock(%[[VAL_24]], 0)
// CHECK:           %[[VAL_30:.*]] = AIE.buffer(%[[VAL_24]]) {sym_name = "buf42"} : memref<64xi32, 2>
// CHECK:           %[[VAL_31:.*]] = AIE.mem(%[[VAL_24]]) {
// CHECK:             %[[VAL_32:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_29]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_30]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_29]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_33:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_27]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_28]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_27]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_34:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_25]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_26]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_25]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = AIE.core(%[[VAL_24]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_29]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_27]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_25]], Acquire, 0)
// CHECK:             affine.for %[[VAL_36:.*]] = 0 to 64 {
// CHECK:               %[[VAL_37:.*]] = affine.load %[[VAL_30]]{{\[}}%[[VAL_36]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_38:.*]] = affine.load %[[VAL_28]]{{\[}}%[[VAL_36]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_39:.*]] = arith.muli %[[VAL_37]], %[[VAL_38]] : i32
// CHECK:               affine.store %[[VAL_39]], %[[VAL_26]]{{\[}}%[[VAL_36]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_25]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_27]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_29]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_40:.*]] = AIE.tile(43, 2)
// CHECK:           %[[VAL_41:.*]] = AIE.tile(43, 1)
// CHECK:           %[[VAL_42:.*]] = AIE.tile(43, 0)
// CHECK:           %[[VAL_43:.*]] = AIE.tile(1, 3)
// CHECK:           %[[VAL_44:.*]] = AIE.tile(8, 5)
// CHECK:           %[[VAL_45:.*]] = AIE.lock(%[[VAL_44]], 2)
// CHECK:           %[[VAL_46:.*]] = AIE.buffer(%[[VAL_44]]) {sym_name = "buf41"} : memref<64xi32, 2>
// CHECK:           %[[VAL_47:.*]] = AIE.lock(%[[VAL_44]], 1)
// CHECK:           %[[VAL_48:.*]] = AIE.buffer(%[[VAL_44]]) {sym_name = "buf40"} : memref<64xi32, 2>
// CHECK:           %[[VAL_49:.*]] = AIE.lock(%[[VAL_44]], 0)
// CHECK:           %[[VAL_50:.*]] = AIE.buffer(%[[VAL_44]]) {sym_name = "buf39"} : memref<64xi32, 2>
// CHECK:           %[[VAL_51:.*]] = AIE.mem(%[[VAL_44]]) {
// CHECK:             %[[VAL_52:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_49]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_50]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_49]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_53:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_47]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_48]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_47]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_54:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_45]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_46]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_45]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_55:.*]] = AIE.core(%[[VAL_44]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_49]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_47]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_45]], Acquire, 0)
// CHECK:             affine.for %[[VAL_56:.*]] = 0 to 64 {
// CHECK:               %[[VAL_57:.*]] = affine.load %[[VAL_50]]{{\[}}%[[VAL_56]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_58:.*]] = affine.load %[[VAL_48]]{{\[}}%[[VAL_56]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_59:.*]] = arith.muli %[[VAL_57]], %[[VAL_58]] : i32
// CHECK:               affine.store %[[VAL_59]], %[[VAL_46]]{{\[}}%[[VAL_56]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_45]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_47]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_49]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_60:.*]] = AIE.tile(42, 2)
// CHECK:           %[[VAL_61:.*]] = AIE.tile(42, 1)
// CHECK:           %[[VAL_62:.*]] = AIE.tile(42, 0)
// CHECK:           %[[VAL_63:.*]] = AIE.tile(0, 3)
// CHECK:           %[[VAL_64:.*]] = AIE.tile(7, 5)
// CHECK:           %[[VAL_65:.*]] = AIE.lock(%[[VAL_64]], 2)
// CHECK:           %[[VAL_66:.*]] = AIE.buffer(%[[VAL_64]]) {sym_name = "buf38"} : memref<64xi32, 2>
// CHECK:           %[[VAL_67:.*]] = AIE.lock(%[[VAL_64]], 1)
// CHECK:           %[[VAL_68:.*]] = AIE.buffer(%[[VAL_64]]) {sym_name = "buf37"} : memref<64xi32, 2>
// CHECK:           %[[VAL_69:.*]] = AIE.lock(%[[VAL_64]], 0)
// CHECK:           %[[VAL_70:.*]] = AIE.buffer(%[[VAL_64]]) {sym_name = "buf36"} : memref<64xi32, 2>
// CHECK:           %[[VAL_71:.*]] = AIE.mem(%[[VAL_64]]) {
// CHECK:             %[[VAL_72:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_69]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_70]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_69]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_73:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_67]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_68]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_67]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_74:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_65]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_66]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_65]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_75:.*]] = AIE.core(%[[VAL_64]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_69]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_67]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_65]], Acquire, 0)
// CHECK:             affine.for %[[VAL_76:.*]] = 0 to 64 {
// CHECK:               %[[VAL_77:.*]] = affine.load %[[VAL_70]]{{\[}}%[[VAL_76]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_78:.*]] = affine.load %[[VAL_68]]{{\[}}%[[VAL_76]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_79:.*]] = arith.muli %[[VAL_77]], %[[VAL_78]] : i32
// CHECK:               affine.store %[[VAL_79]], %[[VAL_66]]{{\[}}%[[VAL_76]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_65]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_67]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_69]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_80:.*]] = AIE.tile(35, 2)
// CHECK:           %[[VAL_81:.*]] = AIE.tile(35, 1)
// CHECK:           %[[VAL_82:.*]] = AIE.tile(35, 0)
// CHECK:           %[[VAL_83:.*]] = AIE.tile(10, 4)
// CHECK:           %[[VAL_84:.*]] = AIE.lock(%[[VAL_83]], 2)
// CHECK:           %[[VAL_85:.*]] = AIE.buffer(%[[VAL_83]]) {sym_name = "buf35"} : memref<64xi32, 2>
// CHECK:           %[[VAL_86:.*]] = AIE.lock(%[[VAL_83]], 1)
// CHECK:           %[[VAL_87:.*]] = AIE.buffer(%[[VAL_83]]) {sym_name = "buf34"} : memref<64xi32, 2>
// CHECK:           %[[VAL_88:.*]] = AIE.lock(%[[VAL_83]], 0)
// CHECK:           %[[VAL_89:.*]] = AIE.buffer(%[[VAL_83]]) {sym_name = "buf33"} : memref<64xi32, 2>
// CHECK:           %[[VAL_90:.*]] = AIE.mem(%[[VAL_83]]) {
// CHECK:             %[[VAL_91:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_88]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_89]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_88]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_92:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_86]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_87]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_86]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_93:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_84]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_85]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_84]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_94:.*]] = AIE.core(%[[VAL_83]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_88]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_86]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_84]], Acquire, 0)
// CHECK:             affine.for %[[VAL_95:.*]] = 0 to 64 {
// CHECK:               %[[VAL_96:.*]] = affine.load %[[VAL_89]]{{\[}}%[[VAL_95]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_97:.*]] = affine.load %[[VAL_87]]{{\[}}%[[VAL_95]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_98:.*]] = arith.muli %[[VAL_96]], %[[VAL_97]] : i32
// CHECK:               affine.store %[[VAL_98]], %[[VAL_85]]{{\[}}%[[VAL_95]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_84]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_86]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_88]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_99:.*]] = AIE.tile(34, 2)
// CHECK:           %[[VAL_100:.*]] = AIE.tile(34, 1)
// CHECK:           %[[VAL_101:.*]] = AIE.tile(34, 0)
// CHECK:           %[[VAL_102:.*]] = AIE.tile(9, 4)
// CHECK:           %[[VAL_103:.*]] = AIE.lock(%[[VAL_102]], 2)
// CHECK:           %[[VAL_104:.*]] = AIE.buffer(%[[VAL_102]]) {sym_name = "buf32"} : memref<64xi32, 2>
// CHECK:           %[[VAL_105:.*]] = AIE.lock(%[[VAL_102]], 1)
// CHECK:           %[[VAL_106:.*]] = AIE.buffer(%[[VAL_102]]) {sym_name = "buf31"} : memref<64xi32, 2>
// CHECK:           %[[VAL_107:.*]] = AIE.lock(%[[VAL_102]], 0)
// CHECK:           %[[VAL_108:.*]] = AIE.buffer(%[[VAL_102]]) {sym_name = "buf30"} : memref<64xi32, 2>
// CHECK:           %[[VAL_109:.*]] = AIE.mem(%[[VAL_102]]) {
// CHECK:             %[[VAL_110:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_107]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_108]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_107]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_111:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_105]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_106]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_105]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_112:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_103]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_104]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_103]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_113:.*]] = AIE.core(%[[VAL_102]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_107]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_105]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_103]], Acquire, 0)
// CHECK:             affine.for %[[VAL_114:.*]] = 0 to 64 {
// CHECK:               %[[VAL_115:.*]] = affine.load %[[VAL_108]]{{\[}}%[[VAL_114]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_116:.*]] = affine.load %[[VAL_106]]{{\[}}%[[VAL_114]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_117:.*]] = arith.muli %[[VAL_115]], %[[VAL_116]] : i32
// CHECK:               affine.store %[[VAL_117]], %[[VAL_104]]{{\[}}%[[VAL_114]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_103]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_105]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_107]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_118:.*]] = AIE.tile(27, 2)
// CHECK:           %[[VAL_119:.*]] = AIE.tile(27, 1)
// CHECK:           %[[VAL_120:.*]] = AIE.tile(27, 0)
// CHECK:           %[[VAL_121:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_122:.*]] = AIE.tile(8, 4)
// CHECK:           %[[VAL_123:.*]] = AIE.lock(%[[VAL_122]], 2)
// CHECK:           %[[VAL_124:.*]] = AIE.buffer(%[[VAL_122]]) {sym_name = "buf29"} : memref<64xi32, 2>
// CHECK:           %[[VAL_125:.*]] = AIE.lock(%[[VAL_122]], 1)
// CHECK:           %[[VAL_126:.*]] = AIE.buffer(%[[VAL_122]]) {sym_name = "buf28"} : memref<64xi32, 2>
// CHECK:           %[[VAL_127:.*]] = AIE.lock(%[[VAL_122]], 0)
// CHECK:           %[[VAL_128:.*]] = AIE.buffer(%[[VAL_122]]) {sym_name = "buf27"} : memref<64xi32, 2>
// CHECK:           %[[VAL_129:.*]] = AIE.mem(%[[VAL_122]]) {
// CHECK:             %[[VAL_130:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_127]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_128]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_127]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_131:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_125]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_126]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_125]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_132:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_123]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_124]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_123]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_133:.*]] = AIE.core(%[[VAL_122]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_127]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_125]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_123]], Acquire, 0)
// CHECK:             affine.for %[[VAL_134:.*]] = 0 to 64 {
// CHECK:               %[[VAL_135:.*]] = affine.load %[[VAL_128]]{{\[}}%[[VAL_134]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_136:.*]] = affine.load %[[VAL_126]]{{\[}}%[[VAL_134]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_137:.*]] = arith.muli %[[VAL_135]], %[[VAL_136]] : i32
// CHECK:               affine.store %[[VAL_137]], %[[VAL_124]]{{\[}}%[[VAL_134]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_123]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_125]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_127]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_138:.*]] = AIE.tile(26, 2)
// CHECK:           %[[VAL_139:.*]] = AIE.tile(26, 1)
// CHECK:           %[[VAL_140:.*]] = AIE.tile(26, 0)
// CHECK:           %[[VAL_141:.*]] = AIE.tile(0, 2)
// CHECK:           %[[VAL_142:.*]] = AIE.tile(7, 4)
// CHECK:           %[[VAL_143:.*]] = AIE.lock(%[[VAL_142]], 2)
// CHECK:           %[[VAL_144:.*]] = AIE.buffer(%[[VAL_142]]) {sym_name = "buf26"} : memref<64xi32, 2>
// CHECK:           %[[VAL_145:.*]] = AIE.lock(%[[VAL_142]], 1)
// CHECK:           %[[VAL_146:.*]] = AIE.buffer(%[[VAL_142]]) {sym_name = "buf25"} : memref<64xi32, 2>
// CHECK:           %[[VAL_147:.*]] = AIE.lock(%[[VAL_142]], 0)
// CHECK:           %[[VAL_148:.*]] = AIE.buffer(%[[VAL_142]]) {sym_name = "buf24"} : memref<64xi32, 2>
// CHECK:           %[[VAL_149:.*]] = AIE.mem(%[[VAL_142]]) {
// CHECK:             %[[VAL_150:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_147]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_148]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_147]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_151:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_145]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_146]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_145]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_152:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_143]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_144]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_143]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_153:.*]] = AIE.core(%[[VAL_142]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_147]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_145]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_143]], Acquire, 0)
// CHECK:             affine.for %[[VAL_154:.*]] = 0 to 64 {
// CHECK:               %[[VAL_155:.*]] = affine.load %[[VAL_148]]{{\[}}%[[VAL_154]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_156:.*]] = affine.load %[[VAL_146]]{{\[}}%[[VAL_154]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_157:.*]] = arith.muli %[[VAL_155]], %[[VAL_156]] : i32
// CHECK:               affine.store %[[VAL_157]], %[[VAL_144]]{{\[}}%[[VAL_154]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_143]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_145]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_147]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_158:.*]] = AIE.tile(19, 2)
// CHECK:           %[[VAL_159:.*]] = AIE.tile(19, 1)
// CHECK:           %[[VAL_160:.*]] = AIE.tile(19, 0)
// CHECK:           %[[VAL_161:.*]] = AIE.tile(10, 3)
// CHECK:           %[[VAL_162:.*]] = AIE.lock(%[[VAL_161]], 2)
// CHECK:           %[[VAL_163:.*]] = AIE.buffer(%[[VAL_161]]) {sym_name = "buf23"} : memref<64xi32, 2>
// CHECK:           %[[VAL_164:.*]] = AIE.lock(%[[VAL_161]], 1)
// CHECK:           %[[VAL_165:.*]] = AIE.buffer(%[[VAL_161]]) {sym_name = "buf22"} : memref<64xi32, 2>
// CHECK:           %[[VAL_166:.*]] = AIE.lock(%[[VAL_161]], 0)
// CHECK:           %[[VAL_167:.*]] = AIE.buffer(%[[VAL_161]]) {sym_name = "buf21"} : memref<64xi32, 2>
// CHECK:           %[[VAL_168:.*]] = AIE.mem(%[[VAL_161]]) {
// CHECK:             %[[VAL_169:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_166]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_167]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_166]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_170:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_164]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_165]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_164]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_171:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_162]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_163]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_162]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_172:.*]] = AIE.core(%[[VAL_161]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_166]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_164]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_162]], Acquire, 0)
// CHECK:             affine.for %[[VAL_173:.*]] = 0 to 64 {
// CHECK:               %[[VAL_174:.*]] = affine.load %[[VAL_167]]{{\[}}%[[VAL_173]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_175:.*]] = affine.load %[[VAL_165]]{{\[}}%[[VAL_173]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_176:.*]] = arith.muli %[[VAL_174]], %[[VAL_175]] : i32
// CHECK:               affine.store %[[VAL_176]], %[[VAL_163]]{{\[}}%[[VAL_173]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_162]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_164]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_166]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_177:.*]] = AIE.tile(18, 2)
// CHECK:           %[[VAL_178:.*]] = AIE.tile(18, 1)
// CHECK:           %[[VAL_179:.*]] = AIE.tile(18, 0)
// CHECK:           %[[VAL_180:.*]] = AIE.tile(9, 3)
// CHECK:           %[[VAL_181:.*]] = AIE.lock(%[[VAL_180]], 2)
// CHECK:           %[[VAL_182:.*]] = AIE.buffer(%[[VAL_180]]) {sym_name = "buf20"} : memref<64xi32, 2>
// CHECK:           %[[VAL_183:.*]] = AIE.lock(%[[VAL_180]], 1)
// CHECK:           %[[VAL_184:.*]] = AIE.buffer(%[[VAL_180]]) {sym_name = "buf19"} : memref<64xi32, 2>
// CHECK:           %[[VAL_185:.*]] = AIE.lock(%[[VAL_180]], 0)
// CHECK:           %[[VAL_186:.*]] = AIE.buffer(%[[VAL_180]]) {sym_name = "buf18"} : memref<64xi32, 2>
// CHECK:           %[[VAL_187:.*]] = AIE.mem(%[[VAL_180]]) {
// CHECK:             %[[VAL_188:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_185]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_186]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_185]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_189:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_183]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_184]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_183]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_190:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_181]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_182]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_181]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_191:.*]] = AIE.core(%[[VAL_180]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_185]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_183]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_181]], Acquire, 0)
// CHECK:             affine.for %[[VAL_192:.*]] = 0 to 64 {
// CHECK:               %[[VAL_193:.*]] = affine.load %[[VAL_186]]{{\[}}%[[VAL_192]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_194:.*]] = affine.load %[[VAL_184]]{{\[}}%[[VAL_192]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_195:.*]] = arith.muli %[[VAL_193]], %[[VAL_194]] : i32
// CHECK:               affine.store %[[VAL_195]], %[[VAL_182]]{{\[}}%[[VAL_192]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_181]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_183]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_185]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_196:.*]] = AIE.tile(11, 2)
// CHECK:           %[[VAL_197:.*]] = AIE.tile(11, 1)
// CHECK:           %[[VAL_198:.*]] = AIE.tile(11, 0)
// CHECK:           %[[VAL_199:.*]] = AIE.tile(1, 1)
// CHECK:           %[[VAL_200:.*]] = AIE.tile(8, 3)
// CHECK:           %[[VAL_201:.*]] = AIE.lock(%[[VAL_200]], 2)
// CHECK:           %[[VAL_202:.*]] = AIE.buffer(%[[VAL_200]]) {sym_name = "buf17"} : memref<64xi32, 2>
// CHECK:           %[[VAL_203:.*]] = AIE.lock(%[[VAL_200]], 1)
// CHECK:           %[[VAL_204:.*]] = AIE.buffer(%[[VAL_200]]) {sym_name = "buf16"} : memref<64xi32, 2>
// CHECK:           %[[VAL_205:.*]] = AIE.lock(%[[VAL_200]], 0)
// CHECK:           %[[VAL_206:.*]] = AIE.buffer(%[[VAL_200]]) {sym_name = "buf15"} : memref<64xi32, 2>
// CHECK:           %[[VAL_207:.*]] = AIE.mem(%[[VAL_200]]) {
// CHECK:             %[[VAL_208:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_205]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_206]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_205]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_209:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_203]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_204]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_203]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_210:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_201]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_202]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_201]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_211:.*]] = AIE.core(%[[VAL_200]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_205]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_203]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_201]], Acquire, 0)
// CHECK:             affine.for %[[VAL_212:.*]] = 0 to 64 {
// CHECK:               %[[VAL_213:.*]] = affine.load %[[VAL_206]]{{\[}}%[[VAL_212]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_214:.*]] = affine.load %[[VAL_204]]{{\[}}%[[VAL_212]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_215:.*]] = arith.muli %[[VAL_213]], %[[VAL_214]] : i32
// CHECK:               affine.store %[[VAL_215]], %[[VAL_202]]{{\[}}%[[VAL_212]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_201]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_203]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_205]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_216:.*]] = AIE.tile(10, 1)
// CHECK:           %[[VAL_217:.*]] = AIE.tile(10, 0)
// CHECK:           %[[VAL_218:.*]] = AIE.tile(0, 1)
// CHECK:           %[[VAL_219:.*]] = AIE.tile(7, 3)
// CHECK:           %[[VAL_220:.*]] = AIE.lock(%[[VAL_219]], 2)
// CHECK:           %[[VAL_221:.*]] = AIE.buffer(%[[VAL_219]]) {sym_name = "buf14"} : memref<64xi32, 2>
// CHECK:           %[[VAL_222:.*]] = AIE.lock(%[[VAL_219]], 1)
// CHECK:           %[[VAL_223:.*]] = AIE.buffer(%[[VAL_219]]) {sym_name = "buf13"} : memref<64xi32, 2>
// CHECK:           %[[VAL_224:.*]] = AIE.lock(%[[VAL_219]], 0)
// CHECK:           %[[VAL_225:.*]] = AIE.buffer(%[[VAL_219]]) {sym_name = "buf12"} : memref<64xi32, 2>
// CHECK:           %[[VAL_226:.*]] = AIE.mem(%[[VAL_219]]) {
// CHECK:             %[[VAL_227:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_224]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_225]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_224]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_228:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_222]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_223]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_222]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_229:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_220]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_221]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_220]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_230:.*]] = AIE.core(%[[VAL_219]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_224]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_222]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_220]], Acquire, 0)
// CHECK:             affine.for %[[VAL_231:.*]] = 0 to 64 {
// CHECK:               %[[VAL_232:.*]] = affine.load %[[VAL_225]]{{\[}}%[[VAL_231]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_233:.*]] = affine.load %[[VAL_223]]{{\[}}%[[VAL_231]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_234:.*]] = arith.muli %[[VAL_232]], %[[VAL_233]] : i32
// CHECK:               affine.store %[[VAL_234]], %[[VAL_221]]{{\[}}%[[VAL_231]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_220]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_222]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_224]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_235:.*]] = AIE.tile(7, 1)
// CHECK:           %[[VAL_236:.*]] = AIE.tile(7, 0)
// CHECK:           %[[VAL_237:.*]] = AIE.tile(10, 2)
// CHECK:           %[[VAL_238:.*]] = AIE.lock(%[[VAL_237]], 2)
// CHECK:           %[[VAL_239:.*]] = AIE.buffer(%[[VAL_237]]) {sym_name = "buf11"} : memref<64xi32, 2>
// CHECK:           %[[VAL_240:.*]] = AIE.lock(%[[VAL_237]], 1)
// CHECK:           %[[VAL_241:.*]] = AIE.buffer(%[[VAL_237]]) {sym_name = "buf10"} : memref<64xi32, 2>
// CHECK:           %[[VAL_242:.*]] = AIE.lock(%[[VAL_237]], 0)
// CHECK:           %[[VAL_243:.*]] = AIE.buffer(%[[VAL_237]]) {sym_name = "buf9"} : memref<64xi32, 2>
// CHECK:           %[[VAL_244:.*]] = AIE.mem(%[[VAL_237]]) {
// CHECK:             %[[VAL_245:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_242]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_243]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_242]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_246:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_240]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_241]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_240]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_247:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_238]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_239]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_238]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_248:.*]] = AIE.core(%[[VAL_237]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_242]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_240]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_238]], Acquire, 0)
// CHECK:             affine.for %[[VAL_249:.*]] = 0 to 64 {
// CHECK:               %[[VAL_250:.*]] = affine.load %[[VAL_243]]{{\[}}%[[VAL_249]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_251:.*]] = affine.load %[[VAL_241]]{{\[}}%[[VAL_249]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_252:.*]] = arith.muli %[[VAL_250]], %[[VAL_251]] : i32
// CHECK:               affine.store %[[VAL_252]], %[[VAL_239]]{{\[}}%[[VAL_249]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_238]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_240]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_242]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_253:.*]] = AIE.tile(6, 2)
// CHECK:           %[[VAL_254:.*]] = AIE.tile(6, 1)
// CHECK:           %[[VAL_255:.*]] = AIE.tile(6, 0)
// CHECK:           %[[VAL_256:.*]] = AIE.tile(9, 2)
// CHECK:           %[[VAL_257:.*]] = AIE.lock(%[[VAL_256]], 2)
// CHECK:           %[[VAL_258:.*]] = AIE.buffer(%[[VAL_256]]) {sym_name = "buf8"} : memref<64xi32, 2>
// CHECK:           %[[VAL_259:.*]] = AIE.lock(%[[VAL_256]], 1)
// CHECK:           %[[VAL_260:.*]] = AIE.buffer(%[[VAL_256]]) {sym_name = "buf7"} : memref<64xi32, 2>
// CHECK:           %[[VAL_261:.*]] = AIE.lock(%[[VAL_256]], 0)
// CHECK:           %[[VAL_262:.*]] = AIE.buffer(%[[VAL_256]]) {sym_name = "buf6"} : memref<64xi32, 2>
// CHECK:           %[[VAL_263:.*]] = AIE.mem(%[[VAL_256]]) {
// CHECK:             %[[VAL_264:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_261]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_262]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_261]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_265:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_259]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_260]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_259]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_266:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_257]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_258]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_257]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_267:.*]] = AIE.core(%[[VAL_256]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_261]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_259]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_257]], Acquire, 0)
// CHECK:             affine.for %[[VAL_268:.*]] = 0 to 64 {
// CHECK:               %[[VAL_269:.*]] = affine.load %[[VAL_262]]{{\[}}%[[VAL_268]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_270:.*]] = affine.load %[[VAL_260]]{{\[}}%[[VAL_268]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_271:.*]] = arith.muli %[[VAL_269]], %[[VAL_270]] : i32
// CHECK:               affine.store %[[VAL_271]], %[[VAL_258]]{{\[}}%[[VAL_268]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_257]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_259]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_261]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_272:.*]] = AIE.tile(3, 2)
// CHECK:           %[[VAL_273:.*]] = AIE.tile(3, 1)
// CHECK:           %[[VAL_274:.*]] = AIE.tile(3, 0)
// CHECK:           %[[VAL_275:.*]] = AIE.tile(1, 0)
// CHECK:           %[[VAL_276:.*]] = AIE.tile(8, 2)
// CHECK:           %[[VAL_277:.*]] = AIE.lock(%[[VAL_276]], 2)
// CHECK:           %[[VAL_278:.*]] = AIE.buffer(%[[VAL_276]]) {sym_name = "buf5"} : memref<64xi32, 2>
// CHECK:           %[[VAL_279:.*]] = AIE.lock(%[[VAL_276]], 1)
// CHECK:           %[[VAL_280:.*]] = AIE.buffer(%[[VAL_276]]) {sym_name = "buf4"} : memref<64xi32, 2>
// CHECK:           %[[VAL_281:.*]] = AIE.lock(%[[VAL_276]], 0)
// CHECK:           %[[VAL_282:.*]] = AIE.buffer(%[[VAL_276]]) {sym_name = "buf3"} : memref<64xi32, 2>
// CHECK:           %[[VAL_283:.*]] = AIE.mem(%[[VAL_276]]) {
// CHECK:             %[[VAL_284:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_281]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_282]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_281]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_285:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_279]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_280]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_279]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_286:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_277]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_278]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_277]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_287:.*]] = AIE.core(%[[VAL_276]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_281]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_279]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_277]], Acquire, 0)
// CHECK:             affine.for %[[VAL_288:.*]] = 0 to 64 {
// CHECK:               %[[VAL_289:.*]] = affine.load %[[VAL_282]]{{\[}}%[[VAL_288]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_290:.*]] = affine.load %[[VAL_280]]{{\[}}%[[VAL_288]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_291:.*]] = arith.muli %[[VAL_289]], %[[VAL_290]] : i32
// CHECK:               affine.store %[[VAL_291]], %[[VAL_278]]{{\[}}%[[VAL_288]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_277]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_279]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_281]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_292:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_293:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_294:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_295:.*]] = AIE.tile(0, 0)
// CHECK:           %[[VAL_296:.*]] = AIE.tile(7, 2)
// CHECK:           %[[VAL_297:.*]] = AIE.lock(%[[VAL_296]], 2)
// CHECK:           %[[VAL_298:.*]] = AIE.buffer(%[[VAL_296]]) {sym_name = "buf2"} : memref<64xi32, 2>
// CHECK:           %[[VAL_299:.*]] = AIE.lock(%[[VAL_296]], 1)
// CHECK:           %[[VAL_300:.*]] = AIE.buffer(%[[VAL_296]]) {sym_name = "buf1"} : memref<64xi32, 2>
// CHECK:           %[[VAL_301:.*]] = AIE.lock(%[[VAL_296]], 0)
// CHECK:           %[[VAL_302:.*]] = AIE.buffer(%[[VAL_296]]) {sym_name = "buf0"} : memref<64xi32, 2>
// CHECK:           %[[VAL_303:.*]] = AIE.mem(%[[VAL_296]]) {
// CHECK:             %[[VAL_304:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_301]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_302]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_301]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_305:.*]] = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             AIE.useLock(%[[VAL_299]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_300]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_299]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_306:.*]] = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             AIE.useLock(%[[VAL_297]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_298]] : memref<64xi32, 2>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_297]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb6:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_307:.*]] = AIE.core(%[[VAL_296]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_301]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_299]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_297]], Acquire, 0)
// CHECK:             affine.for %[[VAL_308:.*]] = 0 to 64 {
// CHECK:               %[[VAL_309:.*]] = affine.load %[[VAL_302]]{{\[}}%[[VAL_308]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_310:.*]] = affine.load %[[VAL_300]]{{\[}}%[[VAL_308]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_311:.*]] = arith.muli %[[VAL_309]], %[[VAL_310]] : i32
// CHECK:               affine.store %[[VAL_311]], %[[VAL_298]]{{\[}}%[[VAL_308]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             AIE.useLock(%[[VAL_297]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_299]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_301]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_312:.*]] = AIE.switchbox(%[[VAL_294]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:             AIE.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_313:.*]] = AIE.shimmux(%[[VAL_294]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_314:.*]] = AIE.switchbox(%[[VAL_293]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_315:.*]] = AIE.switchbox(%[[VAL_292]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, East : 1>
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_316:.*]] = AIE.switchbox(%[[VAL_272]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:             AIE.connect<East : 1, West : 0>
// CHECK:             AIE.connect<East : 2, South : 1>
// CHECK:             AIE.connect<East : 3, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_317:.*]] = AIE.tile(4, 2)
// CHECK:           %[[VAL_318:.*]] = AIE.switchbox(%[[VAL_317]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_319:.*]] = AIE.tile(5, 2)
// CHECK:           %[[VAL_320:.*]] = AIE.switchbox(%[[VAL_319]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_321:.*]] = AIE.switchbox(%[[VAL_253]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<South : 0, East : 2>
// CHECK:             AIE.connect<South : 1, East : 3>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_322:.*]] = AIE.switchbox(%[[VAL_296]]) {
// CHECK:             AIE.connect<West : 0, DMA : 0>
// CHECK:             AIE.connect<West : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, West : 0>
// CHECK:             AIE.connect<West : 2, East : 0>
// CHECK:             AIE.connect<West : 3, East : 1>
// CHECK:             AIE.connect<East : 0, West : 1>
// CHECK:             AIE.connect<East : 1, West : 2>
// CHECK:             AIE.connect<East : 2, West : 3>
// CHECK:             AIE.connect<East : 3, North : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<North : 1, South : 1>
// CHECK:             AIE.connect<North : 2, South : 2>
// CHECK:             AIE.connect<North : 3, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_323:.*]] = AIE.switchbox(%[[VAL_273]]) {
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<North : 1, South : 0>
// CHECK:             AIE.connect<North : 2, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_324:.*]] = AIE.switchbox(%[[VAL_274]]) {
// CHECK:             AIE.connect<South : 3, East : 0>
// CHECK:             AIE.connect<South : 7, East : 1>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:             AIE.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_325:.*]] = AIE.shimmux(%[[VAL_274]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_326:.*]] = AIE.tile(4, 0)
// CHECK:           %[[VAL_327:.*]] = AIE.switchbox(%[[VAL_326]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_328:.*]] = AIE.tile(5, 0)
// CHECK:           %[[VAL_329:.*]] = AIE.switchbox(%[[VAL_328]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_330:.*]] = AIE.switchbox(%[[VAL_255]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<West : 1, North : 1>
// CHECK:             AIE.connect<South : 3, East : 0>
// CHECK:             AIE.connect<South : 7, East : 1>
// CHECK:             AIE.connect<East : 0, South : 2>
// CHECK:             AIE.connect<East : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_331:.*]] = AIE.switchbox(%[[VAL_254]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_332:.*]] = AIE.switchbox(%[[VAL_276]]) {
// CHECK:             AIE.connect<West : 0, DMA : 0>
// CHECK:             AIE.connect<West : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, West : 0>
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, East : 1>
// CHECK:             AIE.connect<East : 0, West : 1>
// CHECK:             AIE.connect<East : 1, West : 2>
// CHECK:             AIE.connect<South : 2, West : 3>
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 4, North : 1>
// CHECK:             AIE.connect<South : 5, North : 2>
// CHECK:             AIE.connect<East : 2, North : 3>
// CHECK:             AIE.connect<East : 3, North : 4>
// CHECK:             AIE.connect<North : 0, East : 2>
// CHECK:             AIE.connect<North : 1, East : 3>
// CHECK:           }
// CHECK:           %[[VAL_333:.*]] = AIE.shimmux(%[[VAL_255]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_334:.*]] = AIE.switchbox(%[[VAL_236]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<South : 3, East : 2>
// CHECK:             AIE.connect<South : 7, East : 3>
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<North : 1, West : 1>
// CHECK:             AIE.connect<North : 2, South : 2>
// CHECK:             AIE.connect<North : 3, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_335:.*]] = AIE.tile(8, 0)
// CHECK:           %[[VAL_336:.*]] = AIE.switchbox(%[[VAL_335]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<West : 1, North : 1>
// CHECK:             AIE.connect<West : 2, East : 0>
// CHECK:             AIE.connect<West : 3, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_337:.*]] = AIE.tile(8, 1)
// CHECK:           %[[VAL_338:.*]] = AIE.switchbox(%[[VAL_337]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:             AIE.connect<East : 0, North : 2>
// CHECK:             AIE.connect<East : 1, North : 3>
// CHECK:             AIE.connect<East : 2, North : 4>
// CHECK:             AIE.connect<East : 3, North : 5>
// CHECK:           }
// CHECK:           %[[VAL_339:.*]] = AIE.switchbox(%[[VAL_256]]) {
// CHECK:             AIE.connect<West : 0, DMA : 0>
// CHECK:             AIE.connect<West : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, West : 0>
// CHECK:             AIE.connect<East : 0, West : 1>
// CHECK:             AIE.connect<East : 1, West : 2>
// CHECK:             AIE.connect<East : 2, West : 3>
// CHECK:             AIE.connect<West : 2, South : 0>
// CHECK:             AIE.connect<East : 3, North : 0>
// CHECK:             AIE.connect<West : 3, South : 1>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_340:.*]] = AIE.shimmux(%[[VAL_236]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_341:.*]] = AIE.tile(9, 0)
// CHECK:           %[[VAL_342:.*]] = AIE.switchbox(%[[VAL_341]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<West : 1, North : 1>
// CHECK:             AIE.connect<East : 0, North : 2>
// CHECK:             AIE.connect<East : 1, North : 3>
// CHECK:             AIE.connect<East : 2, North : 4>
// CHECK:             AIE.connect<East : 3, North : 5>
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<North : 1, East : 1>
// CHECK:             AIE.connect<North : 2, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_343:.*]] = AIE.tile(9, 1)
// CHECK:           %[[VAL_344:.*]] = AIE.switchbox(%[[VAL_343]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, East : 1>
// CHECK:             AIE.connect<South : 2, West : 0>
// CHECK:             AIE.connect<South : 3, West : 1>
// CHECK:             AIE.connect<South : 4, West : 2>
// CHECK:             AIE.connect<South : 5, West : 3>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<North : 1, South : 1>
// CHECK:             AIE.connect<North : 2, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_345:.*]] = AIE.switchbox(%[[VAL_216]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<West : 1, North : 1>
// CHECK:             AIE.connect<East : 0, North : 2>
// CHECK:             AIE.connect<East : 1, North : 3>
// CHECK:             AIE.connect<East : 2, North : 4>
// CHECK:             AIE.connect<East : 3, North : 5>
// CHECK:           }
// CHECK:           %[[VAL_346:.*]] = AIE.switchbox(%[[VAL_237]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<South : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, West : 0>
// CHECK:             AIE.connect<South : 2, North : 0>
// CHECK:             AIE.connect<South : 3, North : 1>
// CHECK:             AIE.connect<South : 4, North : 2>
// CHECK:             AIE.connect<South : 5, North : 3>
// CHECK:             AIE.connect<East : 0, West : 1>
// CHECK:             AIE.connect<East : 1, West : 2>
// CHECK:             AIE.connect<East : 2, West : 3>
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<East : 3, North : 4>
// CHECK:           }
// CHECK:           %[[VAL_347:.*]] = AIE.switchbox(%[[VAL_217]]) {
// CHECK:             AIE.connect<South : 3, West : 0>
// CHECK:             AIE.connect<South : 7, West : 1>
// CHECK:             AIE.connect<East : 0, West : 2>
// CHECK:             AIE.connect<East : 1, West : 3>
// CHECK:             AIE.connect<West : 0, South : 2>
// CHECK:             AIE.connect<West : 1, South : 3>
// CHECK:             AIE.connect<West : 2, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_348:.*]] = AIE.shimmux(%[[VAL_217]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_349:.*]] = AIE.switchbox(%[[VAL_219]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<East : 0, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<East : 1, South : 1>
// CHECK:             AIE.connect<East : 2, South : 2>
// CHECK:             AIE.connect<East : 3, South : 3>
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_350:.*]] = AIE.switchbox(%[[VAL_200]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, DMA : 0>
// CHECK:             AIE.connect<South : 2, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, West : 1>
// CHECK:             AIE.connect<East : 0, West : 2>
// CHECK:             AIE.connect<East : 1, West : 3>
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 4, North : 1>
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<East : 2, North : 2>
// CHECK:             AIE.connect<East : 3, North : 3>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_351:.*]] = AIE.switchbox(%[[VAL_235]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<North : 1, South : 1>
// CHECK:             AIE.connect<North : 2, South : 2>
// CHECK:             AIE.connect<North : 3, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_352:.*]] = AIE.switchbox(%[[VAL_198]]) {
// CHECK:             AIE.connect<South : 3, West : 0>
// CHECK:             AIE.connect<South : 7, West : 1>
// CHECK:             AIE.connect<West : 0, South : 2>
// CHECK:             AIE.connect<North : 0, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_353:.*]] = AIE.shimmux(%[[VAL_198]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_354:.*]] = AIE.switchbox(%[[VAL_179]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:             AIE.connect<East : 0, North : 2>
// CHECK:             AIE.connect<East : 1, North : 3>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:             AIE.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_355:.*]] = AIE.shimmux(%[[VAL_179]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_356:.*]] = AIE.switchbox(%[[VAL_197]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_357:.*]] = AIE.tile(12, 1)
// CHECK:           %[[VAL_358:.*]] = AIE.switchbox(%[[VAL_357]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_359:.*]] = AIE.tile(13, 1)
// CHECK:           %[[VAL_360:.*]] = AIE.switchbox(%[[VAL_359]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_361:.*]] = AIE.tile(14, 1)
// CHECK:           %[[VAL_362:.*]] = AIE.switchbox(%[[VAL_361]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_363:.*]] = AIE.tile(15, 1)
// CHECK:           %[[VAL_364:.*]] = AIE.switchbox(%[[VAL_363]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_365:.*]] = AIE.tile(16, 1)
// CHECK:           %[[VAL_366:.*]] = AIE.switchbox(%[[VAL_365]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_367:.*]] = AIE.tile(17, 1)
// CHECK:           %[[VAL_368:.*]] = AIE.switchbox(%[[VAL_367]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_369:.*]] = AIE.switchbox(%[[VAL_178]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:             AIE.connect<South : 2, West : 2>
// CHECK:             AIE.connect<South : 3, West : 3>
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:             AIE.connect<North : 1, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_370:.*]] = AIE.switchbox(%[[VAL_180]]) {
// CHECK:             AIE.connect<East : 0, DMA : 0>
// CHECK:             AIE.connect<East : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, West : 0>
// CHECK:             AIE.connect<East : 2, West : 1>
// CHECK:             AIE.connect<East : 3, West : 2>
// CHECK:             AIE.connect<South : 0, West : 3>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_371:.*]] = AIE.switchbox(%[[VAL_161]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:             AIE.connect<South : 2, DMA : 0>
// CHECK:             AIE.connect<South : 3, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, West : 2>
// CHECK:             AIE.connect<East : 0, West : 3>
// CHECK:             AIE.connect<East : 1, North : 0>
// CHECK:             AIE.connect<East : 2, North : 1>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<East : 3, North : 2>
// CHECK:             AIE.connect<South : 4, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_372:.*]] = AIE.switchbox(%[[VAL_160]]) {
// CHECK:             AIE.connect<South : 3, West : 0>
// CHECK:             AIE.connect<South : 7, West : 1>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:             AIE.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_373:.*]] = AIE.shimmux(%[[VAL_160]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_374:.*]] = AIE.switchbox(%[[VAL_140]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_375:.*]] = AIE.shimmux(%[[VAL_140]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_376:.*]] = AIE.switchbox(%[[VAL_139]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_377:.*]] = AIE.switchbox(%[[VAL_196]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_378:.*]] = AIE.tile(12, 2)
// CHECK:           %[[VAL_379:.*]] = AIE.switchbox(%[[VAL_378]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_380:.*]] = AIE.tile(13, 2)
// CHECK:           %[[VAL_381:.*]] = AIE.switchbox(%[[VAL_380]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_382:.*]] = AIE.tile(14, 2)
// CHECK:           %[[VAL_383:.*]] = AIE.switchbox(%[[VAL_382]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_384:.*]] = AIE.tile(15, 2)
// CHECK:           %[[VAL_385:.*]] = AIE.switchbox(%[[VAL_384]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_386:.*]] = AIE.tile(16, 2)
// CHECK:           %[[VAL_387:.*]] = AIE.switchbox(%[[VAL_386]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_388:.*]] = AIE.tile(17, 2)
// CHECK:           %[[VAL_389:.*]] = AIE.switchbox(%[[VAL_388]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<North : 1, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_390:.*]] = AIE.switchbox(%[[VAL_177]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<North : 1, East : 0>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<West : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_391:.*]] = AIE.switchbox(%[[VAL_158]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<South : 2, North : 2>
// CHECK:             AIE.connect<South : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_392:.*]] = AIE.tile(20, 2)
// CHECK:           %[[VAL_393:.*]] = AIE.switchbox(%[[VAL_392]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, North : 0>
// CHECK:             AIE.connect<East : 3, West : 2>
// CHECK:           }
// CHECK:           %[[VAL_394:.*]] = AIE.tile(21, 2)
// CHECK:           %[[VAL_395:.*]] = AIE.switchbox(%[[VAL_394]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_396:.*]] = AIE.tile(22, 2)
// CHECK:           %[[VAL_397:.*]] = AIE.switchbox(%[[VAL_396]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_398:.*]] = AIE.tile(23, 2)
// CHECK:           %[[VAL_399:.*]] = AIE.switchbox(%[[VAL_398]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_400:.*]] = AIE.tile(24, 2)
// CHECK:           %[[VAL_401:.*]] = AIE.switchbox(%[[VAL_400]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_402:.*]] = AIE.tile(25, 2)
// CHECK:           %[[VAL_403:.*]] = AIE.switchbox(%[[VAL_402]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_404:.*]] = AIE.switchbox(%[[VAL_138]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:             AIE.connect<East : 0, West : 2>
// CHECK:             AIE.connect<East : 1, West : 3>
// CHECK:             AIE.connect<East : 2, North : 0>
// CHECK:             AIE.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_405:.*]] = AIE.switchbox(%[[VAL_142]]) {
// CHECK:             AIE.connect<East : 0, DMA : 0>
// CHECK:             AIE.connect<East : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_406:.*]] = AIE.switchbox(%[[VAL_122]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:             AIE.connect<South : 2, DMA : 0>
// CHECK:             AIE.connect<South : 3, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_407:.*]] = AIE.switchbox(%[[VAL_120]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_408:.*]] = AIE.shimmux(%[[VAL_120]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_409:.*]] = AIE.switchbox(%[[VAL_119]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_410:.*]] = AIE.switchbox(%[[VAL_118]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:             AIE.connect<East : 0, West : 2>
// CHECK:             AIE.connect<East : 1, West : 3>
// CHECK:             AIE.connect<East : 2, North : 0>
// CHECK:             AIE.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_411:.*]] = AIE.tile(11, 3)
// CHECK:           %[[VAL_412:.*]] = AIE.switchbox(%[[VAL_411]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_413:.*]] = AIE.tile(12, 3)
// CHECK:           %[[VAL_414:.*]] = AIE.switchbox(%[[VAL_413]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_415:.*]] = AIE.tile(13, 3)
// CHECK:           %[[VAL_416:.*]] = AIE.switchbox(%[[VAL_415]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_417:.*]] = AIE.tile(14, 3)
// CHECK:           %[[VAL_418:.*]] = AIE.switchbox(%[[VAL_417]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_419:.*]] = AIE.tile(15, 3)
// CHECK:           %[[VAL_420:.*]] = AIE.switchbox(%[[VAL_419]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_421:.*]] = AIE.tile(16, 3)
// CHECK:           %[[VAL_422:.*]] = AIE.switchbox(%[[VAL_421]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<North : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_423:.*]] = AIE.tile(17, 3)
// CHECK:           %[[VAL_424:.*]] = AIE.switchbox(%[[VAL_423]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<West : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_425:.*]] = AIE.tile(18, 3)
// CHECK:           %[[VAL_426:.*]] = AIE.switchbox(%[[VAL_425]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<North : 1, South : 1>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_427:.*]] = AIE.tile(19, 3)
// CHECK:           %[[VAL_428:.*]] = AIE.switchbox(%[[VAL_427]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<South : 0, West : 1>
// CHECK:             AIE.connect<South : 1, West : 2>
// CHECK:             AIE.connect<South : 2, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_429:.*]] = AIE.tile(20, 3)
// CHECK:           %[[VAL_430:.*]] = AIE.switchbox(%[[VAL_429]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_431:.*]] = AIE.switchbox(%[[VAL_101]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_432:.*]] = AIE.shimmux(%[[VAL_101]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_433:.*]] = AIE.switchbox(%[[VAL_100]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_434:.*]] = AIE.tile(28, 2)
// CHECK:           %[[VAL_435:.*]] = AIE.switchbox(%[[VAL_434]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_436:.*]] = AIE.tile(29, 2)
// CHECK:           %[[VAL_437:.*]] = AIE.switchbox(%[[VAL_436]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_438:.*]] = AIE.tile(30, 2)
// CHECK:           %[[VAL_439:.*]] = AIE.switchbox(%[[VAL_438]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_440:.*]] = AIE.tile(31, 2)
// CHECK:           %[[VAL_441:.*]] = AIE.switchbox(%[[VAL_440]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_442:.*]] = AIE.tile(32, 2)
// CHECK:           %[[VAL_443:.*]] = AIE.switchbox(%[[VAL_442]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_444:.*]] = AIE.tile(33, 2)
// CHECK:           %[[VAL_445:.*]] = AIE.switchbox(%[[VAL_444]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_446:.*]] = AIE.switchbox(%[[VAL_99]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:             AIE.connect<East : 0, West : 2>
// CHECK:             AIE.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_447:.*]] = AIE.tile(26, 3)
// CHECK:           %[[VAL_448:.*]] = AIE.switchbox(%[[VAL_447]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_449:.*]] = AIE.switchbox(%[[VAL_102]]) {
// CHECK:             AIE.connect<East : 0, DMA : 0>
// CHECK:             AIE.connect<East : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<East : 2, North : 0>
// CHECK:             AIE.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_450:.*]] = AIE.switchbox(%[[VAL_83]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<South : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<South : 2, North : 0>
// CHECK:             AIE.connect<South : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_451:.*]] = AIE.tile(11, 4)
// CHECK:           %[[VAL_452:.*]] = AIE.switchbox(%[[VAL_451]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_453:.*]] = AIE.tile(12, 4)
// CHECK:           %[[VAL_454:.*]] = AIE.switchbox(%[[VAL_453]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_455:.*]] = AIE.tile(13, 4)
// CHECK:           %[[VAL_456:.*]] = AIE.switchbox(%[[VAL_455]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_457:.*]] = AIE.tile(14, 4)
// CHECK:           %[[VAL_458:.*]] = AIE.switchbox(%[[VAL_457]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_459:.*]] = AIE.tile(15, 4)
// CHECK:           %[[VAL_460:.*]] = AIE.switchbox(%[[VAL_459]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_461:.*]] = AIE.tile(16, 4)
// CHECK:           %[[VAL_462:.*]] = AIE.switchbox(%[[VAL_461]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_463:.*]] = AIE.tile(17, 4)
// CHECK:           %[[VAL_464:.*]] = AIE.switchbox(%[[VAL_463]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_465:.*]] = AIE.tile(18, 4)
// CHECK:           %[[VAL_466:.*]] = AIE.switchbox(%[[VAL_465]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_467:.*]] = AIE.tile(19, 4)
// CHECK:           %[[VAL_468:.*]] = AIE.switchbox(%[[VAL_467]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_469:.*]] = AIE.tile(20, 4)
// CHECK:           %[[VAL_470:.*]] = AIE.switchbox(%[[VAL_469]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_471:.*]] = AIE.tile(21, 4)
// CHECK:           %[[VAL_472:.*]] = AIE.switchbox(%[[VAL_471]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_473:.*]] = AIE.tile(22, 4)
// CHECK:           %[[VAL_474:.*]] = AIE.switchbox(%[[VAL_473]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_475:.*]] = AIE.tile(23, 4)
// CHECK:           %[[VAL_476:.*]] = AIE.switchbox(%[[VAL_475]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_477:.*]] = AIE.tile(24, 4)
// CHECK:           %[[VAL_478:.*]] = AIE.switchbox(%[[VAL_477]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_479:.*]] = AIE.tile(25, 4)
// CHECK:           %[[VAL_480:.*]] = AIE.switchbox(%[[VAL_479]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_481:.*]] = AIE.tile(26, 4)
// CHECK:           %[[VAL_482:.*]] = AIE.switchbox(%[[VAL_481]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:             AIE.connect<East : 0, West : 2>
// CHECK:             AIE.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_483:.*]] = AIE.switchbox(%[[VAL_82]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_484:.*]] = AIE.shimmux(%[[VAL_82]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_485:.*]] = AIE.switchbox(%[[VAL_159]]) {
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:             AIE.connect<East : 1, North : 1>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<East : 2, North : 2>
// CHECK:             AIE.connect<East : 3, North : 3>
// CHECK:             AIE.connect<West : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_486:.*]] = AIE.tile(20, 1)
// CHECK:           %[[VAL_487:.*]] = AIE.switchbox(%[[VAL_486]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_488:.*]] = AIE.tile(21, 1)
// CHECK:           %[[VAL_489:.*]] = AIE.switchbox(%[[VAL_488]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_490:.*]] = AIE.tile(22, 1)
// CHECK:           %[[VAL_491:.*]] = AIE.switchbox(%[[VAL_490]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_492:.*]] = AIE.tile(23, 1)
// CHECK:           %[[VAL_493:.*]] = AIE.switchbox(%[[VAL_492]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_494:.*]] = AIE.tile(24, 1)
// CHECK:           %[[VAL_495:.*]] = AIE.switchbox(%[[VAL_494]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_496:.*]] = AIE.tile(25, 1)
// CHECK:           %[[VAL_497:.*]] = AIE.switchbox(%[[VAL_496]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_498:.*]] = AIE.tile(28, 1)
// CHECK:           %[[VAL_499:.*]] = AIE.switchbox(%[[VAL_498]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_500:.*]] = AIE.tile(29, 1)
// CHECK:           %[[VAL_501:.*]] = AIE.switchbox(%[[VAL_500]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_502:.*]] = AIE.tile(30, 1)
// CHECK:           %[[VAL_503:.*]] = AIE.switchbox(%[[VAL_502]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_504:.*]] = AIE.tile(31, 1)
// CHECK:           %[[VAL_505:.*]] = AIE.switchbox(%[[VAL_504]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_506:.*]] = AIE.tile(32, 1)
// CHECK:           %[[VAL_507:.*]] = AIE.switchbox(%[[VAL_506]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_508:.*]] = AIE.tile(33, 1)
// CHECK:           %[[VAL_509:.*]] = AIE.switchbox(%[[VAL_508]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_510:.*]] = AIE.switchbox(%[[VAL_81]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:             AIE.connect<East : 0, West : 2>
// CHECK:             AIE.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_511:.*]] = AIE.switchbox(%[[VAL_62]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_512:.*]] = AIE.shimmux(%[[VAL_62]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_513:.*]] = AIE.switchbox(%[[VAL_61]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_514:.*]] = AIE.tile(36, 2)
// CHECK:           %[[VAL_515:.*]] = AIE.switchbox(%[[VAL_514]]) {
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:             AIE.connect<East : 1, North : 1>
// CHECK:             AIE.connect<East : 2, West : 0>
// CHECK:             AIE.connect<East : 3, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_516:.*]] = AIE.tile(37, 2)
// CHECK:           %[[VAL_517:.*]] = AIE.switchbox(%[[VAL_516]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_518:.*]] = AIE.tile(38, 2)
// CHECK:           %[[VAL_519:.*]] = AIE.switchbox(%[[VAL_518]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_520:.*]] = AIE.tile(39, 2)
// CHECK:           %[[VAL_521:.*]] = AIE.switchbox(%[[VAL_520]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_522:.*]] = AIE.tile(40, 2)
// CHECK:           %[[VAL_523:.*]] = AIE.switchbox(%[[VAL_522]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_524:.*]] = AIE.tile(41, 2)
// CHECK:           %[[VAL_525:.*]] = AIE.switchbox(%[[VAL_524]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_526:.*]] = AIE.switchbox(%[[VAL_60]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:             AIE.connect<East : 0, West : 2>
// CHECK:             AIE.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_527:.*]] = AIE.tile(36, 3)
// CHECK:           %[[VAL_528:.*]] = AIE.switchbox(%[[VAL_527]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_529:.*]] = AIE.tile(36, 4)
// CHECK:           %[[VAL_530:.*]] = AIE.switchbox(%[[VAL_529]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_531:.*]] = AIE.switchbox(%[[VAL_64]]) {
// CHECK:             AIE.connect<East : 0, DMA : 0>
// CHECK:             AIE.connect<East : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_532:.*]] = AIE.switchbox(%[[VAL_44]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 2, DMA : 0>
// CHECK:             AIE.connect<East : 3, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_533:.*]] = AIE.switchbox(%[[VAL_24]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<South : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_534:.*]] = AIE.switchbox(%[[VAL_4]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<West : 2, East : 2>
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<South : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, East : 3>
// CHECK:           }
// CHECK:           %[[VAL_535:.*]] = AIE.tile(11, 5)
// CHECK:           %[[VAL_536:.*]] = AIE.switchbox(%[[VAL_535]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<West : 2, East : 2>
// CHECK:             AIE.connect<West : 3, East : 3>
// CHECK:           }
// CHECK:           %[[VAL_537:.*]] = AIE.tile(12, 5)
// CHECK:           %[[VAL_538:.*]] = AIE.switchbox(%[[VAL_537]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<West : 2, East : 2>
// CHECK:             AIE.connect<West : 3, East : 3>
// CHECK:           }
// CHECK:           %[[VAL_539:.*]] = AIE.tile(13, 5)
// CHECK:           %[[VAL_540:.*]] = AIE.switchbox(%[[VAL_539]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<West : 2, East : 2>
// CHECK:             AIE.connect<West : 3, East : 3>
// CHECK:           }
// CHECK:           %[[VAL_541:.*]] = AIE.tile(14, 5)
// CHECK:           %[[VAL_542:.*]] = AIE.switchbox(%[[VAL_541]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<West : 1, South : 0>
// CHECK:             AIE.connect<West : 2, East : 1>
// CHECK:             AIE.connect<West : 3, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_543:.*]] = AIE.tile(15, 5)
// CHECK:           %[[VAL_544:.*]] = AIE.switchbox(%[[VAL_543]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<West : 2, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_545:.*]] = AIE.tile(16, 5)
// CHECK:           %[[VAL_546:.*]] = AIE.switchbox(%[[VAL_545]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<West : 1, East : 0>
// CHECK:             AIE.connect<West : 2, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_547:.*]] = AIE.tile(17, 5)
// CHECK:           %[[VAL_548:.*]] = AIE.switchbox(%[[VAL_547]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_549:.*]] = AIE.tile(18, 5)
// CHECK:           %[[VAL_550:.*]] = AIE.switchbox(%[[VAL_549]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_551:.*]] = AIE.tile(19, 5)
// CHECK:           %[[VAL_552:.*]] = AIE.switchbox(%[[VAL_551]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_553:.*]] = AIE.tile(20, 5)
// CHECK:           %[[VAL_554:.*]] = AIE.switchbox(%[[VAL_553]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_555:.*]] = AIE.tile(21, 5)
// CHECK:           %[[VAL_556:.*]] = AIE.switchbox(%[[VAL_555]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_557:.*]] = AIE.tile(22, 5)
// CHECK:           %[[VAL_558:.*]] = AIE.switchbox(%[[VAL_557]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_559:.*]] = AIE.tile(23, 5)
// CHECK:           %[[VAL_560:.*]] = AIE.switchbox(%[[VAL_559]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_561:.*]] = AIE.tile(24, 5)
// CHECK:           %[[VAL_562:.*]] = AIE.switchbox(%[[VAL_561]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_563:.*]] = AIE.tile(25, 5)
// CHECK:           %[[VAL_564:.*]] = AIE.switchbox(%[[VAL_563]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_565:.*]] = AIE.tile(26, 5)
// CHECK:           %[[VAL_566:.*]] = AIE.switchbox(%[[VAL_565]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_567:.*]] = AIE.tile(27, 5)
// CHECK:           %[[VAL_568:.*]] = AIE.switchbox(%[[VAL_567]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_569:.*]] = AIE.tile(28, 5)
// CHECK:           %[[VAL_570:.*]] = AIE.switchbox(%[[VAL_569]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_571:.*]] = AIE.tile(29, 5)
// CHECK:           %[[VAL_572:.*]] = AIE.switchbox(%[[VAL_571]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_573:.*]] = AIE.tile(30, 5)
// CHECK:           %[[VAL_574:.*]] = AIE.switchbox(%[[VAL_573]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_575:.*]] = AIE.tile(31, 5)
// CHECK:           %[[VAL_576:.*]] = AIE.switchbox(%[[VAL_575]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_577:.*]] = AIE.tile(32, 5)
// CHECK:           %[[VAL_578:.*]] = AIE.switchbox(%[[VAL_577]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_579:.*]] = AIE.tile(33, 5)
// CHECK:           %[[VAL_580:.*]] = AIE.switchbox(%[[VAL_579]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_581:.*]] = AIE.tile(34, 5)
// CHECK:           %[[VAL_582:.*]] = AIE.switchbox(%[[VAL_581]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_583:.*]] = AIE.tile(35, 5)
// CHECK:           %[[VAL_584:.*]] = AIE.switchbox(%[[VAL_583]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_585:.*]] = AIE.tile(36, 5)
// CHECK:           %[[VAL_586:.*]] = AIE.switchbox(%[[VAL_585]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:             AIE.connect<East : 0, West : 2>
// CHECK:             AIE.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_587:.*]] = AIE.switchbox(%[[VAL_42]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_588:.*]] = AIE.shimmux(%[[VAL_42]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_589:.*]] = AIE.switchbox(%[[VAL_41]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_590:.*]] = AIE.switchbox(%[[VAL_40]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_591:.*]] = AIE.tile(43, 3)
// CHECK:           %[[VAL_592:.*]] = AIE.switchbox(%[[VAL_591]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_593:.*]] = AIE.tile(42, 4)
// CHECK:           %[[VAL_594:.*]] = AIE.switchbox(%[[VAL_593]]) {
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:             AIE.connect<East : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_595:.*]] = AIE.tile(43, 4)
// CHECK:           %[[VAL_596:.*]] = AIE.switchbox(%[[VAL_595]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_597:.*]] = AIE.tile(37, 5)
// CHECK:           %[[VAL_598:.*]] = AIE.switchbox(%[[VAL_597]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_599:.*]] = AIE.tile(38, 5)
// CHECK:           %[[VAL_600:.*]] = AIE.switchbox(%[[VAL_599]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_601:.*]] = AIE.tile(39, 5)
// CHECK:           %[[VAL_602:.*]] = AIE.switchbox(%[[VAL_601]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_603:.*]] = AIE.tile(40, 5)
// CHECK:           %[[VAL_604:.*]] = AIE.switchbox(%[[VAL_603]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_605:.*]] = AIE.tile(41, 5)
// CHECK:           %[[VAL_606:.*]] = AIE.switchbox(%[[VAL_605]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_607:.*]] = AIE.tile(42, 5)
// CHECK:           %[[VAL_608:.*]] = AIE.switchbox(%[[VAL_607]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_609:.*]] = AIE.switchbox(%[[VAL_22]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_610:.*]] = AIE.shimmux(%[[VAL_22]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_611:.*]] = AIE.tile(44, 1)
// CHECK:           %[[VAL_612:.*]] = AIE.switchbox(%[[VAL_611]]) {
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:             AIE.connect<East : 1, North : 1>
// CHECK:             AIE.connect<East : 2, West : 0>
// CHECK:             AIE.connect<East : 3, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_613:.*]] = AIE.tile(45, 1)
// CHECK:           %[[VAL_614:.*]] = AIE.switchbox(%[[VAL_613]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_615:.*]] = AIE.switchbox(%[[VAL_21]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:             AIE.connect<East : 0, West : 2>
// CHECK:             AIE.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_616:.*]] = AIE.switchbox(%[[VAL_80]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_617:.*]] = AIE.tile(44, 2)
// CHECK:           %[[VAL_618:.*]] = AIE.switchbox(%[[VAL_617]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_619:.*]] = AIE.tile(27, 3)
// CHECK:           %[[VAL_620:.*]] = AIE.switchbox(%[[VAL_619]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_621:.*]] = AIE.tile(27, 4)
// CHECK:           %[[VAL_622:.*]] = AIE.switchbox(%[[VAL_621]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_623:.*]] = AIE.switchbox(%[[VAL_2]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_624:.*]] = AIE.shimmux(%[[VAL_2]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_625:.*]] = AIE.tile(36, 1)
// CHECK:           %[[VAL_626:.*]] = AIE.switchbox(%[[VAL_625]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_627:.*]] = AIE.tile(37, 1)
// CHECK:           %[[VAL_628:.*]] = AIE.switchbox(%[[VAL_627]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_629:.*]] = AIE.tile(38, 1)
// CHECK:           %[[VAL_630:.*]] = AIE.switchbox(%[[VAL_629]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_631:.*]] = AIE.tile(39, 1)
// CHECK:           %[[VAL_632:.*]] = AIE.switchbox(%[[VAL_631]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_633:.*]] = AIE.tile(40, 1)
// CHECK:           %[[VAL_634:.*]] = AIE.switchbox(%[[VAL_633]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_635:.*]] = AIE.tile(41, 1)
// CHECK:           %[[VAL_636:.*]] = AIE.switchbox(%[[VAL_635]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_637:.*]] = AIE.switchbox(%[[VAL_1]]) {
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_313]] : North, %[[VAL_312]] : South)
// CHECK:           AIE.wire(%[[VAL_294]] : DMA, %[[VAL_313]] : DMA)
// CHECK:           AIE.wire(%[[VAL_293]] : Core, %[[VAL_314]] : Core)
// CHECK:           AIE.wire(%[[VAL_293]] : DMA, %[[VAL_314]] : DMA)
// CHECK:           AIE.wire(%[[VAL_312]] : North, %[[VAL_314]] : South)
// CHECK:           AIE.wire(%[[VAL_292]] : Core, %[[VAL_315]] : Core)
// CHECK:           AIE.wire(%[[VAL_292]] : DMA, %[[VAL_315]] : DMA)
// CHECK:           AIE.wire(%[[VAL_314]] : North, %[[VAL_315]] : South)
// CHECK:           AIE.wire(%[[VAL_312]] : East, %[[VAL_324]] : West)
// CHECK:           AIE.wire(%[[VAL_325]] : North, %[[VAL_324]] : South)
// CHECK:           AIE.wire(%[[VAL_274]] : DMA, %[[VAL_325]] : DMA)
// CHECK:           AIE.wire(%[[VAL_314]] : East, %[[VAL_323]] : West)
// CHECK:           AIE.wire(%[[VAL_273]] : Core, %[[VAL_323]] : Core)
// CHECK:           AIE.wire(%[[VAL_273]] : DMA, %[[VAL_323]] : DMA)
// CHECK:           AIE.wire(%[[VAL_324]] : North, %[[VAL_323]] : South)
// CHECK:           AIE.wire(%[[VAL_315]] : East, %[[VAL_316]] : West)
// CHECK:           AIE.wire(%[[VAL_272]] : Core, %[[VAL_316]] : Core)
// CHECK:           AIE.wire(%[[VAL_272]] : DMA, %[[VAL_316]] : DMA)
// CHECK:           AIE.wire(%[[VAL_323]] : North, %[[VAL_316]] : South)
// CHECK:           AIE.wire(%[[VAL_324]] : East, %[[VAL_327]] : West)
// CHECK:           AIE.wire(%[[VAL_316]] : East, %[[VAL_318]] : West)
// CHECK:           AIE.wire(%[[VAL_317]] : Core, %[[VAL_318]] : Core)
// CHECK:           AIE.wire(%[[VAL_317]] : DMA, %[[VAL_318]] : DMA)
// CHECK:           AIE.wire(%[[VAL_327]] : East, %[[VAL_329]] : West)
// CHECK:           AIE.wire(%[[VAL_318]] : East, %[[VAL_320]] : West)
// CHECK:           AIE.wire(%[[VAL_319]] : Core, %[[VAL_320]] : Core)
// CHECK:           AIE.wire(%[[VAL_319]] : DMA, %[[VAL_320]] : DMA)
// CHECK:           AIE.wire(%[[VAL_329]] : East, %[[VAL_330]] : West)
// CHECK:           AIE.wire(%[[VAL_333]] : North, %[[VAL_330]] : South)
// CHECK:           AIE.wire(%[[VAL_255]] : DMA, %[[VAL_333]] : DMA)
// CHECK:           AIE.wire(%[[VAL_254]] : Core, %[[VAL_331]] : Core)
// CHECK:           AIE.wire(%[[VAL_254]] : DMA, %[[VAL_331]] : DMA)
// CHECK:           AIE.wire(%[[VAL_330]] : North, %[[VAL_331]] : South)
// CHECK:           AIE.wire(%[[VAL_320]] : East, %[[VAL_321]] : West)
// CHECK:           AIE.wire(%[[VAL_253]] : Core, %[[VAL_321]] : Core)
// CHECK:           AIE.wire(%[[VAL_253]] : DMA, %[[VAL_321]] : DMA)
// CHECK:           AIE.wire(%[[VAL_331]] : North, %[[VAL_321]] : South)
// CHECK:           AIE.wire(%[[VAL_330]] : East, %[[VAL_334]] : West)
// CHECK:           AIE.wire(%[[VAL_340]] : North, %[[VAL_334]] : South)
// CHECK:           AIE.wire(%[[VAL_236]] : DMA, %[[VAL_340]] : DMA)
// CHECK:           AIE.wire(%[[VAL_331]] : East, %[[VAL_351]] : West)
// CHECK:           AIE.wire(%[[VAL_235]] : Core, %[[VAL_351]] : Core)
// CHECK:           AIE.wire(%[[VAL_235]] : DMA, %[[VAL_351]] : DMA)
// CHECK:           AIE.wire(%[[VAL_334]] : North, %[[VAL_351]] : South)
// CHECK:           AIE.wire(%[[VAL_321]] : East, %[[VAL_322]] : West)
// CHECK:           AIE.wire(%[[VAL_296]] : Core, %[[VAL_322]] : Core)
// CHECK:           AIE.wire(%[[VAL_296]] : DMA, %[[VAL_322]] : DMA)
// CHECK:           AIE.wire(%[[VAL_351]] : North, %[[VAL_322]] : South)
// CHECK:           AIE.wire(%[[VAL_219]] : Core, %[[VAL_349]] : Core)
// CHECK:           AIE.wire(%[[VAL_219]] : DMA, %[[VAL_349]] : DMA)
// CHECK:           AIE.wire(%[[VAL_322]] : North, %[[VAL_349]] : South)
// CHECK:           AIE.wire(%[[VAL_142]] : Core, %[[VAL_405]] : Core)
// CHECK:           AIE.wire(%[[VAL_142]] : DMA, %[[VAL_405]] : DMA)
// CHECK:           AIE.wire(%[[VAL_349]] : North, %[[VAL_405]] : South)
// CHECK:           AIE.wire(%[[VAL_64]] : Core, %[[VAL_531]] : Core)
// CHECK:           AIE.wire(%[[VAL_64]] : DMA, %[[VAL_531]] : DMA)
// CHECK:           AIE.wire(%[[VAL_405]] : North, %[[VAL_531]] : South)
// CHECK:           AIE.wire(%[[VAL_334]] : East, %[[VAL_336]] : West)
// CHECK:           AIE.wire(%[[VAL_351]] : East, %[[VAL_338]] : West)
// CHECK:           AIE.wire(%[[VAL_337]] : Core, %[[VAL_338]] : Core)
// CHECK:           AIE.wire(%[[VAL_337]] : DMA, %[[VAL_338]] : DMA)
// CHECK:           AIE.wire(%[[VAL_336]] : North, %[[VAL_338]] : South)
// CHECK:           AIE.wire(%[[VAL_322]] : East, %[[VAL_332]] : West)
// CHECK:           AIE.wire(%[[VAL_276]] : Core, %[[VAL_332]] : Core)
// CHECK:           AIE.wire(%[[VAL_276]] : DMA, %[[VAL_332]] : DMA)
// CHECK:           AIE.wire(%[[VAL_338]] : North, %[[VAL_332]] : South)
// CHECK:           AIE.wire(%[[VAL_349]] : East, %[[VAL_350]] : West)
// CHECK:           AIE.wire(%[[VAL_200]] : Core, %[[VAL_350]] : Core)
// CHECK:           AIE.wire(%[[VAL_200]] : DMA, %[[VAL_350]] : DMA)
// CHECK:           AIE.wire(%[[VAL_332]] : North, %[[VAL_350]] : South)
// CHECK:           AIE.wire(%[[VAL_405]] : East, %[[VAL_406]] : West)
// CHECK:           AIE.wire(%[[VAL_122]] : Core, %[[VAL_406]] : Core)
// CHECK:           AIE.wire(%[[VAL_122]] : DMA, %[[VAL_406]] : DMA)
// CHECK:           AIE.wire(%[[VAL_350]] : North, %[[VAL_406]] : South)
// CHECK:           AIE.wire(%[[VAL_531]] : East, %[[VAL_532]] : West)
// CHECK:           AIE.wire(%[[VAL_44]] : Core, %[[VAL_532]] : Core)
// CHECK:           AIE.wire(%[[VAL_44]] : DMA, %[[VAL_532]] : DMA)
// CHECK:           AIE.wire(%[[VAL_406]] : North, %[[VAL_532]] : South)
// CHECK:           AIE.wire(%[[VAL_336]] : East, %[[VAL_342]] : West)
// CHECK:           AIE.wire(%[[VAL_338]] : East, %[[VAL_344]] : West)
// CHECK:           AIE.wire(%[[VAL_343]] : Core, %[[VAL_344]] : Core)
// CHECK:           AIE.wire(%[[VAL_343]] : DMA, %[[VAL_344]] : DMA)
// CHECK:           AIE.wire(%[[VAL_342]] : North, %[[VAL_344]] : South)
// CHECK:           AIE.wire(%[[VAL_332]] : East, %[[VAL_339]] : West)
// CHECK:           AIE.wire(%[[VAL_256]] : Core, %[[VAL_339]] : Core)
// CHECK:           AIE.wire(%[[VAL_256]] : DMA, %[[VAL_339]] : DMA)
// CHECK:           AIE.wire(%[[VAL_344]] : North, %[[VAL_339]] : South)
// CHECK:           AIE.wire(%[[VAL_350]] : East, %[[VAL_370]] : West)
// CHECK:           AIE.wire(%[[VAL_180]] : Core, %[[VAL_370]] : Core)
// CHECK:           AIE.wire(%[[VAL_180]] : DMA, %[[VAL_370]] : DMA)
// CHECK:           AIE.wire(%[[VAL_339]] : North, %[[VAL_370]] : South)
// CHECK:           AIE.wire(%[[VAL_406]] : East, %[[VAL_449]] : West)
// CHECK:           AIE.wire(%[[VAL_102]] : Core, %[[VAL_449]] : Core)
// CHECK:           AIE.wire(%[[VAL_102]] : DMA, %[[VAL_449]] : DMA)
// CHECK:           AIE.wire(%[[VAL_370]] : North, %[[VAL_449]] : South)
// CHECK:           AIE.wire(%[[VAL_532]] : East, %[[VAL_533]] : West)
// CHECK:           AIE.wire(%[[VAL_24]] : Core, %[[VAL_533]] : Core)
// CHECK:           AIE.wire(%[[VAL_24]] : DMA, %[[VAL_533]] : DMA)
// CHECK:           AIE.wire(%[[VAL_449]] : North, %[[VAL_533]] : South)
// CHECK:           AIE.wire(%[[VAL_342]] : East, %[[VAL_347]] : West)
// CHECK:           AIE.wire(%[[VAL_348]] : North, %[[VAL_347]] : South)
// CHECK:           AIE.wire(%[[VAL_217]] : DMA, %[[VAL_348]] : DMA)
// CHECK:           AIE.wire(%[[VAL_344]] : East, %[[VAL_345]] : West)
// CHECK:           AIE.wire(%[[VAL_216]] : Core, %[[VAL_345]] : Core)
// CHECK:           AIE.wire(%[[VAL_216]] : DMA, %[[VAL_345]] : DMA)
// CHECK:           AIE.wire(%[[VAL_347]] : North, %[[VAL_345]] : South)
// CHECK:           AIE.wire(%[[VAL_339]] : East, %[[VAL_346]] : West)
// CHECK:           AIE.wire(%[[VAL_237]] : Core, %[[VAL_346]] : Core)
// CHECK:           AIE.wire(%[[VAL_237]] : DMA, %[[VAL_346]] : DMA)
// CHECK:           AIE.wire(%[[VAL_345]] : North, %[[VAL_346]] : South)
// CHECK:           AIE.wire(%[[VAL_370]] : East, %[[VAL_371]] : West)
// CHECK:           AIE.wire(%[[VAL_161]] : Core, %[[VAL_371]] : Core)
// CHECK:           AIE.wire(%[[VAL_161]] : DMA, %[[VAL_371]] : DMA)
// CHECK:           AIE.wire(%[[VAL_346]] : North, %[[VAL_371]] : South)
// CHECK:           AIE.wire(%[[VAL_449]] : East, %[[VAL_450]] : West)
// CHECK:           AIE.wire(%[[VAL_83]] : Core, %[[VAL_450]] : Core)
// CHECK:           AIE.wire(%[[VAL_83]] : DMA, %[[VAL_450]] : DMA)
// CHECK:           AIE.wire(%[[VAL_371]] : North, %[[VAL_450]] : South)
// CHECK:           AIE.wire(%[[VAL_533]] : East, %[[VAL_534]] : West)
// CHECK:           AIE.wire(%[[VAL_4]] : Core, %[[VAL_534]] : Core)
// CHECK:           AIE.wire(%[[VAL_4]] : DMA, %[[VAL_534]] : DMA)
// CHECK:           AIE.wire(%[[VAL_450]] : North, %[[VAL_534]] : South)
// CHECK:           AIE.wire(%[[VAL_347]] : East, %[[VAL_352]] : West)
// CHECK:           AIE.wire(%[[VAL_353]] : North, %[[VAL_352]] : South)
// CHECK:           AIE.wire(%[[VAL_198]] : DMA, %[[VAL_353]] : DMA)
// CHECK:           AIE.wire(%[[VAL_345]] : East, %[[VAL_356]] : West)
// CHECK:           AIE.wire(%[[VAL_197]] : Core, %[[VAL_356]] : Core)
// CHECK:           AIE.wire(%[[VAL_197]] : DMA, %[[VAL_356]] : DMA)
// CHECK:           AIE.wire(%[[VAL_352]] : North, %[[VAL_356]] : South)
// CHECK:           AIE.wire(%[[VAL_346]] : East, %[[VAL_377]] : West)
// CHECK:           AIE.wire(%[[VAL_196]] : Core, %[[VAL_377]] : Core)
// CHECK:           AIE.wire(%[[VAL_196]] : DMA, %[[VAL_377]] : DMA)
// CHECK:           AIE.wire(%[[VAL_356]] : North, %[[VAL_377]] : South)
// CHECK:           AIE.wire(%[[VAL_371]] : East, %[[VAL_412]] : West)
// CHECK:           AIE.wire(%[[VAL_411]] : Core, %[[VAL_412]] : Core)
// CHECK:           AIE.wire(%[[VAL_411]] : DMA, %[[VAL_412]] : DMA)
// CHECK:           AIE.wire(%[[VAL_377]] : North, %[[VAL_412]] : South)
// CHECK:           AIE.wire(%[[VAL_450]] : East, %[[VAL_452]] : West)
// CHECK:           AIE.wire(%[[VAL_451]] : Core, %[[VAL_452]] : Core)
// CHECK:           AIE.wire(%[[VAL_451]] : DMA, %[[VAL_452]] : DMA)
// CHECK:           AIE.wire(%[[VAL_412]] : North, %[[VAL_452]] : South)
// CHECK:           AIE.wire(%[[VAL_534]] : East, %[[VAL_536]] : West)
// CHECK:           AIE.wire(%[[VAL_535]] : Core, %[[VAL_536]] : Core)
// CHECK:           AIE.wire(%[[VAL_535]] : DMA, %[[VAL_536]] : DMA)
// CHECK:           AIE.wire(%[[VAL_452]] : North, %[[VAL_536]] : South)
// CHECK:           AIE.wire(%[[VAL_356]] : East, %[[VAL_358]] : West)
// CHECK:           AIE.wire(%[[VAL_357]] : Core, %[[VAL_358]] : Core)
// CHECK:           AIE.wire(%[[VAL_357]] : DMA, %[[VAL_358]] : DMA)
// CHECK:           AIE.wire(%[[VAL_377]] : East, %[[VAL_379]] : West)
// CHECK:           AIE.wire(%[[VAL_378]] : Core, %[[VAL_379]] : Core)
// CHECK:           AIE.wire(%[[VAL_378]] : DMA, %[[VAL_379]] : DMA)
// CHECK:           AIE.wire(%[[VAL_358]] : North, %[[VAL_379]] : South)
// CHECK:           AIE.wire(%[[VAL_412]] : East, %[[VAL_414]] : West)
// CHECK:           AIE.wire(%[[VAL_413]] : Core, %[[VAL_414]] : Core)
// CHECK:           AIE.wire(%[[VAL_413]] : DMA, %[[VAL_414]] : DMA)
// CHECK:           AIE.wire(%[[VAL_379]] : North, %[[VAL_414]] : South)
// CHECK:           AIE.wire(%[[VAL_452]] : East, %[[VAL_454]] : West)
// CHECK:           AIE.wire(%[[VAL_453]] : Core, %[[VAL_454]] : Core)
// CHECK:           AIE.wire(%[[VAL_453]] : DMA, %[[VAL_454]] : DMA)
// CHECK:           AIE.wire(%[[VAL_414]] : North, %[[VAL_454]] : South)
// CHECK:           AIE.wire(%[[VAL_536]] : East, %[[VAL_538]] : West)
// CHECK:           AIE.wire(%[[VAL_537]] : Core, %[[VAL_538]] : Core)
// CHECK:           AIE.wire(%[[VAL_537]] : DMA, %[[VAL_538]] : DMA)
// CHECK:           AIE.wire(%[[VAL_454]] : North, %[[VAL_538]] : South)
// CHECK:           AIE.wire(%[[VAL_358]] : East, %[[VAL_360]] : West)
// CHECK:           AIE.wire(%[[VAL_359]] : Core, %[[VAL_360]] : Core)
// CHECK:           AIE.wire(%[[VAL_359]] : DMA, %[[VAL_360]] : DMA)
// CHECK:           AIE.wire(%[[VAL_379]] : East, %[[VAL_381]] : West)
// CHECK:           AIE.wire(%[[VAL_380]] : Core, %[[VAL_381]] : Core)
// CHECK:           AIE.wire(%[[VAL_380]] : DMA, %[[VAL_381]] : DMA)
// CHECK:           AIE.wire(%[[VAL_360]] : North, %[[VAL_381]] : South)
// CHECK:           AIE.wire(%[[VAL_414]] : East, %[[VAL_416]] : West)
// CHECK:           AIE.wire(%[[VAL_415]] : Core, %[[VAL_416]] : Core)
// CHECK:           AIE.wire(%[[VAL_415]] : DMA, %[[VAL_416]] : DMA)
// CHECK:           AIE.wire(%[[VAL_381]] : North, %[[VAL_416]] : South)
// CHECK:           AIE.wire(%[[VAL_454]] : East, %[[VAL_456]] : West)
// CHECK:           AIE.wire(%[[VAL_455]] : Core, %[[VAL_456]] : Core)
// CHECK:           AIE.wire(%[[VAL_455]] : DMA, %[[VAL_456]] : DMA)
// CHECK:           AIE.wire(%[[VAL_416]] : North, %[[VAL_456]] : South)
// CHECK:           AIE.wire(%[[VAL_538]] : East, %[[VAL_540]] : West)
// CHECK:           AIE.wire(%[[VAL_539]] : Core, %[[VAL_540]] : Core)
// CHECK:           AIE.wire(%[[VAL_539]] : DMA, %[[VAL_540]] : DMA)
// CHECK:           AIE.wire(%[[VAL_456]] : North, %[[VAL_540]] : South)
// CHECK:           AIE.wire(%[[VAL_360]] : East, %[[VAL_362]] : West)
// CHECK:           AIE.wire(%[[VAL_361]] : Core, %[[VAL_362]] : Core)
// CHECK:           AIE.wire(%[[VAL_361]] : DMA, %[[VAL_362]] : DMA)
// CHECK:           AIE.wire(%[[VAL_381]] : East, %[[VAL_383]] : West)
// CHECK:           AIE.wire(%[[VAL_382]] : Core, %[[VAL_383]] : Core)
// CHECK:           AIE.wire(%[[VAL_382]] : DMA, %[[VAL_383]] : DMA)
// CHECK:           AIE.wire(%[[VAL_362]] : North, %[[VAL_383]] : South)
// CHECK:           AIE.wire(%[[VAL_416]] : East, %[[VAL_418]] : West)
// CHECK:           AIE.wire(%[[VAL_417]] : Core, %[[VAL_418]] : Core)
// CHECK:           AIE.wire(%[[VAL_417]] : DMA, %[[VAL_418]] : DMA)
// CHECK:           AIE.wire(%[[VAL_383]] : North, %[[VAL_418]] : South)
// CHECK:           AIE.wire(%[[VAL_456]] : East, %[[VAL_458]] : West)
// CHECK:           AIE.wire(%[[VAL_457]] : Core, %[[VAL_458]] : Core)
// CHECK:           AIE.wire(%[[VAL_457]] : DMA, %[[VAL_458]] : DMA)
// CHECK:           AIE.wire(%[[VAL_418]] : North, %[[VAL_458]] : South)
// CHECK:           AIE.wire(%[[VAL_540]] : East, %[[VAL_542]] : West)
// CHECK:           AIE.wire(%[[VAL_541]] : Core, %[[VAL_542]] : Core)
// CHECK:           AIE.wire(%[[VAL_541]] : DMA, %[[VAL_542]] : DMA)
// CHECK:           AIE.wire(%[[VAL_458]] : North, %[[VAL_542]] : South)
// CHECK:           AIE.wire(%[[VAL_362]] : East, %[[VAL_364]] : West)
// CHECK:           AIE.wire(%[[VAL_363]] : Core, %[[VAL_364]] : Core)
// CHECK:           AIE.wire(%[[VAL_363]] : DMA, %[[VAL_364]] : DMA)
// CHECK:           AIE.wire(%[[VAL_383]] : East, %[[VAL_385]] : West)
// CHECK:           AIE.wire(%[[VAL_384]] : Core, %[[VAL_385]] : Core)
// CHECK:           AIE.wire(%[[VAL_384]] : DMA, %[[VAL_385]] : DMA)
// CHECK:           AIE.wire(%[[VAL_364]] : North, %[[VAL_385]] : South)
// CHECK:           AIE.wire(%[[VAL_418]] : East, %[[VAL_420]] : West)
// CHECK:           AIE.wire(%[[VAL_419]] : Core, %[[VAL_420]] : Core)
// CHECK:           AIE.wire(%[[VAL_419]] : DMA, %[[VAL_420]] : DMA)
// CHECK:           AIE.wire(%[[VAL_385]] : North, %[[VAL_420]] : South)
// CHECK:           AIE.wire(%[[VAL_458]] : East, %[[VAL_460]] : West)
// CHECK:           AIE.wire(%[[VAL_459]] : Core, %[[VAL_460]] : Core)
// CHECK:           AIE.wire(%[[VAL_459]] : DMA, %[[VAL_460]] : DMA)
// CHECK:           AIE.wire(%[[VAL_420]] : North, %[[VAL_460]] : South)
// CHECK:           AIE.wire(%[[VAL_542]] : East, %[[VAL_544]] : West)
// CHECK:           AIE.wire(%[[VAL_543]] : Core, %[[VAL_544]] : Core)
// CHECK:           AIE.wire(%[[VAL_543]] : DMA, %[[VAL_544]] : DMA)
// CHECK:           AIE.wire(%[[VAL_460]] : North, %[[VAL_544]] : South)
// CHECK:           AIE.wire(%[[VAL_364]] : East, %[[VAL_366]] : West)
// CHECK:           AIE.wire(%[[VAL_365]] : Core, %[[VAL_366]] : Core)
// CHECK:           AIE.wire(%[[VAL_365]] : DMA, %[[VAL_366]] : DMA)
// CHECK:           AIE.wire(%[[VAL_385]] : East, %[[VAL_387]] : West)
// CHECK:           AIE.wire(%[[VAL_386]] : Core, %[[VAL_387]] : Core)
// CHECK:           AIE.wire(%[[VAL_386]] : DMA, %[[VAL_387]] : DMA)
// CHECK:           AIE.wire(%[[VAL_366]] : North, %[[VAL_387]] : South)
// CHECK:           AIE.wire(%[[VAL_420]] : East, %[[VAL_422]] : West)
// CHECK:           AIE.wire(%[[VAL_421]] : Core, %[[VAL_422]] : Core)
// CHECK:           AIE.wire(%[[VAL_421]] : DMA, %[[VAL_422]] : DMA)
// CHECK:           AIE.wire(%[[VAL_387]] : North, %[[VAL_422]] : South)
// CHECK:           AIE.wire(%[[VAL_460]] : East, %[[VAL_462]] : West)
// CHECK:           AIE.wire(%[[VAL_461]] : Core, %[[VAL_462]] : Core)
// CHECK:           AIE.wire(%[[VAL_461]] : DMA, %[[VAL_462]] : DMA)
// CHECK:           AIE.wire(%[[VAL_422]] : North, %[[VAL_462]] : South)
// CHECK:           AIE.wire(%[[VAL_544]] : East, %[[VAL_546]] : West)
// CHECK:           AIE.wire(%[[VAL_545]] : Core, %[[VAL_546]] : Core)
// CHECK:           AIE.wire(%[[VAL_545]] : DMA, %[[VAL_546]] : DMA)
// CHECK:           AIE.wire(%[[VAL_462]] : North, %[[VAL_546]] : South)
// CHECK:           AIE.wire(%[[VAL_366]] : East, %[[VAL_368]] : West)
// CHECK:           AIE.wire(%[[VAL_367]] : Core, %[[VAL_368]] : Core)
// CHECK:           AIE.wire(%[[VAL_367]] : DMA, %[[VAL_368]] : DMA)
// CHECK:           AIE.wire(%[[VAL_387]] : East, %[[VAL_389]] : West)
// CHECK:           AIE.wire(%[[VAL_388]] : Core, %[[VAL_389]] : Core)
// CHECK:           AIE.wire(%[[VAL_388]] : DMA, %[[VAL_389]] : DMA)
// CHECK:           AIE.wire(%[[VAL_368]] : North, %[[VAL_389]] : South)
// CHECK:           AIE.wire(%[[VAL_422]] : East, %[[VAL_424]] : West)
// CHECK:           AIE.wire(%[[VAL_423]] : Core, %[[VAL_424]] : Core)
// CHECK:           AIE.wire(%[[VAL_423]] : DMA, %[[VAL_424]] : DMA)
// CHECK:           AIE.wire(%[[VAL_389]] : North, %[[VAL_424]] : South)
// CHECK:           AIE.wire(%[[VAL_462]] : East, %[[VAL_464]] : West)
// CHECK:           AIE.wire(%[[VAL_463]] : Core, %[[VAL_464]] : Core)
// CHECK:           AIE.wire(%[[VAL_463]] : DMA, %[[VAL_464]] : DMA)
// CHECK:           AIE.wire(%[[VAL_424]] : North, %[[VAL_464]] : South)
// CHECK:           AIE.wire(%[[VAL_546]] : East, %[[VAL_548]] : West)
// CHECK:           AIE.wire(%[[VAL_547]] : Core, %[[VAL_548]] : Core)
// CHECK:           AIE.wire(%[[VAL_547]] : DMA, %[[VAL_548]] : DMA)
// CHECK:           AIE.wire(%[[VAL_464]] : North, %[[VAL_548]] : South)
// CHECK:           AIE.wire(%[[VAL_355]] : North, %[[VAL_354]] : South)
// CHECK:           AIE.wire(%[[VAL_179]] : DMA, %[[VAL_355]] : DMA)
// CHECK:           AIE.wire(%[[VAL_368]] : East, %[[VAL_369]] : West)
// CHECK:           AIE.wire(%[[VAL_178]] : Core, %[[VAL_369]] : Core)
// CHECK:           AIE.wire(%[[VAL_178]] : DMA, %[[VAL_369]] : DMA)
// CHECK:           AIE.wire(%[[VAL_354]] : North, %[[VAL_369]] : South)
// CHECK:           AIE.wire(%[[VAL_389]] : East, %[[VAL_390]] : West)
// CHECK:           AIE.wire(%[[VAL_177]] : Core, %[[VAL_390]] : Core)
// CHECK:           AIE.wire(%[[VAL_177]] : DMA, %[[VAL_390]] : DMA)
// CHECK:           AIE.wire(%[[VAL_369]] : North, %[[VAL_390]] : South)
// CHECK:           AIE.wire(%[[VAL_424]] : East, %[[VAL_426]] : West)
// CHECK:           AIE.wire(%[[VAL_425]] : Core, %[[VAL_426]] : Core)
// CHECK:           AIE.wire(%[[VAL_425]] : DMA, %[[VAL_426]] : DMA)
// CHECK:           AIE.wire(%[[VAL_390]] : North, %[[VAL_426]] : South)
// CHECK:           AIE.wire(%[[VAL_464]] : East, %[[VAL_466]] : West)
// CHECK:           AIE.wire(%[[VAL_465]] : Core, %[[VAL_466]] : Core)
// CHECK:           AIE.wire(%[[VAL_465]] : DMA, %[[VAL_466]] : DMA)
// CHECK:           AIE.wire(%[[VAL_426]] : North, %[[VAL_466]] : South)
// CHECK:           AIE.wire(%[[VAL_548]] : East, %[[VAL_550]] : West)
// CHECK:           AIE.wire(%[[VAL_549]] : Core, %[[VAL_550]] : Core)
// CHECK:           AIE.wire(%[[VAL_549]] : DMA, %[[VAL_550]] : DMA)
// CHECK:           AIE.wire(%[[VAL_466]] : North, %[[VAL_550]] : South)
// CHECK:           AIE.wire(%[[VAL_354]] : East, %[[VAL_372]] : West)
// CHECK:           AIE.wire(%[[VAL_373]] : North, %[[VAL_372]] : South)
// CHECK:           AIE.wire(%[[VAL_160]] : DMA, %[[VAL_373]] : DMA)
// CHECK:           AIE.wire(%[[VAL_369]] : East, %[[VAL_485]] : West)
// CHECK:           AIE.wire(%[[VAL_159]] : Core, %[[VAL_485]] : Core)
// CHECK:           AIE.wire(%[[VAL_159]] : DMA, %[[VAL_485]] : DMA)
// CHECK:           AIE.wire(%[[VAL_372]] : North, %[[VAL_485]] : South)
// CHECK:           AIE.wire(%[[VAL_390]] : East, %[[VAL_391]] : West)
// CHECK:           AIE.wire(%[[VAL_158]] : Core, %[[VAL_391]] : Core)
// CHECK:           AIE.wire(%[[VAL_158]] : DMA, %[[VAL_391]] : DMA)
// CHECK:           AIE.wire(%[[VAL_485]] : North, %[[VAL_391]] : South)
// CHECK:           AIE.wire(%[[VAL_426]] : East, %[[VAL_428]] : West)
// CHECK:           AIE.wire(%[[VAL_427]] : Core, %[[VAL_428]] : Core)
// CHECK:           AIE.wire(%[[VAL_427]] : DMA, %[[VAL_428]] : DMA)
// CHECK:           AIE.wire(%[[VAL_391]] : North, %[[VAL_428]] : South)
// CHECK:           AIE.wire(%[[VAL_466]] : East, %[[VAL_468]] : West)
// CHECK:           AIE.wire(%[[VAL_467]] : Core, %[[VAL_468]] : Core)
// CHECK:           AIE.wire(%[[VAL_467]] : DMA, %[[VAL_468]] : DMA)
// CHECK:           AIE.wire(%[[VAL_428]] : North, %[[VAL_468]] : South)
// CHECK:           AIE.wire(%[[VAL_550]] : East, %[[VAL_552]] : West)
// CHECK:           AIE.wire(%[[VAL_551]] : Core, %[[VAL_552]] : Core)
// CHECK:           AIE.wire(%[[VAL_551]] : DMA, %[[VAL_552]] : DMA)
// CHECK:           AIE.wire(%[[VAL_468]] : North, %[[VAL_552]] : South)
// CHECK:           AIE.wire(%[[VAL_485]] : East, %[[VAL_487]] : West)
// CHECK:           AIE.wire(%[[VAL_486]] : Core, %[[VAL_487]] : Core)
// CHECK:           AIE.wire(%[[VAL_486]] : DMA, %[[VAL_487]] : DMA)
// CHECK:           AIE.wire(%[[VAL_391]] : East, %[[VAL_393]] : West)
// CHECK:           AIE.wire(%[[VAL_392]] : Core, %[[VAL_393]] : Core)
// CHECK:           AIE.wire(%[[VAL_392]] : DMA, %[[VAL_393]] : DMA)
// CHECK:           AIE.wire(%[[VAL_487]] : North, %[[VAL_393]] : South)
// CHECK:           AIE.wire(%[[VAL_428]] : East, %[[VAL_430]] : West)
// CHECK:           AIE.wire(%[[VAL_429]] : Core, %[[VAL_430]] : Core)
// CHECK:           AIE.wire(%[[VAL_429]] : DMA, %[[VAL_430]] : DMA)
// CHECK:           AIE.wire(%[[VAL_393]] : North, %[[VAL_430]] : South)
// CHECK:           AIE.wire(%[[VAL_468]] : East, %[[VAL_470]] : West)
// CHECK:           AIE.wire(%[[VAL_469]] : Core, %[[VAL_470]] : Core)
// CHECK:           AIE.wire(%[[VAL_469]] : DMA, %[[VAL_470]] : DMA)
// CHECK:           AIE.wire(%[[VAL_430]] : North, %[[VAL_470]] : South)
// CHECK:           AIE.wire(%[[VAL_552]] : East, %[[VAL_554]] : West)
// CHECK:           AIE.wire(%[[VAL_553]] : Core, %[[VAL_554]] : Core)
// CHECK:           AIE.wire(%[[VAL_553]] : DMA, %[[VAL_554]] : DMA)
// CHECK:           AIE.wire(%[[VAL_470]] : North, %[[VAL_554]] : South)
// CHECK:           AIE.wire(%[[VAL_487]] : East, %[[VAL_489]] : West)
// CHECK:           AIE.wire(%[[VAL_488]] : Core, %[[VAL_489]] : Core)
// CHECK:           AIE.wire(%[[VAL_488]] : DMA, %[[VAL_489]] : DMA)
// CHECK:           AIE.wire(%[[VAL_393]] : East, %[[VAL_395]] : West)
// CHECK:           AIE.wire(%[[VAL_394]] : Core, %[[VAL_395]] : Core)
// CHECK:           AIE.wire(%[[VAL_394]] : DMA, %[[VAL_395]] : DMA)
// CHECK:           AIE.wire(%[[VAL_489]] : North, %[[VAL_395]] : South)
// CHECK:           AIE.wire(%[[VAL_470]] : East, %[[VAL_472]] : West)
// CHECK:           AIE.wire(%[[VAL_471]] : Core, %[[VAL_472]] : Core)
// CHECK:           AIE.wire(%[[VAL_471]] : DMA, %[[VAL_472]] : DMA)
// CHECK:           AIE.wire(%[[VAL_554]] : East, %[[VAL_556]] : West)
// CHECK:           AIE.wire(%[[VAL_555]] : Core, %[[VAL_556]] : Core)
// CHECK:           AIE.wire(%[[VAL_555]] : DMA, %[[VAL_556]] : DMA)
// CHECK:           AIE.wire(%[[VAL_472]] : North, %[[VAL_556]] : South)
// CHECK:           AIE.wire(%[[VAL_489]] : East, %[[VAL_491]] : West)
// CHECK:           AIE.wire(%[[VAL_490]] : Core, %[[VAL_491]] : Core)
// CHECK:           AIE.wire(%[[VAL_490]] : DMA, %[[VAL_491]] : DMA)
// CHECK:           AIE.wire(%[[VAL_395]] : East, %[[VAL_397]] : West)
// CHECK:           AIE.wire(%[[VAL_396]] : Core, %[[VAL_397]] : Core)
// CHECK:           AIE.wire(%[[VAL_396]] : DMA, %[[VAL_397]] : DMA)
// CHECK:           AIE.wire(%[[VAL_491]] : North, %[[VAL_397]] : South)
// CHECK:           AIE.wire(%[[VAL_472]] : East, %[[VAL_474]] : West)
// CHECK:           AIE.wire(%[[VAL_473]] : Core, %[[VAL_474]] : Core)
// CHECK:           AIE.wire(%[[VAL_473]] : DMA, %[[VAL_474]] : DMA)
// CHECK:           AIE.wire(%[[VAL_556]] : East, %[[VAL_558]] : West)
// CHECK:           AIE.wire(%[[VAL_557]] : Core, %[[VAL_558]] : Core)
// CHECK:           AIE.wire(%[[VAL_557]] : DMA, %[[VAL_558]] : DMA)
// CHECK:           AIE.wire(%[[VAL_474]] : North, %[[VAL_558]] : South)
// CHECK:           AIE.wire(%[[VAL_491]] : East, %[[VAL_493]] : West)
// CHECK:           AIE.wire(%[[VAL_492]] : Core, %[[VAL_493]] : Core)
// CHECK:           AIE.wire(%[[VAL_492]] : DMA, %[[VAL_493]] : DMA)
// CHECK:           AIE.wire(%[[VAL_397]] : East, %[[VAL_399]] : West)
// CHECK:           AIE.wire(%[[VAL_398]] : Core, %[[VAL_399]] : Core)
// CHECK:           AIE.wire(%[[VAL_398]] : DMA, %[[VAL_399]] : DMA)
// CHECK:           AIE.wire(%[[VAL_493]] : North, %[[VAL_399]] : South)
// CHECK:           AIE.wire(%[[VAL_474]] : East, %[[VAL_476]] : West)
// CHECK:           AIE.wire(%[[VAL_475]] : Core, %[[VAL_476]] : Core)
// CHECK:           AIE.wire(%[[VAL_475]] : DMA, %[[VAL_476]] : DMA)
// CHECK:           AIE.wire(%[[VAL_558]] : East, %[[VAL_560]] : West)
// CHECK:           AIE.wire(%[[VAL_559]] : Core, %[[VAL_560]] : Core)
// CHECK:           AIE.wire(%[[VAL_559]] : DMA, %[[VAL_560]] : DMA)
// CHECK:           AIE.wire(%[[VAL_476]] : North, %[[VAL_560]] : South)
// CHECK:           AIE.wire(%[[VAL_493]] : East, %[[VAL_495]] : West)
// CHECK:           AIE.wire(%[[VAL_494]] : Core, %[[VAL_495]] : Core)
// CHECK:           AIE.wire(%[[VAL_494]] : DMA, %[[VAL_495]] : DMA)
// CHECK:           AIE.wire(%[[VAL_399]] : East, %[[VAL_401]] : West)
// CHECK:           AIE.wire(%[[VAL_400]] : Core, %[[VAL_401]] : Core)
// CHECK:           AIE.wire(%[[VAL_400]] : DMA, %[[VAL_401]] : DMA)
// CHECK:           AIE.wire(%[[VAL_495]] : North, %[[VAL_401]] : South)
// CHECK:           AIE.wire(%[[VAL_476]] : East, %[[VAL_478]] : West)
// CHECK:           AIE.wire(%[[VAL_477]] : Core, %[[VAL_478]] : Core)
// CHECK:           AIE.wire(%[[VAL_477]] : DMA, %[[VAL_478]] : DMA)
// CHECK:           AIE.wire(%[[VAL_560]] : East, %[[VAL_562]] : West)
// CHECK:           AIE.wire(%[[VAL_561]] : Core, %[[VAL_562]] : Core)
// CHECK:           AIE.wire(%[[VAL_561]] : DMA, %[[VAL_562]] : DMA)
// CHECK:           AIE.wire(%[[VAL_478]] : North, %[[VAL_562]] : South)
// CHECK:           AIE.wire(%[[VAL_495]] : East, %[[VAL_497]] : West)
// CHECK:           AIE.wire(%[[VAL_496]] : Core, %[[VAL_497]] : Core)
// CHECK:           AIE.wire(%[[VAL_496]] : DMA, %[[VAL_497]] : DMA)
// CHECK:           AIE.wire(%[[VAL_401]] : East, %[[VAL_403]] : West)
// CHECK:           AIE.wire(%[[VAL_402]] : Core, %[[VAL_403]] : Core)
// CHECK:           AIE.wire(%[[VAL_402]] : DMA, %[[VAL_403]] : DMA)
// CHECK:           AIE.wire(%[[VAL_497]] : North, %[[VAL_403]] : South)
// CHECK:           AIE.wire(%[[VAL_478]] : East, %[[VAL_480]] : West)
// CHECK:           AIE.wire(%[[VAL_479]] : Core, %[[VAL_480]] : Core)
// CHECK:           AIE.wire(%[[VAL_479]] : DMA, %[[VAL_480]] : DMA)
// CHECK:           AIE.wire(%[[VAL_562]] : East, %[[VAL_564]] : West)
// CHECK:           AIE.wire(%[[VAL_563]] : Core, %[[VAL_564]] : Core)
// CHECK:           AIE.wire(%[[VAL_563]] : DMA, %[[VAL_564]] : DMA)
// CHECK:           AIE.wire(%[[VAL_480]] : North, %[[VAL_564]] : South)
// CHECK:           AIE.wire(%[[VAL_375]] : North, %[[VAL_374]] : South)
// CHECK:           AIE.wire(%[[VAL_140]] : DMA, %[[VAL_375]] : DMA)
// CHECK:           AIE.wire(%[[VAL_497]] : East, %[[VAL_376]] : West)
// CHECK:           AIE.wire(%[[VAL_139]] : Core, %[[VAL_376]] : Core)
// CHECK:           AIE.wire(%[[VAL_139]] : DMA, %[[VAL_376]] : DMA)
// CHECK:           AIE.wire(%[[VAL_374]] : North, %[[VAL_376]] : South)
// CHECK:           AIE.wire(%[[VAL_403]] : East, %[[VAL_404]] : West)
// CHECK:           AIE.wire(%[[VAL_138]] : Core, %[[VAL_404]] : Core)
// CHECK:           AIE.wire(%[[VAL_138]] : DMA, %[[VAL_404]] : DMA)
// CHECK:           AIE.wire(%[[VAL_376]] : North, %[[VAL_404]] : South)
// CHECK:           AIE.wire(%[[VAL_447]] : Core, %[[VAL_448]] : Core)
// CHECK:           AIE.wire(%[[VAL_447]] : DMA, %[[VAL_448]] : DMA)
// CHECK:           AIE.wire(%[[VAL_404]] : North, %[[VAL_448]] : South)
// CHECK:           AIE.wire(%[[VAL_480]] : East, %[[VAL_482]] : West)
// CHECK:           AIE.wire(%[[VAL_481]] : Core, %[[VAL_482]] : Core)
// CHECK:           AIE.wire(%[[VAL_481]] : DMA, %[[VAL_482]] : DMA)
// CHECK:           AIE.wire(%[[VAL_448]] : North, %[[VAL_482]] : South)
// CHECK:           AIE.wire(%[[VAL_564]] : East, %[[VAL_566]] : West)
// CHECK:           AIE.wire(%[[VAL_565]] : Core, %[[VAL_566]] : Core)
// CHECK:           AIE.wire(%[[VAL_565]] : DMA, %[[VAL_566]] : DMA)
// CHECK:           AIE.wire(%[[VAL_482]] : North, %[[VAL_566]] : South)
// CHECK:           AIE.wire(%[[VAL_374]] : East, %[[VAL_407]] : West)
// CHECK:           AIE.wire(%[[VAL_408]] : North, %[[VAL_407]] : South)
// CHECK:           AIE.wire(%[[VAL_120]] : DMA, %[[VAL_408]] : DMA)
// CHECK:           AIE.wire(%[[VAL_376]] : East, %[[VAL_409]] : West)
// CHECK:           AIE.wire(%[[VAL_119]] : Core, %[[VAL_409]] : Core)
// CHECK:           AIE.wire(%[[VAL_119]] : DMA, %[[VAL_409]] : DMA)
// CHECK:           AIE.wire(%[[VAL_407]] : North, %[[VAL_409]] : South)
// CHECK:           AIE.wire(%[[VAL_404]] : East, %[[VAL_410]] : West)
// CHECK:           AIE.wire(%[[VAL_118]] : Core, %[[VAL_410]] : Core)
// CHECK:           AIE.wire(%[[VAL_118]] : DMA, %[[VAL_410]] : DMA)
// CHECK:           AIE.wire(%[[VAL_409]] : North, %[[VAL_410]] : South)
// CHECK:           AIE.wire(%[[VAL_448]] : East, %[[VAL_620]] : West)
// CHECK:           AIE.wire(%[[VAL_619]] : Core, %[[VAL_620]] : Core)
// CHECK:           AIE.wire(%[[VAL_619]] : DMA, %[[VAL_620]] : DMA)
// CHECK:           AIE.wire(%[[VAL_410]] : North, %[[VAL_620]] : South)
// CHECK:           AIE.wire(%[[VAL_482]] : East, %[[VAL_622]] : West)
// CHECK:           AIE.wire(%[[VAL_621]] : Core, %[[VAL_622]] : Core)
// CHECK:           AIE.wire(%[[VAL_621]] : DMA, %[[VAL_622]] : DMA)
// CHECK:           AIE.wire(%[[VAL_620]] : North, %[[VAL_622]] : South)
// CHECK:           AIE.wire(%[[VAL_566]] : East, %[[VAL_568]] : West)
// CHECK:           AIE.wire(%[[VAL_567]] : Core, %[[VAL_568]] : Core)
// CHECK:           AIE.wire(%[[VAL_567]] : DMA, %[[VAL_568]] : DMA)
// CHECK:           AIE.wire(%[[VAL_622]] : North, %[[VAL_568]] : South)
// CHECK:           AIE.wire(%[[VAL_409]] : East, %[[VAL_499]] : West)
// CHECK:           AIE.wire(%[[VAL_498]] : Core, %[[VAL_499]] : Core)
// CHECK:           AIE.wire(%[[VAL_498]] : DMA, %[[VAL_499]] : DMA)
// CHECK:           AIE.wire(%[[VAL_410]] : East, %[[VAL_435]] : West)
// CHECK:           AIE.wire(%[[VAL_434]] : Core, %[[VAL_435]] : Core)
// CHECK:           AIE.wire(%[[VAL_434]] : DMA, %[[VAL_435]] : DMA)
// CHECK:           AIE.wire(%[[VAL_499]] : North, %[[VAL_435]] : South)
// CHECK:           AIE.wire(%[[VAL_568]] : East, %[[VAL_570]] : West)
// CHECK:           AIE.wire(%[[VAL_569]] : Core, %[[VAL_570]] : Core)
// CHECK:           AIE.wire(%[[VAL_569]] : DMA, %[[VAL_570]] : DMA)
// CHECK:           AIE.wire(%[[VAL_499]] : East, %[[VAL_501]] : West)
// CHECK:           AIE.wire(%[[VAL_500]] : Core, %[[VAL_501]] : Core)
// CHECK:           AIE.wire(%[[VAL_500]] : DMA, %[[VAL_501]] : DMA)
// CHECK:           AIE.wire(%[[VAL_435]] : East, %[[VAL_437]] : West)
// CHECK:           AIE.wire(%[[VAL_436]] : Core, %[[VAL_437]] : Core)
// CHECK:           AIE.wire(%[[VAL_436]] : DMA, %[[VAL_437]] : DMA)
// CHECK:           AIE.wire(%[[VAL_501]] : North, %[[VAL_437]] : South)
// CHECK:           AIE.wire(%[[VAL_570]] : East, %[[VAL_572]] : West)
// CHECK:           AIE.wire(%[[VAL_571]] : Core, %[[VAL_572]] : Core)
// CHECK:           AIE.wire(%[[VAL_571]] : DMA, %[[VAL_572]] : DMA)
// CHECK:           AIE.wire(%[[VAL_501]] : East, %[[VAL_503]] : West)
// CHECK:           AIE.wire(%[[VAL_502]] : Core, %[[VAL_503]] : Core)
// CHECK:           AIE.wire(%[[VAL_502]] : DMA, %[[VAL_503]] : DMA)
// CHECK:           AIE.wire(%[[VAL_437]] : East, %[[VAL_439]] : West)
// CHECK:           AIE.wire(%[[VAL_438]] : Core, %[[VAL_439]] : Core)
// CHECK:           AIE.wire(%[[VAL_438]] : DMA, %[[VAL_439]] : DMA)
// CHECK:           AIE.wire(%[[VAL_503]] : North, %[[VAL_439]] : South)
// CHECK:           AIE.wire(%[[VAL_572]] : East, %[[VAL_574]] : West)
// CHECK:           AIE.wire(%[[VAL_573]] : Core, %[[VAL_574]] : Core)
// CHECK:           AIE.wire(%[[VAL_573]] : DMA, %[[VAL_574]] : DMA)
// CHECK:           AIE.wire(%[[VAL_503]] : East, %[[VAL_505]] : West)
// CHECK:           AIE.wire(%[[VAL_504]] : Core, %[[VAL_505]] : Core)
// CHECK:           AIE.wire(%[[VAL_504]] : DMA, %[[VAL_505]] : DMA)
// CHECK:           AIE.wire(%[[VAL_439]] : East, %[[VAL_441]] : West)
// CHECK:           AIE.wire(%[[VAL_440]] : Core, %[[VAL_441]] : Core)
// CHECK:           AIE.wire(%[[VAL_440]] : DMA, %[[VAL_441]] : DMA)
// CHECK:           AIE.wire(%[[VAL_505]] : North, %[[VAL_441]] : South)
// CHECK:           AIE.wire(%[[VAL_574]] : East, %[[VAL_576]] : West)
// CHECK:           AIE.wire(%[[VAL_575]] : Core, %[[VAL_576]] : Core)
// CHECK:           AIE.wire(%[[VAL_575]] : DMA, %[[VAL_576]] : DMA)
// CHECK:           AIE.wire(%[[VAL_505]] : East, %[[VAL_507]] : West)
// CHECK:           AIE.wire(%[[VAL_506]] : Core, %[[VAL_507]] : Core)
// CHECK:           AIE.wire(%[[VAL_506]] : DMA, %[[VAL_507]] : DMA)
// CHECK:           AIE.wire(%[[VAL_441]] : East, %[[VAL_443]] : West)
// CHECK:           AIE.wire(%[[VAL_442]] : Core, %[[VAL_443]] : Core)
// CHECK:           AIE.wire(%[[VAL_442]] : DMA, %[[VAL_443]] : DMA)
// CHECK:           AIE.wire(%[[VAL_507]] : North, %[[VAL_443]] : South)
// CHECK:           AIE.wire(%[[VAL_576]] : East, %[[VAL_578]] : West)
// CHECK:           AIE.wire(%[[VAL_577]] : Core, %[[VAL_578]] : Core)
// CHECK:           AIE.wire(%[[VAL_577]] : DMA, %[[VAL_578]] : DMA)
// CHECK:           AIE.wire(%[[VAL_507]] : East, %[[VAL_509]] : West)
// CHECK:           AIE.wire(%[[VAL_508]] : Core, %[[VAL_509]] : Core)
// CHECK:           AIE.wire(%[[VAL_508]] : DMA, %[[VAL_509]] : DMA)
// CHECK:           AIE.wire(%[[VAL_443]] : East, %[[VAL_445]] : West)
// CHECK:           AIE.wire(%[[VAL_444]] : Core, %[[VAL_445]] : Core)
// CHECK:           AIE.wire(%[[VAL_444]] : DMA, %[[VAL_445]] : DMA)
// CHECK:           AIE.wire(%[[VAL_509]] : North, %[[VAL_445]] : South)
// CHECK:           AIE.wire(%[[VAL_578]] : East, %[[VAL_580]] : West)
// CHECK:           AIE.wire(%[[VAL_579]] : Core, %[[VAL_580]] : Core)
// CHECK:           AIE.wire(%[[VAL_579]] : DMA, %[[VAL_580]] : DMA)
// CHECK:           AIE.wire(%[[VAL_432]] : North, %[[VAL_431]] : South)
// CHECK:           AIE.wire(%[[VAL_101]] : DMA, %[[VAL_432]] : DMA)
// CHECK:           AIE.wire(%[[VAL_509]] : East, %[[VAL_433]] : West)
// CHECK:           AIE.wire(%[[VAL_100]] : Core, %[[VAL_433]] : Core)
// CHECK:           AIE.wire(%[[VAL_100]] : DMA, %[[VAL_433]] : DMA)
// CHECK:           AIE.wire(%[[VAL_431]] : North, %[[VAL_433]] : South)
// CHECK:           AIE.wire(%[[VAL_445]] : East, %[[VAL_446]] : West)
// CHECK:           AIE.wire(%[[VAL_99]] : Core, %[[VAL_446]] : Core)
// CHECK:           AIE.wire(%[[VAL_99]] : DMA, %[[VAL_446]] : DMA)
// CHECK:           AIE.wire(%[[VAL_433]] : North, %[[VAL_446]] : South)
// CHECK:           AIE.wire(%[[VAL_580]] : East, %[[VAL_582]] : West)
// CHECK:           AIE.wire(%[[VAL_581]] : Core, %[[VAL_582]] : Core)
// CHECK:           AIE.wire(%[[VAL_581]] : DMA, %[[VAL_582]] : DMA)
// CHECK:           AIE.wire(%[[VAL_431]] : East, %[[VAL_483]] : West)
// CHECK:           AIE.wire(%[[VAL_484]] : North, %[[VAL_483]] : South)
// CHECK:           AIE.wire(%[[VAL_82]] : DMA, %[[VAL_484]] : DMA)
// CHECK:           AIE.wire(%[[VAL_433]] : East, %[[VAL_510]] : West)
// CHECK:           AIE.wire(%[[VAL_81]] : Core, %[[VAL_510]] : Core)
// CHECK:           AIE.wire(%[[VAL_81]] : DMA, %[[VAL_510]] : DMA)
// CHECK:           AIE.wire(%[[VAL_483]] : North, %[[VAL_510]] : South)
// CHECK:           AIE.wire(%[[VAL_446]] : East, %[[VAL_616]] : West)
// CHECK:           AIE.wire(%[[VAL_80]] : Core, %[[VAL_616]] : Core)
// CHECK:           AIE.wire(%[[VAL_80]] : DMA, %[[VAL_616]] : DMA)
// CHECK:           AIE.wire(%[[VAL_510]] : North, %[[VAL_616]] : South)
// CHECK:           AIE.wire(%[[VAL_582]] : East, %[[VAL_584]] : West)
// CHECK:           AIE.wire(%[[VAL_583]] : Core, %[[VAL_584]] : Core)
// CHECK:           AIE.wire(%[[VAL_583]] : DMA, %[[VAL_584]] : DMA)
// CHECK:           AIE.wire(%[[VAL_510]] : East, %[[VAL_626]] : West)
// CHECK:           AIE.wire(%[[VAL_625]] : Core, %[[VAL_626]] : Core)
// CHECK:           AIE.wire(%[[VAL_625]] : DMA, %[[VAL_626]] : DMA)
// CHECK:           AIE.wire(%[[VAL_616]] : East, %[[VAL_515]] : West)
// CHECK:           AIE.wire(%[[VAL_514]] : Core, %[[VAL_515]] : Core)
// CHECK:           AIE.wire(%[[VAL_514]] : DMA, %[[VAL_515]] : DMA)
// CHECK:           AIE.wire(%[[VAL_626]] : North, %[[VAL_515]] : South)
// CHECK:           AIE.wire(%[[VAL_527]] : Core, %[[VAL_528]] : Core)
// CHECK:           AIE.wire(%[[VAL_527]] : DMA, %[[VAL_528]] : DMA)
// CHECK:           AIE.wire(%[[VAL_515]] : North, %[[VAL_528]] : South)
// CHECK:           AIE.wire(%[[VAL_529]] : Core, %[[VAL_530]] : Core)
// CHECK:           AIE.wire(%[[VAL_529]] : DMA, %[[VAL_530]] : DMA)
// CHECK:           AIE.wire(%[[VAL_528]] : North, %[[VAL_530]] : South)
// CHECK:           AIE.wire(%[[VAL_584]] : East, %[[VAL_586]] : West)
// CHECK:           AIE.wire(%[[VAL_585]] : Core, %[[VAL_586]] : Core)
// CHECK:           AIE.wire(%[[VAL_585]] : DMA, %[[VAL_586]] : DMA)
// CHECK:           AIE.wire(%[[VAL_530]] : North, %[[VAL_586]] : South)
// CHECK:           AIE.wire(%[[VAL_626]] : East, %[[VAL_628]] : West)
// CHECK:           AIE.wire(%[[VAL_627]] : Core, %[[VAL_628]] : Core)
// CHECK:           AIE.wire(%[[VAL_627]] : DMA, %[[VAL_628]] : DMA)
// CHECK:           AIE.wire(%[[VAL_515]] : East, %[[VAL_517]] : West)
// CHECK:           AIE.wire(%[[VAL_516]] : Core, %[[VAL_517]] : Core)
// CHECK:           AIE.wire(%[[VAL_516]] : DMA, %[[VAL_517]] : DMA)
// CHECK:           AIE.wire(%[[VAL_628]] : North, %[[VAL_517]] : South)
// CHECK:           AIE.wire(%[[VAL_586]] : East, %[[VAL_598]] : West)
// CHECK:           AIE.wire(%[[VAL_597]] : Core, %[[VAL_598]] : Core)
// CHECK:           AIE.wire(%[[VAL_597]] : DMA, %[[VAL_598]] : DMA)
// CHECK:           AIE.wire(%[[VAL_628]] : East, %[[VAL_630]] : West)
// CHECK:           AIE.wire(%[[VAL_629]] : Core, %[[VAL_630]] : Core)
// CHECK:           AIE.wire(%[[VAL_629]] : DMA, %[[VAL_630]] : DMA)
// CHECK:           AIE.wire(%[[VAL_517]] : East, %[[VAL_519]] : West)
// CHECK:           AIE.wire(%[[VAL_518]] : Core, %[[VAL_519]] : Core)
// CHECK:           AIE.wire(%[[VAL_518]] : DMA, %[[VAL_519]] : DMA)
// CHECK:           AIE.wire(%[[VAL_630]] : North, %[[VAL_519]] : South)
// CHECK:           AIE.wire(%[[VAL_598]] : East, %[[VAL_600]] : West)
// CHECK:           AIE.wire(%[[VAL_599]] : Core, %[[VAL_600]] : Core)
// CHECK:           AIE.wire(%[[VAL_599]] : DMA, %[[VAL_600]] : DMA)
// CHECK:           AIE.wire(%[[VAL_630]] : East, %[[VAL_632]] : West)
// CHECK:           AIE.wire(%[[VAL_631]] : Core, %[[VAL_632]] : Core)
// CHECK:           AIE.wire(%[[VAL_631]] : DMA, %[[VAL_632]] : DMA)
// CHECK:           AIE.wire(%[[VAL_519]] : East, %[[VAL_521]] : West)
// CHECK:           AIE.wire(%[[VAL_520]] : Core, %[[VAL_521]] : Core)
// CHECK:           AIE.wire(%[[VAL_520]] : DMA, %[[VAL_521]] : DMA)
// CHECK:           AIE.wire(%[[VAL_632]] : North, %[[VAL_521]] : South)
// CHECK:           AIE.wire(%[[VAL_600]] : East, %[[VAL_602]] : West)
// CHECK:           AIE.wire(%[[VAL_601]] : Core, %[[VAL_602]] : Core)
// CHECK:           AIE.wire(%[[VAL_601]] : DMA, %[[VAL_602]] : DMA)
// CHECK:           AIE.wire(%[[VAL_632]] : East, %[[VAL_634]] : West)
// CHECK:           AIE.wire(%[[VAL_633]] : Core, %[[VAL_634]] : Core)
// CHECK:           AIE.wire(%[[VAL_633]] : DMA, %[[VAL_634]] : DMA)
// CHECK:           AIE.wire(%[[VAL_521]] : East, %[[VAL_523]] : West)
// CHECK:           AIE.wire(%[[VAL_522]] : Core, %[[VAL_523]] : Core)
// CHECK:           AIE.wire(%[[VAL_522]] : DMA, %[[VAL_523]] : DMA)
// CHECK:           AIE.wire(%[[VAL_634]] : North, %[[VAL_523]] : South)
// CHECK:           AIE.wire(%[[VAL_602]] : East, %[[VAL_604]] : West)
// CHECK:           AIE.wire(%[[VAL_603]] : Core, %[[VAL_604]] : Core)
// CHECK:           AIE.wire(%[[VAL_603]] : DMA, %[[VAL_604]] : DMA)
// CHECK:           AIE.wire(%[[VAL_634]] : East, %[[VAL_636]] : West)
// CHECK:           AIE.wire(%[[VAL_635]] : Core, %[[VAL_636]] : Core)
// CHECK:           AIE.wire(%[[VAL_635]] : DMA, %[[VAL_636]] : DMA)
// CHECK:           AIE.wire(%[[VAL_523]] : East, %[[VAL_525]] : West)
// CHECK:           AIE.wire(%[[VAL_524]] : Core, %[[VAL_525]] : Core)
// CHECK:           AIE.wire(%[[VAL_524]] : DMA, %[[VAL_525]] : DMA)
// CHECK:           AIE.wire(%[[VAL_636]] : North, %[[VAL_525]] : South)
// CHECK:           AIE.wire(%[[VAL_604]] : East, %[[VAL_606]] : West)
// CHECK:           AIE.wire(%[[VAL_605]] : Core, %[[VAL_606]] : Core)
// CHECK:           AIE.wire(%[[VAL_605]] : DMA, %[[VAL_606]] : DMA)
// CHECK:           AIE.wire(%[[VAL_512]] : North, %[[VAL_511]] : South)
// CHECK:           AIE.wire(%[[VAL_62]] : DMA, %[[VAL_512]] : DMA)
// CHECK:           AIE.wire(%[[VAL_636]] : East, %[[VAL_513]] : West)
// CHECK:           AIE.wire(%[[VAL_61]] : Core, %[[VAL_513]] : Core)
// CHECK:           AIE.wire(%[[VAL_61]] : DMA, %[[VAL_513]] : DMA)
// CHECK:           AIE.wire(%[[VAL_511]] : North, %[[VAL_513]] : South)
// CHECK:           AIE.wire(%[[VAL_525]] : East, %[[VAL_526]] : West)
// CHECK:           AIE.wire(%[[VAL_60]] : Core, %[[VAL_526]] : Core)
// CHECK:           AIE.wire(%[[VAL_60]] : DMA, %[[VAL_526]] : DMA)
// CHECK:           AIE.wire(%[[VAL_513]] : North, %[[VAL_526]] : South)
// CHECK:           AIE.wire(%[[VAL_593]] : Core, %[[VAL_594]] : Core)
// CHECK:           AIE.wire(%[[VAL_593]] : DMA, %[[VAL_594]] : DMA)
// CHECK:           AIE.wire(%[[VAL_606]] : East, %[[VAL_608]] : West)
// CHECK:           AIE.wire(%[[VAL_607]] : Core, %[[VAL_608]] : Core)
// CHECK:           AIE.wire(%[[VAL_607]] : DMA, %[[VAL_608]] : DMA)
// CHECK:           AIE.wire(%[[VAL_594]] : North, %[[VAL_608]] : South)
// CHECK:           AIE.wire(%[[VAL_511]] : East, %[[VAL_587]] : West)
// CHECK:           AIE.wire(%[[VAL_588]] : North, %[[VAL_587]] : South)
// CHECK:           AIE.wire(%[[VAL_42]] : DMA, %[[VAL_588]] : DMA)
// CHECK:           AIE.wire(%[[VAL_513]] : East, %[[VAL_589]] : West)
// CHECK:           AIE.wire(%[[VAL_41]] : Core, %[[VAL_589]] : Core)
// CHECK:           AIE.wire(%[[VAL_41]] : DMA, %[[VAL_589]] : DMA)
// CHECK:           AIE.wire(%[[VAL_587]] : North, %[[VAL_589]] : South)
// CHECK:           AIE.wire(%[[VAL_526]] : East, %[[VAL_590]] : West)
// CHECK:           AIE.wire(%[[VAL_40]] : Core, %[[VAL_590]] : Core)
// CHECK:           AIE.wire(%[[VAL_40]] : DMA, %[[VAL_590]] : DMA)
// CHECK:           AIE.wire(%[[VAL_589]] : North, %[[VAL_590]] : South)
// CHECK:           AIE.wire(%[[VAL_591]] : Core, %[[VAL_592]] : Core)
// CHECK:           AIE.wire(%[[VAL_591]] : DMA, %[[VAL_592]] : DMA)
// CHECK:           AIE.wire(%[[VAL_590]] : North, %[[VAL_592]] : South)
// CHECK:           AIE.wire(%[[VAL_594]] : East, %[[VAL_596]] : West)
// CHECK:           AIE.wire(%[[VAL_595]] : Core, %[[VAL_596]] : Core)
// CHECK:           AIE.wire(%[[VAL_595]] : DMA, %[[VAL_596]] : DMA)
// CHECK:           AIE.wire(%[[VAL_592]] : North, %[[VAL_596]] : South)
// CHECK:           AIE.wire(%[[VAL_589]] : East, %[[VAL_612]] : West)
// CHECK:           AIE.wire(%[[VAL_611]] : Core, %[[VAL_612]] : Core)
// CHECK:           AIE.wire(%[[VAL_611]] : DMA, %[[VAL_612]] : DMA)
// CHECK:           AIE.wire(%[[VAL_590]] : East, %[[VAL_618]] : West)
// CHECK:           AIE.wire(%[[VAL_617]] : Core, %[[VAL_618]] : Core)
// CHECK:           AIE.wire(%[[VAL_617]] : DMA, %[[VAL_618]] : DMA)
// CHECK:           AIE.wire(%[[VAL_612]] : North, %[[VAL_618]] : South)
// CHECK:           AIE.wire(%[[VAL_612]] : East, %[[VAL_614]] : West)
// CHECK:           AIE.wire(%[[VAL_613]] : Core, %[[VAL_614]] : Core)
// CHECK:           AIE.wire(%[[VAL_613]] : DMA, %[[VAL_614]] : DMA)
// CHECK:           AIE.wire(%[[VAL_610]] : North, %[[VAL_609]] : South)
// CHECK:           AIE.wire(%[[VAL_22]] : DMA, %[[VAL_610]] : DMA)
// CHECK:           AIE.wire(%[[VAL_614]] : East, %[[VAL_615]] : West)
// CHECK:           AIE.wire(%[[VAL_21]] : Core, %[[VAL_615]] : Core)
// CHECK:           AIE.wire(%[[VAL_21]] : DMA, %[[VAL_615]] : DMA)
// CHECK:           AIE.wire(%[[VAL_609]] : North, %[[VAL_615]] : South)
// CHECK:           AIE.wire(%[[VAL_609]] : East, %[[VAL_623]] : West)
// CHECK:           AIE.wire(%[[VAL_624]] : North, %[[VAL_623]] : South)
// CHECK:           AIE.wire(%[[VAL_2]] : DMA, %[[VAL_624]] : DMA)
// CHECK:           AIE.wire(%[[VAL_615]] : East, %[[VAL_637]] : West)
// CHECK:           AIE.wire(%[[VAL_1]] : Core, %[[VAL_637]] : Core)
// CHECK:           AIE.wire(%[[VAL_1]] : DMA, %[[VAL_637]] : DMA)
// CHECK:           AIE.wire(%[[VAL_623]] : North, %[[VAL_637]] : South)
// CHECK:         }


//

module @vecmul_4x4  {
  AIE.device(xcvc1902) {
    %0 = AIE.tile(47, 2)
    %1 = AIE.tile(47, 1)
    %2 = AIE.tile(47, 0)
    %3 = AIE.tile(3, 3)
    %4 = AIE.tile(10, 5)
    %5 = AIE.lock(%4, 2)
    %6 = AIE.buffer(%4) {sym_name = "buf47"} : memref<64xi32, 2>
    %7 = AIE.lock(%4, 1)
    %8 = AIE.buffer(%4) {sym_name = "buf46"} : memref<64xi32, 2>
    %9 = AIE.lock(%4, 0)
    %10 = AIE.buffer(%4) {sym_name = "buf45"} : memref<64xi32, 2>
    %11 = AIE.mem(%4)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%9, Acquire, 0)
      AIE.dmaBd(<%10 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%9, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%7, Acquire, 0)
      AIE.dmaBd(<%8 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%7, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%5, Acquire, 1)
      AIE.dmaBd(<%6 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%5, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %12 = AIE.core(%4)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%9, Acquire, 1)
      AIE.useLock(%7, Acquire, 1)
      AIE.useLock(%5, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %10[%arg0] : memref<64xi32, 2>
        %201 = affine.load %8[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %6[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%5, Release, 1)
      AIE.useLock(%7, Release, 0)
      AIE.useLock(%9, Release, 0)
      cf.br ^bb1
    }
    %13 = AIE.tile(46, 2)
    %14 = AIE.tile(46, 1)
    %15 = AIE.tile(46, 0)
    %16 = AIE.tile(2, 3)
    %17 = AIE.tile(9, 5)
    %18 = AIE.lock(%17, 2)
    %19 = AIE.buffer(%17) {sym_name = "buf44"} : memref<64xi32, 2>
    %20 = AIE.lock(%17, 1)
    %21 = AIE.buffer(%17) {sym_name = "buf43"} : memref<64xi32, 2>
    %22 = AIE.lock(%17, 0)
    %23 = AIE.buffer(%17) {sym_name = "buf42"} : memref<64xi32, 2>
    %24 = AIE.mem(%17)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%22, Acquire, 0)
      AIE.dmaBd(<%23 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%22, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%20, Acquire, 0)
      AIE.dmaBd(<%21 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%20, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%18, Acquire, 1)
      AIE.dmaBd(<%19 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%18, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %25 = AIE.core(%17)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%22, Acquire, 1)
      AIE.useLock(%20, Acquire, 1)
      AIE.useLock(%18, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %23[%arg0] : memref<64xi32, 2>
        %201 = affine.load %21[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %19[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%18, Release, 1)
      AIE.useLock(%20, Release, 0)
      AIE.useLock(%22, Release, 0)
      cf.br ^bb1
    }
    %26 = AIE.tile(43, 2)
    %27 = AIE.tile(43, 1)
    %28 = AIE.tile(43, 0)
    %29 = AIE.tile(1, 3)
    %30 = AIE.tile(8, 5)
    %31 = AIE.lock(%30, 2)
    %32 = AIE.buffer(%30) {sym_name = "buf41"} : memref<64xi32, 2>
    %33 = AIE.lock(%30, 1)
    %34 = AIE.buffer(%30) {sym_name = "buf40"} : memref<64xi32, 2>
    %35 = AIE.lock(%30, 0)
    %36 = AIE.buffer(%30) {sym_name = "buf39"} : memref<64xi32, 2>
    %37 = AIE.mem(%30)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%35, Acquire, 0)
      AIE.dmaBd(<%36 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%35, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%33, Acquire, 0)
      AIE.dmaBd(<%34 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%33, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%31, Acquire, 1)
      AIE.dmaBd(<%32 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%31, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %38 = AIE.core(%30)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%35, Acquire, 1)
      AIE.useLock(%33, Acquire, 1)
      AIE.useLock(%31, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %36[%arg0] : memref<64xi32, 2>
        %201 = affine.load %34[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %32[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%31, Release, 1)
      AIE.useLock(%33, Release, 0)
      AIE.useLock(%35, Release, 0)
      cf.br ^bb1
    }
    %39 = AIE.tile(42, 2)
    %40 = AIE.tile(42, 1)
    %41 = AIE.tile(42, 0)
    %42 = AIE.tile(0, 3)
    %43 = AIE.tile(7, 5)
    %44 = AIE.lock(%43, 2)
    %45 = AIE.buffer(%43) {sym_name = "buf38"} : memref<64xi32, 2>
    %46 = AIE.lock(%43, 1)
    %47 = AIE.buffer(%43) {sym_name = "buf37"} : memref<64xi32, 2>
    %48 = AIE.lock(%43, 0)
    %49 = AIE.buffer(%43) {sym_name = "buf36"} : memref<64xi32, 2>
    %50 = AIE.mem(%43)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%48, Acquire, 0)
      AIE.dmaBd(<%49 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%48, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%46, Acquire, 0)
      AIE.dmaBd(<%47 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%46, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%44, Acquire, 1)
      AIE.dmaBd(<%45 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%44, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %51 = AIE.core(%43)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%48, Acquire, 1)
      AIE.useLock(%46, Acquire, 1)
      AIE.useLock(%44, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %49[%arg0] : memref<64xi32, 2>
        %201 = affine.load %47[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %45[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%44, Release, 1)
      AIE.useLock(%46, Release, 0)
      AIE.useLock(%48, Release, 0)
      cf.br ^bb1
    }
    %52 = AIE.tile(35, 2)
    %53 = AIE.tile(35, 1)
    %54 = AIE.tile(35, 0)
    %55 = AIE.tile(10, 4)
    %56 = AIE.lock(%55, 2)
    %57 = AIE.buffer(%55) {sym_name = "buf35"} : memref<64xi32, 2>
    %58 = AIE.lock(%55, 1)
    %59 = AIE.buffer(%55) {sym_name = "buf34"} : memref<64xi32, 2>
    %60 = AIE.lock(%55, 0)
    %61 = AIE.buffer(%55) {sym_name = "buf33"} : memref<64xi32, 2>
    %62 = AIE.mem(%55)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%60, Acquire, 0)
      AIE.dmaBd(<%61 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%60, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%58, Acquire, 0)
      AIE.dmaBd(<%59 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%58, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%56, Acquire, 1)
      AIE.dmaBd(<%57 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%56, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %63 = AIE.core(%55)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%60, Acquire, 1)
      AIE.useLock(%58, Acquire, 1)
      AIE.useLock(%56, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %61[%arg0] : memref<64xi32, 2>
        %201 = affine.load %59[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %57[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%56, Release, 1)
      AIE.useLock(%58, Release, 0)
      AIE.useLock(%60, Release, 0)
      cf.br ^bb1
    }
    %64 = AIE.tile(34, 2)
    %65 = AIE.tile(34, 1)
    %66 = AIE.tile(34, 0)
    %67 = AIE.tile(9, 4)
    %68 = AIE.lock(%67, 2)
    %69 = AIE.buffer(%67) {sym_name = "buf32"} : memref<64xi32, 2>
    %70 = AIE.lock(%67, 1)
    %71 = AIE.buffer(%67) {sym_name = "buf31"} : memref<64xi32, 2>
    %72 = AIE.lock(%67, 0)
    %73 = AIE.buffer(%67) {sym_name = "buf30"} : memref<64xi32, 2>
    %74 = AIE.mem(%67)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%72, Acquire, 0)
      AIE.dmaBd(<%73 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%72, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%70, Acquire, 0)
      AIE.dmaBd(<%71 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%70, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%68, Acquire, 1)
      AIE.dmaBd(<%69 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%68, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %75 = AIE.core(%67)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%72, Acquire, 1)
      AIE.useLock(%70, Acquire, 1)
      AIE.useLock(%68, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %73[%arg0] : memref<64xi32, 2>
        %201 = affine.load %71[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %69[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%68, Release, 1)
      AIE.useLock(%70, Release, 0)
      AIE.useLock(%72, Release, 0)
      cf.br ^bb1
    }
    %76 = AIE.tile(27, 2)
    %77 = AIE.tile(27, 1)
    %78 = AIE.tile(27, 0)
    %79 = AIE.tile(1, 2)
    %80 = AIE.tile(8, 4)
    %81 = AIE.lock(%80, 2)
    %82 = AIE.buffer(%80) {sym_name = "buf29"} : memref<64xi32, 2>
    %83 = AIE.lock(%80, 1)
    %84 = AIE.buffer(%80) {sym_name = "buf28"} : memref<64xi32, 2>
    %85 = AIE.lock(%80, 0)
    %86 = AIE.buffer(%80) {sym_name = "buf27"} : memref<64xi32, 2>
    %87 = AIE.mem(%80)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%85, Acquire, 0)
      AIE.dmaBd(<%86 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%85, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%83, Acquire, 0)
      AIE.dmaBd(<%84 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%83, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%81, Acquire, 1)
      AIE.dmaBd(<%82 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%81, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %88 = AIE.core(%80)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%85, Acquire, 1)
      AIE.useLock(%83, Acquire, 1)
      AIE.useLock(%81, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %86[%arg0] : memref<64xi32, 2>
        %201 = affine.load %84[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %82[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%81, Release, 1)
      AIE.useLock(%83, Release, 0)
      AIE.useLock(%85, Release, 0)
      cf.br ^bb1
    }
    %89 = AIE.tile(26, 2)
    %90 = AIE.tile(26, 1)
    %91 = AIE.tile(26, 0)
    %92 = AIE.tile(0, 2)
    %93 = AIE.tile(7, 4)
    %94 = AIE.lock(%93, 2)
    %95 = AIE.buffer(%93) {sym_name = "buf26"} : memref<64xi32, 2>
    %96 = AIE.lock(%93, 1)
    %97 = AIE.buffer(%93) {sym_name = "buf25"} : memref<64xi32, 2>
    %98 = AIE.lock(%93, 0)
    %99 = AIE.buffer(%93) {sym_name = "buf24"} : memref<64xi32, 2>
    %100 = AIE.mem(%93)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%98, Acquire, 0)
      AIE.dmaBd(<%99 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%98, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%96, Acquire, 0)
      AIE.dmaBd(<%97 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%96, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%94, Acquire, 1)
      AIE.dmaBd(<%95 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%94, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %101 = AIE.core(%93)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%98, Acquire, 1)
      AIE.useLock(%96, Acquire, 1)
      AIE.useLock(%94, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %99[%arg0] : memref<64xi32, 2>
        %201 = affine.load %97[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %95[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%94, Release, 1)
      AIE.useLock(%96, Release, 0)
      AIE.useLock(%98, Release, 0)
      cf.br ^bb1
    }
    %102 = AIE.tile(19, 2)
    %103 = AIE.tile(19, 1)
    %104 = AIE.tile(19, 0)
    %105 = AIE.tile(10, 3)
    %106 = AIE.lock(%105, 2)
    %107 = AIE.buffer(%105) {sym_name = "buf23"} : memref<64xi32, 2>
    %108 = AIE.lock(%105, 1)
    %109 = AIE.buffer(%105) {sym_name = "buf22"} : memref<64xi32, 2>
    %110 = AIE.lock(%105, 0)
    %111 = AIE.buffer(%105) {sym_name = "buf21"} : memref<64xi32, 2>
    %112 = AIE.mem(%105)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%110, Acquire, 0)
      AIE.dmaBd(<%111 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%110, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%108, Acquire, 0)
      AIE.dmaBd(<%109 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%108, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%106, Acquire, 1)
      AIE.dmaBd(<%107 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%106, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %113 = AIE.core(%105)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%110, Acquire, 1)
      AIE.useLock(%108, Acquire, 1)
      AIE.useLock(%106, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %111[%arg0] : memref<64xi32, 2>
        %201 = affine.load %109[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %107[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%106, Release, 1)
      AIE.useLock(%108, Release, 0)
      AIE.useLock(%110, Release, 0)
      cf.br ^bb1
    }
    %114 = AIE.tile(18, 2)
    %115 = AIE.tile(18, 1)
    %116 = AIE.tile(18, 0)
    %117 = AIE.tile(9, 3)
    %118 = AIE.lock(%117, 2)
    %119 = AIE.buffer(%117) {sym_name = "buf20"} : memref<64xi32, 2>
    %120 = AIE.lock(%117, 1)
    %121 = AIE.buffer(%117) {sym_name = "buf19"} : memref<64xi32, 2>
    %122 = AIE.lock(%117, 0)
    %123 = AIE.buffer(%117) {sym_name = "buf18"} : memref<64xi32, 2>
    %124 = AIE.mem(%117)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%122, Acquire, 0)
      AIE.dmaBd(<%123 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%122, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%120, Acquire, 0)
      AIE.dmaBd(<%121 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%120, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%118, Acquire, 1)
      AIE.dmaBd(<%119 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%118, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %125 = AIE.core(%117)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%122, Acquire, 1)
      AIE.useLock(%120, Acquire, 1)
      AIE.useLock(%118, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %123[%arg0] : memref<64xi32, 2>
        %201 = affine.load %121[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %119[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%118, Release, 1)
      AIE.useLock(%120, Release, 0)
      AIE.useLock(%122, Release, 0)
      cf.br ^bb1
    }
    %126 = AIE.tile(11, 2)
    %127 = AIE.tile(11, 1)
    %128 = AIE.tile(11, 0)
    %129 = AIE.tile(1, 1)
    %130 = AIE.tile(8, 3)
    %131 = AIE.lock(%130, 2)
    %132 = AIE.buffer(%130) {sym_name = "buf17"} : memref<64xi32, 2>
    %133 = AIE.lock(%130, 1)
    %134 = AIE.buffer(%130) {sym_name = "buf16"} : memref<64xi32, 2>
    %135 = AIE.lock(%130, 0)
    %136 = AIE.buffer(%130) {sym_name = "buf15"} : memref<64xi32, 2>
    %137 = AIE.mem(%130)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%135, Acquire, 0)
      AIE.dmaBd(<%136 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%135, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%133, Acquire, 0)
      AIE.dmaBd(<%134 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%133, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%131, Acquire, 1)
      AIE.dmaBd(<%132 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%131, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %138 = AIE.core(%130)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%135, Acquire, 1)
      AIE.useLock(%133, Acquire, 1)
      AIE.useLock(%131, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %136[%arg0] : memref<64xi32, 2>
        %201 = affine.load %134[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %132[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%131, Release, 1)
      AIE.useLock(%133, Release, 0)
      AIE.useLock(%135, Release, 0)
      cf.br ^bb1
    }
    %139 = AIE.tile(10, 1)
    %140 = AIE.tile(10, 0)
    %141 = AIE.tile(0, 1)
    %142 = AIE.tile(7, 3)
    %143 = AIE.lock(%142, 2)
    %144 = AIE.buffer(%142) {sym_name = "buf14"} : memref<64xi32, 2>
    %145 = AIE.lock(%142, 1)
    %146 = AIE.buffer(%142) {sym_name = "buf13"} : memref<64xi32, 2>
    %147 = AIE.lock(%142, 0)
    %148 = AIE.buffer(%142) {sym_name = "buf12"} : memref<64xi32, 2>
    %149 = AIE.mem(%142)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%147, Acquire, 0)
      AIE.dmaBd(<%148 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%147, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%145, Acquire, 0)
      AIE.dmaBd(<%146 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%145, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%143, Acquire, 1)
      AIE.dmaBd(<%144 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%143, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %150 = AIE.core(%142)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%147, Acquire, 1)
      AIE.useLock(%145, Acquire, 1)
      AIE.useLock(%143, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %148[%arg0] : memref<64xi32, 2>
        %201 = affine.load %146[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %144[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%143, Release, 1)
      AIE.useLock(%145, Release, 0)
      AIE.useLock(%147, Release, 0)
      cf.br ^bb1
    }
    %151 = AIE.tile(7, 1)
    %152 = AIE.tile(7, 0)
    %153 = AIE.tile(10, 2)
    %154 = AIE.lock(%153, 2)
    %155 = AIE.buffer(%153) {sym_name = "buf11"} : memref<64xi32, 2>
    %156 = AIE.lock(%153, 1)
    %157 = AIE.buffer(%153) {sym_name = "buf10"} : memref<64xi32, 2>
    %158 = AIE.lock(%153, 0)
    %159 = AIE.buffer(%153) {sym_name = "buf9"} : memref<64xi32, 2>
    %160 = AIE.mem(%153)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%158, Acquire, 0)
      AIE.dmaBd(<%159 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%158, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%156, Acquire, 0)
      AIE.dmaBd(<%157 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%156, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%154, Acquire, 1)
      AIE.dmaBd(<%155 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%154, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %161 = AIE.core(%153)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%158, Acquire, 1)
      AIE.useLock(%156, Acquire, 1)
      AIE.useLock(%154, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %159[%arg0] : memref<64xi32, 2>
        %201 = affine.load %157[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %155[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%154, Release, 1)
      AIE.useLock(%156, Release, 0)
      AIE.useLock(%158, Release, 0)
      cf.br ^bb1
    }
    %162 = AIE.tile(6, 2)
    %163 = AIE.tile(6, 1)
    %164 = AIE.tile(6, 0)
    %165 = AIE.tile(9, 2)
    %166 = AIE.lock(%165, 2)
    %167 = AIE.buffer(%165) {sym_name = "buf8"} : memref<64xi32, 2>
    %168 = AIE.lock(%165, 1)
    %169 = AIE.buffer(%165) {sym_name = "buf7"} : memref<64xi32, 2>
    %170 = AIE.lock(%165, 0)
    %171 = AIE.buffer(%165) {sym_name = "buf6"} : memref<64xi32, 2>
    %172 = AIE.mem(%165)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%170, Acquire, 0)
      AIE.dmaBd(<%171 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%170, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%168, Acquire, 0)
      AIE.dmaBd(<%169 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%168, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%166, Acquire, 1)
      AIE.dmaBd(<%167 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%166, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %173 = AIE.core(%165)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%170, Acquire, 1)
      AIE.useLock(%168, Acquire, 1)
      AIE.useLock(%166, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %171[%arg0] : memref<64xi32, 2>
        %201 = affine.load %169[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %167[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%166, Release, 1)
      AIE.useLock(%168, Release, 0)
      AIE.useLock(%170, Release, 0)
      cf.br ^bb1
    }
    %174 = AIE.tile(3, 2)
    %175 = AIE.tile(3, 1)
    %176 = AIE.tile(3, 0)
    %177 = AIE.tile(1, 0)
    %178 = AIE.tile(8, 2)
    %179 = AIE.lock(%178, 2)
    %180 = AIE.buffer(%178) {sym_name = "buf5"} : memref<64xi32, 2>
    %181 = AIE.lock(%178, 1)
    %182 = AIE.buffer(%178) {sym_name = "buf4"} : memref<64xi32, 2>
    %183 = AIE.lock(%178, 0)
    %184 = AIE.buffer(%178) {sym_name = "buf3"} : memref<64xi32, 2>
    %185 = AIE.mem(%178)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%183, Acquire, 0)
      AIE.dmaBd(<%184 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%183, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%181, Acquire, 0)
      AIE.dmaBd(<%182 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%181, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%179, Acquire, 1)
      AIE.dmaBd(<%180 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%179, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %186 = AIE.core(%178)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%183, Acquire, 1)
      AIE.useLock(%181, Acquire, 1)
      AIE.useLock(%179, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %184[%arg0] : memref<64xi32, 2>
        %201 = affine.load %182[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %180[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%179, Release, 1)
      AIE.useLock(%181, Release, 0)
      AIE.useLock(%183, Release, 0)
      cf.br ^bb1
    }
    %187 = AIE.tile(2, 2)
    %188 = AIE.tile(2, 1)
    %189 = AIE.tile(2, 0)
    %190 = AIE.tile(0, 0)
    %191 = AIE.tile(7, 2)
    %192 = AIE.lock(%191, 2)
    %193 = AIE.buffer(%191) {sym_name = "buf2"} : memref<64xi32, 2>
    %194 = AIE.lock(%191, 1)
    %195 = AIE.buffer(%191) {sym_name = "buf1"} : memref<64xi32, 2>
    %196 = AIE.lock(%191, 0)
    %197 = AIE.buffer(%191) {sym_name = "buf0"} : memref<64xi32, 2>
    %198 = AIE.mem(%191)  {
      %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      AIE.useLock(%196, Acquire, 0)
      AIE.dmaBd(<%197 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%196, Release, 1)
      AIE.nextBd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      AIE.useLock(%194, Acquire, 0)
      AIE.dmaBd(<%195 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%194, Release, 1)
      AIE.nextBd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      AIE.useLock(%192, Acquire, 1)
      AIE.dmaBd(<%193 : memref<64xi32, 2>, 0, 64>, 0)
      AIE.useLock(%192, Release, 0)
      AIE.nextBd ^bb5
    ^bb6:  // pred: ^bb2
      AIE.end
    }
    %199 = AIE.core(%191)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%196, Acquire, 1)
      AIE.useLock(%194, Acquire, 1)
      AIE.useLock(%192, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %197[%arg0] : memref<64xi32, 2>
        %201 = affine.load %195[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %193[%arg0] : memref<64xi32, 2>
      }
      AIE.useLock(%192, Release, 1)
      AIE.useLock(%194, Release, 0)
      AIE.useLock(%196, Release, 0)
      cf.br ^bb1
    }
    AIE.flow(%189, DMA : 0, %191, DMA : 0)
    AIE.flow(%189, DMA : 1, %191, DMA : 1)
    AIE.flow(%191, DMA : 0, %189, DMA : 0)
    AIE.flow(%176, DMA : 0, %178, DMA : 0)
    AIE.flow(%176, DMA : 1, %178, DMA : 1)
    AIE.flow(%178, DMA : 0, %189, DMA : 1)
    AIE.flow(%164, DMA : 0, %165, DMA : 0)
    AIE.flow(%164, DMA : 1, %165, DMA : 1)
    AIE.flow(%165, DMA : 0, %176, DMA : 0)
    AIE.flow(%152, DMA : 0, %153, DMA : 0)
    AIE.flow(%152, DMA : 1, %153, DMA : 1)
    AIE.flow(%153, DMA : 0, %176, DMA : 1)
    AIE.flow(%140, DMA : 0, %142, DMA : 0)
    AIE.flow(%140, DMA : 1, %142, DMA : 1)
    AIE.flow(%142, DMA : 0, %164, DMA : 0)
    AIE.flow(%128, DMA : 0, %130, DMA : 0)
    AIE.flow(%128, DMA : 1, %130, DMA : 1)
    AIE.flow(%130, DMA : 0, %164, DMA : 1)
    AIE.flow(%116, DMA : 0, %117, DMA : 0)
    AIE.flow(%116, DMA : 1, %117, DMA : 1)
    AIE.flow(%117, DMA : 0, %152, DMA : 0)
    AIE.flow(%104, DMA : 0, %105, DMA : 0)
    AIE.flow(%104, DMA : 1, %105, DMA : 1)
    AIE.flow(%105, DMA : 0, %152, DMA : 1)
    AIE.flow(%91, DMA : 0, %93, DMA : 0)
    AIE.flow(%91, DMA : 1, %93, DMA : 1)
    AIE.flow(%93, DMA : 0, %140, DMA : 0)
    AIE.flow(%78, DMA : 0, %80, DMA : 0)
    AIE.flow(%78, DMA : 1, %80, DMA : 1)
    AIE.flow(%80, DMA : 0, %140, DMA : 1)
    AIE.flow(%66, DMA : 0, %67, DMA : 0)
    AIE.flow(%66, DMA : 1, %67, DMA : 1)
    AIE.flow(%67, DMA : 0, %128, DMA : 0)
    AIE.flow(%54, DMA : 0, %55, DMA : 0)
    AIE.flow(%54, DMA : 1, %55, DMA : 1)
    AIE.flow(%55, DMA : 0, %128, DMA : 1)
    AIE.flow(%41, DMA : 0, %43, DMA : 0)
    AIE.flow(%41, DMA : 1, %43, DMA : 1)
    AIE.flow(%43, DMA : 0, %116, DMA : 0)
    AIE.flow(%28, DMA : 0, %30, DMA : 0)
    AIE.flow(%28, DMA : 1, %30, DMA : 1)
    AIE.flow(%30, DMA : 0, %116, DMA : 1)
    AIE.flow(%15, DMA : 0, %17, DMA : 0)
    AIE.flow(%15, DMA : 1, %17, DMA : 1)
    AIE.flow(%17, DMA : 0, %104, DMA : 0)
    AIE.flow(%2, DMA : 0, %4, DMA : 0)
    AIE.flow(%2, DMA : 1, %4, DMA : 1)
    AIE.flow(%4, DMA : 0, %104, DMA : 1)
  }
}
