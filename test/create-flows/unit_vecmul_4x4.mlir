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

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(47, 2)
// CHECK:           %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:           }
// CHECK:           %[[VAL_2:.*]] = aie.tile(47, 1)
// CHECK:           %[[VAL_3:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK:           }
// CHECK:           %[[VAL_4:.*]] = aie.tile(47, 0)
// CHECK:           %[[VAL_5:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_6:.*]] = aie.switchbox(%[[VAL_5]]) {
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = aie.tile(10, 5)
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_7]], 2)
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "buf47"} : memref<64xi32, 2>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_7]], 1)
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "buf46"} : memref<64xi32, 2>
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_7]], 0)
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "buf45"} : memref<64xi32, 2>
// CHECK:           %[[VAL_14:.*]] = aie.mem(%[[VAL_7]]) {
// CHECK:             %[[VAL_15:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_12]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_16:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_10]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_17:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_8]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_8]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = aie.core(%[[VAL_7]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_12]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_10]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_8]], Acquire, 0)
// CHECK:             affine.for %[[VAL_19:.*]] = 0 to 64 {
// CHECK:               %[[VAL_20:.*]] = affine.load %[[VAL_13]]{{\[}}%[[VAL_19]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_21:.*]] = affine.load %[[VAL_11]]{{\[}}%[[VAL_19]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_22:.*]] = arith.muli %[[VAL_20]], %[[VAL_21]] : i32
// CHECK:               affine.store %[[VAL_22]], %[[VAL_9]]{{\[}}%[[VAL_19]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = aie.tile(46, 2)
// CHECK:           %[[VAL_24:.*]] = aie.tile(46, 1)
// CHECK:           %[[VAL_25:.*]] = aie.tile(46, 0)
// CHECK:           %[[VAL_26:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_27:.*]] = aie.switchbox(%[[VAL_26]]) {
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = aie.tile(9, 5)
// CHECK:           %[[VAL_29:.*]] = aie.lock(%[[VAL_28]], 2)
// CHECK:           %[[VAL_30:.*]] = aie.buffer(%[[VAL_28]]) {sym_name = "buf44"} : memref<64xi32, 2>
// CHECK:           %[[VAL_31:.*]] = aie.lock(%[[VAL_28]], 1)
// CHECK:           %[[VAL_32:.*]] = aie.buffer(%[[VAL_28]]) {sym_name = "buf43"} : memref<64xi32, 2>
// CHECK:           %[[VAL_33:.*]] = aie.lock(%[[VAL_28]], 0)
// CHECK:           %[[VAL_34:.*]] = aie.buffer(%[[VAL_28]]) {sym_name = "buf42"} : memref<64xi32, 2>
// CHECK:           %[[VAL_35:.*]] = aie.mem(%[[VAL_28]]) {
// CHECK:             %[[VAL_36:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_33]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_34]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_33]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_37:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_31]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_32]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_31]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_38:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_29]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_30]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_29]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_39:.*]] = aie.core(%[[VAL_28]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_33]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_31]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_29]], Acquire, 0)
// CHECK:             affine.for %[[VAL_40:.*]] = 0 to 64 {
// CHECK:               %[[VAL_41:.*]] = affine.load %[[VAL_34]]{{\[}}%[[VAL_40]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_42:.*]] = affine.load %[[VAL_32]]{{\[}}%[[VAL_40]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_43:.*]] = arith.muli %[[VAL_41]], %[[VAL_42]] : i32
// CHECK:               affine.store %[[VAL_43]], %[[VAL_30]]{{\[}}%[[VAL_40]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_29]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_31]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_33]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_44:.*]] = aie.tile(43, 2)
// CHECK:           %[[VAL_45:.*]] = aie.switchbox(%[[VAL_44]]) {
// CHECK:           }
// CHECK:           %[[VAL_46:.*]] = aie.tile(43, 1)
// CHECK:           %[[VAL_47:.*]] = aie.switchbox(%[[VAL_46]]) {
// CHECK:           }
// CHECK:           %[[VAL_48:.*]] = aie.tile(43, 0)
// CHECK:           %[[VAL_49:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_50:.*]] = aie.switchbox(%[[VAL_49]]) {
// CHECK:           }
// CHECK:           %[[VAL_51:.*]] = aie.tile(8, 5)
// CHECK:           %[[VAL_52:.*]] = aie.lock(%[[VAL_51]], 2)
// CHECK:           %[[VAL_53:.*]] = aie.buffer(%[[VAL_51]]) {sym_name = "buf41"} : memref<64xi32, 2>
// CHECK:           %[[VAL_54:.*]] = aie.lock(%[[VAL_51]], 1)
// CHECK:           %[[VAL_55:.*]] = aie.buffer(%[[VAL_51]]) {sym_name = "buf40"} : memref<64xi32, 2>
// CHECK:           %[[VAL_56:.*]] = aie.lock(%[[VAL_51]], 0)
// CHECK:           %[[VAL_57:.*]] = aie.buffer(%[[VAL_51]]) {sym_name = "buf39"} : memref<64xi32, 2>
// CHECK:           %[[VAL_58:.*]] = aie.mem(%[[VAL_51]]) {
// CHECK:             %[[VAL_59:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_56]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_57]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_56]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_60:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_54]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_55]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_54]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_61:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_52]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_53]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_52]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_62:.*]] = aie.core(%[[VAL_51]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_56]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_54]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_52]], Acquire, 0)
// CHECK:             affine.for %[[VAL_63:.*]] = 0 to 64 {
// CHECK:               %[[VAL_64:.*]] = affine.load %[[VAL_57]]{{\[}}%[[VAL_63]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_65:.*]] = affine.load %[[VAL_55]]{{\[}}%[[VAL_63]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_66:.*]] = arith.muli %[[VAL_64]], %[[VAL_65]] : i32
// CHECK:               affine.store %[[VAL_66]], %[[VAL_53]]{{\[}}%[[VAL_63]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_52]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_54]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_56]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_67:.*]] = aie.tile(42, 2)
// CHECK:           %[[VAL_68:.*]] = aie.switchbox(%[[VAL_67]]) {
// CHECK:           }
// CHECK:           %[[VAL_69:.*]] = aie.tile(42, 1)
// CHECK:           %[[VAL_70:.*]] = aie.tile(42, 0)
// CHECK:           %[[VAL_71:.*]] = aie.tile(0, 3)
// CHECK:           %[[VAL_72:.*]] = aie.switchbox(%[[VAL_71]]) {
// CHECK:           }
// CHECK:           %[[VAL_73:.*]] = aie.tile(7, 5)
// CHECK:           %[[VAL_74:.*]] = aie.lock(%[[VAL_73]], 2)
// CHECK:           %[[VAL_75:.*]] = aie.buffer(%[[VAL_73]]) {sym_name = "buf38"} : memref<64xi32, 2>
// CHECK:           %[[VAL_76:.*]] = aie.lock(%[[VAL_73]], 1)
// CHECK:           %[[VAL_77:.*]] = aie.buffer(%[[VAL_73]]) {sym_name = "buf37"} : memref<64xi32, 2>
// CHECK:           %[[VAL_78:.*]] = aie.lock(%[[VAL_73]], 0)
// CHECK:           %[[VAL_79:.*]] = aie.buffer(%[[VAL_73]]) {sym_name = "buf36"} : memref<64xi32, 2>
// CHECK:           %[[VAL_80:.*]] = aie.mem(%[[VAL_73]]) {
// CHECK:             %[[VAL_81:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_78]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_79]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_78]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_82:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_76]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_77]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_76]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_83:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_74]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_75]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_74]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_84:.*]] = aie.core(%[[VAL_73]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_78]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_76]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_74]], Acquire, 0)
// CHECK:             affine.for %[[VAL_85:.*]] = 0 to 64 {
// CHECK:               %[[VAL_86:.*]] = affine.load %[[VAL_79]]{{\[}}%[[VAL_85]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_87:.*]] = affine.load %[[VAL_77]]{{\[}}%[[VAL_85]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_88:.*]] = arith.muli %[[VAL_86]], %[[VAL_87]] : i32
// CHECK:               affine.store %[[VAL_88]], %[[VAL_75]]{{\[}}%[[VAL_85]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_74]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_76]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_78]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_89:.*]] = aie.tile(35, 2)
// CHECK:           %[[VAL_90:.*]] = aie.tile(35, 1)
// CHECK:           %[[VAL_91:.*]] = aie.tile(35, 0)
// CHECK:           %[[VAL_92:.*]] = aie.tile(10, 4)
// CHECK:           %[[VAL_93:.*]] = aie.lock(%[[VAL_92]], 2)
// CHECK:           %[[VAL_94:.*]] = aie.buffer(%[[VAL_92]]) {sym_name = "buf35"} : memref<64xi32, 2>
// CHECK:           %[[VAL_95:.*]] = aie.lock(%[[VAL_92]], 1)
// CHECK:           %[[VAL_96:.*]] = aie.buffer(%[[VAL_92]]) {sym_name = "buf34"} : memref<64xi32, 2>
// CHECK:           %[[VAL_97:.*]] = aie.lock(%[[VAL_92]], 0)
// CHECK:           %[[VAL_98:.*]] = aie.buffer(%[[VAL_92]]) {sym_name = "buf33"} : memref<64xi32, 2>
// CHECK:           %[[VAL_99:.*]] = aie.mem(%[[VAL_92]]) {
// CHECK:             %[[VAL_100:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_97]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_98]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_97]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_101:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_95]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_96]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_95]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_102:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_93]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_94]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_93]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_103:.*]] = aie.core(%[[VAL_92]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_97]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_95]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_93]], Acquire, 0)
// CHECK:             affine.for %[[VAL_104:.*]] = 0 to 64 {
// CHECK:               %[[VAL_105:.*]] = affine.load %[[VAL_98]]{{\[}}%[[VAL_104]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_106:.*]] = affine.load %[[VAL_96]]{{\[}}%[[VAL_104]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_107:.*]] = arith.muli %[[VAL_105]], %[[VAL_106]] : i32
// CHECK:               affine.store %[[VAL_107]], %[[VAL_94]]{{\[}}%[[VAL_104]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_93]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_95]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_97]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_108:.*]] = aie.tile(34, 2)
// CHECK:           %[[VAL_109:.*]] = aie.tile(34, 1)
// CHECK:           %[[VAL_110:.*]] = aie.tile(34, 0)
// CHECK:           %[[VAL_111:.*]] = aie.tile(9, 4)
// CHECK:           %[[VAL_112:.*]] = aie.lock(%[[VAL_111]], 2)
// CHECK:           %[[VAL_113:.*]] = aie.buffer(%[[VAL_111]]) {sym_name = "buf32"} : memref<64xi32, 2>
// CHECK:           %[[VAL_114:.*]] = aie.lock(%[[VAL_111]], 1)
// CHECK:           %[[VAL_115:.*]] = aie.buffer(%[[VAL_111]]) {sym_name = "buf31"} : memref<64xi32, 2>
// CHECK:           %[[VAL_116:.*]] = aie.lock(%[[VAL_111]], 0)
// CHECK:           %[[VAL_117:.*]] = aie.buffer(%[[VAL_111]]) {sym_name = "buf30"} : memref<64xi32, 2>
// CHECK:           %[[VAL_118:.*]] = aie.mem(%[[VAL_111]]) {
// CHECK:             %[[VAL_119:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_116]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_117]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_116]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_120:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_114]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_115]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_114]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_121:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_112]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_113]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_112]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_122:.*]] = aie.core(%[[VAL_111]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_116]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_114]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_112]], Acquire, 0)
// CHECK:             affine.for %[[VAL_123:.*]] = 0 to 64 {
// CHECK:               %[[VAL_124:.*]] = affine.load %[[VAL_117]]{{\[}}%[[VAL_123]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_125:.*]] = affine.load %[[VAL_115]]{{\[}}%[[VAL_123]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_126:.*]] = arith.muli %[[VAL_124]], %[[VAL_125]] : i32
// CHECK:               affine.store %[[VAL_126]], %[[VAL_113]]{{\[}}%[[VAL_123]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_112]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_114]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_116]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_127:.*]] = aie.tile(27, 2)
// CHECK:           %[[VAL_128:.*]] = aie.tile(27, 1)
// CHECK:           %[[VAL_129:.*]] = aie.tile(27, 0)
// CHECK:           %[[VAL_130:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_131:.*]] = aie.switchbox(%[[VAL_130]]) {
// CHECK:           }
// CHECK:           %[[VAL_132:.*]] = aie.tile(8, 4)
// CHECK:           %[[VAL_133:.*]] = aie.lock(%[[VAL_132]], 2)
// CHECK:           %[[VAL_134:.*]] = aie.buffer(%[[VAL_132]]) {sym_name = "buf29"} : memref<64xi32, 2>
// CHECK:           %[[VAL_135:.*]] = aie.lock(%[[VAL_132]], 1)
// CHECK:           %[[VAL_136:.*]] = aie.buffer(%[[VAL_132]]) {sym_name = "buf28"} : memref<64xi32, 2>
// CHECK:           %[[VAL_137:.*]] = aie.lock(%[[VAL_132]], 0)
// CHECK:           %[[VAL_138:.*]] = aie.buffer(%[[VAL_132]]) {sym_name = "buf27"} : memref<64xi32, 2>
// CHECK:           %[[VAL_139:.*]] = aie.mem(%[[VAL_132]]) {
// CHECK:             %[[VAL_140:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_137]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_138]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_137]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_141:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_135]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_136]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_135]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_142:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_133]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_134]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_133]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_143:.*]] = aie.core(%[[VAL_132]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_137]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_135]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_133]], Acquire, 0)
// CHECK:             affine.for %[[VAL_144:.*]] = 0 to 64 {
// CHECK:               %[[VAL_145:.*]] = affine.load %[[VAL_138]]{{\[}}%[[VAL_144]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_146:.*]] = affine.load %[[VAL_136]]{{\[}}%[[VAL_144]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_147:.*]] = arith.muli %[[VAL_145]], %[[VAL_146]] : i32
// CHECK:               affine.store %[[VAL_147]], %[[VAL_134]]{{\[}}%[[VAL_144]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_133]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_135]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_137]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_148:.*]] = aie.tile(26, 2)
// CHECK:           %[[VAL_149:.*]] = aie.tile(26, 1)
// CHECK:           %[[VAL_150:.*]] = aie.tile(26, 0)
// CHECK:           %[[VAL_151:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_152:.*]] = aie.switchbox(%[[VAL_151]]) {
// CHECK:           }
// CHECK:           %[[VAL_153:.*]] = aie.tile(7, 4)
// CHECK:           %[[VAL_154:.*]] = aie.lock(%[[VAL_153]], 2)
// CHECK:           %[[VAL_155:.*]] = aie.buffer(%[[VAL_153]]) {sym_name = "buf26"} : memref<64xi32, 2>
// CHECK:           %[[VAL_156:.*]] = aie.lock(%[[VAL_153]], 1)
// CHECK:           %[[VAL_157:.*]] = aie.buffer(%[[VAL_153]]) {sym_name = "buf25"} : memref<64xi32, 2>
// CHECK:           %[[VAL_158:.*]] = aie.lock(%[[VAL_153]], 0)
// CHECK:           %[[VAL_159:.*]] = aie.buffer(%[[VAL_153]]) {sym_name = "buf24"} : memref<64xi32, 2>
// CHECK:           %[[VAL_160:.*]] = aie.mem(%[[VAL_153]]) {
// CHECK:             %[[VAL_161:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_158]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_159]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_158]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_162:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_156]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_157]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_156]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_163:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_154]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_155]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_154]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_164:.*]] = aie.core(%[[VAL_153]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_158]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_156]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_154]], Acquire, 0)
// CHECK:             affine.for %[[VAL_165:.*]] = 0 to 64 {
// CHECK:               %[[VAL_166:.*]] = affine.load %[[VAL_159]]{{\[}}%[[VAL_165]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_167:.*]] = affine.load %[[VAL_157]]{{\[}}%[[VAL_165]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_168:.*]] = arith.muli %[[VAL_166]], %[[VAL_167]] : i32
// CHECK:               affine.store %[[VAL_168]], %[[VAL_155]]{{\[}}%[[VAL_165]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_154]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_156]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_158]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_169:.*]] = aie.tile(19, 2)
// CHECK:           %[[VAL_170:.*]] = aie.tile(19, 1)
// CHECK:           %[[VAL_171:.*]] = aie.tile(19, 0)
// CHECK:           %[[VAL_172:.*]] = aie.tile(10, 3)
// CHECK:           %[[VAL_173:.*]] = aie.lock(%[[VAL_172]], 2)
// CHECK:           %[[VAL_174:.*]] = aie.buffer(%[[VAL_172]]) {sym_name = "buf23"} : memref<64xi32, 2>
// CHECK:           %[[VAL_175:.*]] = aie.lock(%[[VAL_172]], 1)
// CHECK:           %[[VAL_176:.*]] = aie.buffer(%[[VAL_172]]) {sym_name = "buf22"} : memref<64xi32, 2>
// CHECK:           %[[VAL_177:.*]] = aie.lock(%[[VAL_172]], 0)
// CHECK:           %[[VAL_178:.*]] = aie.buffer(%[[VAL_172]]) {sym_name = "buf21"} : memref<64xi32, 2>
// CHECK:           %[[VAL_179:.*]] = aie.mem(%[[VAL_172]]) {
// CHECK:             %[[VAL_180:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_177]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_178]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_177]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_181:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_175]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_176]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_175]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_182:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_173]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_174]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_173]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_183:.*]] = aie.core(%[[VAL_172]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_177]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_175]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_173]], Acquire, 0)
// CHECK:             affine.for %[[VAL_184:.*]] = 0 to 64 {
// CHECK:               %[[VAL_185:.*]] = affine.load %[[VAL_178]]{{\[}}%[[VAL_184]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_186:.*]] = affine.load %[[VAL_176]]{{\[}}%[[VAL_184]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_187:.*]] = arith.muli %[[VAL_185]], %[[VAL_186]] : i32
// CHECK:               affine.store %[[VAL_187]], %[[VAL_174]]{{\[}}%[[VAL_184]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_173]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_175]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_177]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_188:.*]] = aie.tile(18, 2)
// CHECK:           %[[VAL_189:.*]] = aie.tile(18, 1)
// CHECK:           %[[VAL_190:.*]] = aie.tile(18, 0)
// CHECK:           %[[VAL_191:.*]] = aie.tile(9, 3)
// CHECK:           %[[VAL_192:.*]] = aie.lock(%[[VAL_191]], 2)
// CHECK:           %[[VAL_193:.*]] = aie.buffer(%[[VAL_191]]) {sym_name = "buf20"} : memref<64xi32, 2>
// CHECK:           %[[VAL_194:.*]] = aie.lock(%[[VAL_191]], 1)
// CHECK:           %[[VAL_195:.*]] = aie.buffer(%[[VAL_191]]) {sym_name = "buf19"} : memref<64xi32, 2>
// CHECK:           %[[VAL_196:.*]] = aie.lock(%[[VAL_191]], 0)
// CHECK:           %[[VAL_197:.*]] = aie.buffer(%[[VAL_191]]) {sym_name = "buf18"} : memref<64xi32, 2>
// CHECK:           %[[VAL_198:.*]] = aie.mem(%[[VAL_191]]) {
// CHECK:             %[[VAL_199:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_196]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_197]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_196]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_200:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_194]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_195]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_194]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_201:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_192]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_193]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_192]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_202:.*]] = aie.core(%[[VAL_191]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_196]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_194]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_192]], Acquire, 0)
// CHECK:             affine.for %[[VAL_203:.*]] = 0 to 64 {
// CHECK:               %[[VAL_204:.*]] = affine.load %[[VAL_197]]{{\[}}%[[VAL_203]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_205:.*]] = affine.load %[[VAL_195]]{{\[}}%[[VAL_203]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_206:.*]] = arith.muli %[[VAL_204]], %[[VAL_205]] : i32
// CHECK:               affine.store %[[VAL_206]], %[[VAL_193]]{{\[}}%[[VAL_203]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_192]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_194]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_196]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_207:.*]] = aie.tile(11, 2)
// CHECK:           %[[VAL_208:.*]] = aie.tile(11, 1)
// CHECK:           %[[VAL_209:.*]] = aie.tile(11, 0)
// CHECK:           %[[VAL_210:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_211:.*]] = aie.switchbox(%[[VAL_210]]) {
// CHECK:           }
// CHECK:           %[[VAL_212:.*]] = aie.tile(8, 3)
// CHECK:           %[[VAL_213:.*]] = aie.lock(%[[VAL_212]], 2)
// CHECK:           %[[VAL_214:.*]] = aie.buffer(%[[VAL_212]]) {sym_name = "buf17"} : memref<64xi32, 2>
// CHECK:           %[[VAL_215:.*]] = aie.lock(%[[VAL_212]], 1)
// CHECK:           %[[VAL_216:.*]] = aie.buffer(%[[VAL_212]]) {sym_name = "buf16"} : memref<64xi32, 2>
// CHECK:           %[[VAL_217:.*]] = aie.lock(%[[VAL_212]], 0)
// CHECK:           %[[VAL_218:.*]] = aie.buffer(%[[VAL_212]]) {sym_name = "buf15"} : memref<64xi32, 2>
// CHECK:           %[[VAL_219:.*]] = aie.mem(%[[VAL_212]]) {
// CHECK:             %[[VAL_220:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_217]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_218]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_217]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_221:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_215]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_216]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_215]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_222:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_213]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_214]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_213]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_223:.*]] = aie.core(%[[VAL_212]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_217]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_215]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_213]], Acquire, 0)
// CHECK:             affine.for %[[VAL_224:.*]] = 0 to 64 {
// CHECK:               %[[VAL_225:.*]] = affine.load %[[VAL_218]]{{\[}}%[[VAL_224]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_226:.*]] = affine.load %[[VAL_216]]{{\[}}%[[VAL_224]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_227:.*]] = arith.muli %[[VAL_225]], %[[VAL_226]] : i32
// CHECK:               affine.store %[[VAL_227]], %[[VAL_214]]{{\[}}%[[VAL_224]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_213]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_215]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_217]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_228:.*]] = aie.tile(10, 1)
// CHECK:           %[[VAL_229:.*]] = aie.tile(10, 0)
// CHECK:           %[[VAL_230:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_231:.*]] = aie.switchbox(%[[VAL_230]]) {
// CHECK:           }
// CHECK:           %[[VAL_232:.*]] = aie.tile(7, 3)
// CHECK:           %[[VAL_233:.*]] = aie.lock(%[[VAL_232]], 2)
// CHECK:           %[[VAL_234:.*]] = aie.buffer(%[[VAL_232]]) {sym_name = "buf14"} : memref<64xi32, 2>
// CHECK:           %[[VAL_235:.*]] = aie.lock(%[[VAL_232]], 1)
// CHECK:           %[[VAL_236:.*]] = aie.buffer(%[[VAL_232]]) {sym_name = "buf13"} : memref<64xi32, 2>
// CHECK:           %[[VAL_237:.*]] = aie.lock(%[[VAL_232]], 0)
// CHECK:           %[[VAL_238:.*]] = aie.buffer(%[[VAL_232]]) {sym_name = "buf12"} : memref<64xi32, 2>
// CHECK:           %[[VAL_239:.*]] = aie.mem(%[[VAL_232]]) {
// CHECK:             %[[VAL_240:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_237]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_238]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_237]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_241:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_235]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_236]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_235]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_242:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_233]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_234]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_233]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_243:.*]] = aie.core(%[[VAL_232]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_237]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_235]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_233]], Acquire, 0)
// CHECK:             affine.for %[[VAL_244:.*]] = 0 to 64 {
// CHECK:               %[[VAL_245:.*]] = affine.load %[[VAL_238]]{{\[}}%[[VAL_244]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_246:.*]] = affine.load %[[VAL_236]]{{\[}}%[[VAL_244]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_247:.*]] = arith.muli %[[VAL_245]], %[[VAL_246]] : i32
// CHECK:               affine.store %[[VAL_247]], %[[VAL_234]]{{\[}}%[[VAL_244]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_233]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_235]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_237]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_248:.*]] = aie.tile(7, 1)
// CHECK:           %[[VAL_249:.*]] = aie.tile(7, 0)
// CHECK:           %[[VAL_250:.*]] = aie.tile(10, 2)
// CHECK:           %[[VAL_251:.*]] = aie.lock(%[[VAL_250]], 2)
// CHECK:           %[[VAL_252:.*]] = aie.buffer(%[[VAL_250]]) {sym_name = "buf11"} : memref<64xi32, 2>
// CHECK:           %[[VAL_253:.*]] = aie.lock(%[[VAL_250]], 1)
// CHECK:           %[[VAL_254:.*]] = aie.buffer(%[[VAL_250]]) {sym_name = "buf10"} : memref<64xi32, 2>
// CHECK:           %[[VAL_255:.*]] = aie.lock(%[[VAL_250]], 0)
// CHECK:           %[[VAL_256:.*]] = aie.buffer(%[[VAL_250]]) {sym_name = "buf9"} : memref<64xi32, 2>
// CHECK:           %[[VAL_257:.*]] = aie.mem(%[[VAL_250]]) {
// CHECK:             %[[VAL_258:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_255]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_256]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_255]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_259:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_253]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_254]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_253]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_260:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_251]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_252]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_251]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_261:.*]] = aie.core(%[[VAL_250]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_255]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_253]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_251]], Acquire, 0)
// CHECK:             affine.for %[[VAL_262:.*]] = 0 to 64 {
// CHECK:               %[[VAL_263:.*]] = affine.load %[[VAL_256]]{{\[}}%[[VAL_262]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_264:.*]] = affine.load %[[VAL_254]]{{\[}}%[[VAL_262]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_265:.*]] = arith.muli %[[VAL_263]], %[[VAL_264]] : i32
// CHECK:               affine.store %[[VAL_265]], %[[VAL_252]]{{\[}}%[[VAL_262]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_251]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_253]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_255]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_266:.*]] = aie.tile(6, 2)
// CHECK:           %[[VAL_267:.*]] = aie.tile(6, 1)
// CHECK:           %[[VAL_268:.*]] = aie.tile(6, 0)
// CHECK:           %[[VAL_269:.*]] = aie.tile(9, 2)
// CHECK:           %[[VAL_270:.*]] = aie.lock(%[[VAL_269]], 2)
// CHECK:           %[[VAL_271:.*]] = aie.buffer(%[[VAL_269]]) {sym_name = "buf8"} : memref<64xi32, 2>
// CHECK:           %[[VAL_272:.*]] = aie.lock(%[[VAL_269]], 1)
// CHECK:           %[[VAL_273:.*]] = aie.buffer(%[[VAL_269]]) {sym_name = "buf7"} : memref<64xi32, 2>
// CHECK:           %[[VAL_274:.*]] = aie.lock(%[[VAL_269]], 0)
// CHECK:           %[[VAL_275:.*]] = aie.buffer(%[[VAL_269]]) {sym_name = "buf6"} : memref<64xi32, 2>
// CHECK:           %[[VAL_276:.*]] = aie.mem(%[[VAL_269]]) {
// CHECK:             %[[VAL_277:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_274]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_275]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_274]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_278:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_272]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_273]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_272]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_279:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_270]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_271]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_270]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_280:.*]] = aie.core(%[[VAL_269]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_274]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_272]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_270]], Acquire, 0)
// CHECK:             affine.for %[[VAL_281:.*]] = 0 to 64 {
// CHECK:               %[[VAL_282:.*]] = affine.load %[[VAL_275]]{{\[}}%[[VAL_281]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_283:.*]] = affine.load %[[VAL_273]]{{\[}}%[[VAL_281]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_284:.*]] = arith.muli %[[VAL_282]], %[[VAL_283]] : i32
// CHECK:               affine.store %[[VAL_284]], %[[VAL_271]]{{\[}}%[[VAL_281]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_270]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_272]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_274]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_285:.*]] = aie.tile(3, 2)
// CHECK:           %[[VAL_286:.*]] = aie.tile(3, 1)
// CHECK:           %[[VAL_287:.*]] = aie.tile(3, 0)
// CHECK:           %[[VAL_288:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_289:.*]] = aie.switchbox(%[[VAL_288]]) {
// CHECK:           }
// CHECK:           %[[VAL_290:.*]] = aie.tile(8, 2)
// CHECK:           %[[VAL_291:.*]] = aie.lock(%[[VAL_290]], 2)
// CHECK:           %[[VAL_292:.*]] = aie.buffer(%[[VAL_290]]) {sym_name = "buf5"} : memref<64xi32, 2>
// CHECK:           %[[VAL_293:.*]] = aie.lock(%[[VAL_290]], 1)
// CHECK:           %[[VAL_294:.*]] = aie.buffer(%[[VAL_290]]) {sym_name = "buf4"} : memref<64xi32, 2>
// CHECK:           %[[VAL_295:.*]] = aie.lock(%[[VAL_290]], 0)
// CHECK:           %[[VAL_296:.*]] = aie.buffer(%[[VAL_290]]) {sym_name = "buf3"} : memref<64xi32, 2>
// CHECK:           %[[VAL_297:.*]] = aie.mem(%[[VAL_290]]) {
// CHECK:             %[[VAL_298:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_295]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_296]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_295]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_299:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_293]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_294]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_293]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_300:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_291]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_292]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_291]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_301:.*]] = aie.core(%[[VAL_290]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_295]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_293]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_291]], Acquire, 0)
// CHECK:             affine.for %[[VAL_302:.*]] = 0 to 64 {
// CHECK:               %[[VAL_303:.*]] = affine.load %[[VAL_296]]{{\[}}%[[VAL_302]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_304:.*]] = affine.load %[[VAL_294]]{{\[}}%[[VAL_302]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_305:.*]] = arith.muli %[[VAL_303]], %[[VAL_304]] : i32
// CHECK:               affine.store %[[VAL_305]], %[[VAL_292]]{{\[}}%[[VAL_302]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_291]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_293]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_295]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_306:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_307:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_308:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_309:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_310:.*]] = aie.switchbox(%[[VAL_309]]) {
// CHECK:           }
// CHECK:           %[[VAL_311:.*]] = aie.tile(7, 2)
// CHECK:           %[[VAL_312:.*]] = aie.lock(%[[VAL_311]], 2)
// CHECK:           %[[VAL_313:.*]] = aie.buffer(%[[VAL_311]]) {sym_name = "buf2"} : memref<64xi32, 2>
// CHECK:           %[[VAL_314:.*]] = aie.lock(%[[VAL_311]], 1)
// CHECK:           %[[VAL_315:.*]] = aie.buffer(%[[VAL_311]]) {sym_name = "buf1"} : memref<64xi32, 2>
// CHECK:           %[[VAL_316:.*]] = aie.lock(%[[VAL_311]], 0)
// CHECK:           %[[VAL_317:.*]] = aie.buffer(%[[VAL_311]]) {sym_name = "buf0"} : memref<64xi32, 2>
// CHECK:           %[[VAL_318:.*]] = aie.mem(%[[VAL_311]]) {
// CHECK:             %[[VAL_319:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_316]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_317]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_316]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_320:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_314]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_315]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_314]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_321:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_312]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_313]] : memref<64xi32, 2>, 0, 64)
// CHECK:             aie.use_lock(%[[VAL_312]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_322:.*]] = aie.core(%[[VAL_311]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_316]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_314]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_312]], Acquire, 0)
// CHECK:             affine.for %[[VAL_323:.*]] = 0 to 64 {
// CHECK:               %[[VAL_324:.*]] = affine.load %[[VAL_317]]{{\[}}%[[VAL_323]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_325:.*]] = affine.load %[[VAL_315]]{{\[}}%[[VAL_323]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_326:.*]] = arith.muli %[[VAL_324]], %[[VAL_325]] : i32
// CHECK:               affine.store %[[VAL_326]], %[[VAL_313]]{{\[}}%[[VAL_323]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_312]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_314]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_316]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_327:.*]] = aie.switchbox(%[[VAL_308]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_328:.*]] = aie.shim_mux(%[[VAL_308]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_329:.*]] = aie.switchbox(%[[VAL_307]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_330:.*]] = aie.switchbox(%[[VAL_306]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:             aie.connect<East : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_331:.*]] = aie.switchbox(%[[VAL_285]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, South : 0>
// CHECK:             aie.connect<East : 3, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_332:.*]] = aie.tile(4, 2)
// CHECK:           %[[VAL_333:.*]] = aie.switchbox(%[[VAL_332]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_334:.*]] = aie.tile(5, 2)
// CHECK:           %[[VAL_335:.*]] = aie.switchbox(%[[VAL_334]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_336:.*]] = aie.switchbox(%[[VAL_266]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<South : 0, East : 2>
// CHECK:             aie.connect<South : 1, East : 3>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_337:.*]] = aie.switchbox(%[[VAL_311]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<West : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:             aie.connect<West : 2, East : 0>
// CHECK:             aie.connect<West : 3, East : 1>
// CHECK:             aie.connect<East : 0, West : 1>
// CHECK:             aie.connect<East : 1, West : 2>
// CHECK:             aie.connect<East : 2, West : 3>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_338:.*]] = aie.switchbox(%[[VAL_287]]) {
// CHECK:             aie.connect<South : 3, East : 0>
// CHECK:             aie.connect<South : 7, East : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_339:.*]] = aie.shim_mux(%[[VAL_287]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_340:.*]] = aie.tile(4, 0)
// CHECK:           %[[VAL_341:.*]] = aie.switchbox(%[[VAL_340]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_342:.*]] = aie.tile(5, 0)
// CHECK:           %[[VAL_343:.*]] = aie.switchbox(%[[VAL_342]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_344:.*]] = aie.switchbox(%[[VAL_268]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, North : 1>
// CHECK:             aie.connect<South : 3, East : 0>
// CHECK:             aie.connect<South : 7, East : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_345:.*]] = aie.switchbox(%[[VAL_267]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:             aie.connect<East : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_346:.*]] = aie.switchbox(%[[VAL_290]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<West : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:             aie.connect<East : 0, West : 1>
// CHECK:             aie.connect<East : 1, West : 2>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<East : 2, North : 2>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_347:.*]] = aie.shim_mux(%[[VAL_268]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_348:.*]] = aie.switchbox(%[[VAL_249]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<South : 3, East : 2>
// CHECK:             aie.connect<South : 7, East : 3>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_349:.*]] = aie.tile(8, 0)
// CHECK:           %[[VAL_350:.*]] = aie.switchbox(%[[VAL_349]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<West : 2, East : 2>
// CHECK:             aie.connect<West : 3, East : 3>
// CHECK:           }
// CHECK:           %[[VAL_351:.*]] = aie.tile(9, 0)
// CHECK:           %[[VAL_352:.*]] = aie.switchbox(%[[VAL_351]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, North : 1>
// CHECK:             aie.connect<West : 2, East : 0>
// CHECK:             aie.connect<West : 3, East : 1>
// CHECK:             aie.connect<East : 0, North : 2>
// CHECK:             aie.connect<East : 1, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_353:.*]] = aie.tile(9, 1)
// CHECK:           %[[VAL_354:.*]] = aie.switchbox(%[[VAL_353]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<South : 2, West : 0>
// CHECK:             aie.connect<South : 3, West : 1>
// CHECK:             aie.connect<East : 0, North : 2>
// CHECK:             aie.connect<East : 1, North : 3>
// CHECK:             aie.connect<North : 0, West : 2>
// CHECK:             aie.connect<East : 2, West : 3>
// CHECK:             aie.connect<East : 3, North : 4>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<North : 1, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_355:.*]] = aie.switchbox(%[[VAL_269]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:             aie.connect<East : 0, West : 1>
// CHECK:             aie.connect<South : 2, North : 0>
// CHECK:             aie.connect<South : 3, North : 1>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<South : 4, West : 2>
// CHECK:             aie.connect<East : 1, North : 2>
// CHECK:             aie.connect<East : 2, North : 3>
// CHECK:             aie.connect<East : 3, North : 4>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_356:.*]] = aie.switchbox(%[[VAL_286]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_357:.*]] = aie.shim_mux(%[[VAL_249]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_358:.*]] = aie.switchbox(%[[VAL_229]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, North : 1>
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, North : 2>
// CHECK:             aie.connect<East : 1, North : 3>
// CHECK:             aie.connect<East : 2, North : 4>
// CHECK:             aie.connect<East : 3, North : 5>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_359:.*]] = aie.switchbox(%[[VAL_228]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<South : 2, West : 0>
// CHECK:             aie.connect<South : 3, West : 1>
// CHECK:             aie.connect<South : 4, North : 2>
// CHECK:             aie.connect<South : 5, North : 3>
// CHECK:             aie.connect<North : 0, West : 2>
// CHECK:             aie.connect<East : 0, West : 3>
// CHECK:             aie.connect<East : 1, North : 4>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<East : 2, North : 5>
// CHECK:             aie.connect<West : 1, South : 1>
// CHECK:             aie.connect<West : 2, East : 0>
// CHECK:             aie.connect<North : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_360:.*]] = aie.switchbox(%[[VAL_250]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:             aie.connect<South : 2, North : 0>
// CHECK:             aie.connect<South : 3, North : 1>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<South : 4, West : 1>
// CHECK:             aie.connect<South : 5, West : 2>
// CHECK:             aie.connect<East : 0, West : 3>
// CHECK:             aie.connect<East : 1, North : 2>
// CHECK:             aie.connect<East : 2, North : 3>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_361:.*]] = aie.shim_mux(%[[VAL_229]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_362:.*]] = aie.tile(8, 1)
// CHECK:           %[[VAL_363:.*]] = aie.switchbox(%[[VAL_362]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<East : 2, West : 1>
// CHECK:             aie.connect<East : 3, West : 2>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<North : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_364:.*]] = aie.switchbox(%[[VAL_232]]) {
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<East : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_365:.*]] = aie.switchbox(%[[VAL_212]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<East : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<South : 2, West : 2>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_366:.*]] = aie.switchbox(%[[VAL_248]]) {
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<East : 0, West : 1>
// CHECK:             aie.connect<East : 1, South : 0>
// CHECK:             aie.connect<East : 2, South : 1>
// CHECK:             aie.connect<North : 1, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_367:.*]] = aie.switchbox(%[[VAL_209]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_368:.*]] = aie.shim_mux(%[[VAL_209]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_369:.*]] = aie.switchbox(%[[VAL_191]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<East : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<South : 2, West : 2>
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 4, North : 1>
// CHECK:             aie.connect<East : 2, North : 2>
// CHECK:             aie.connect<East : 3, North : 3>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:             aie.connect<North : 1, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_370:.*]] = aie.tile(12, 0)
// CHECK:           %[[VAL_371:.*]] = aie.switchbox(%[[VAL_370]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_372:.*]] = aie.tile(13, 0)
// CHECK:           %[[VAL_373:.*]] = aie.switchbox(%[[VAL_372]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_374:.*]] = aie.tile(14, 0)
// CHECK:           %[[VAL_375:.*]] = aie.switchbox(%[[VAL_374]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_376:.*]] = aie.tile(15, 0)
// CHECK:           %[[VAL_377:.*]] = aie.switchbox(%[[VAL_376]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_378:.*]] = aie.tile(16, 0)
// CHECK:           %[[VAL_379:.*]] = aie.switchbox(%[[VAL_378]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_380:.*]] = aie.tile(17, 0)
// CHECK:           %[[VAL_381:.*]] = aie.switchbox(%[[VAL_380]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_382:.*]] = aie.switchbox(%[[VAL_190]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_383:.*]] = aie.shim_mux(%[[VAL_190]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_384:.*]] = aie.switchbox(%[[VAL_172]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<East : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<South : 2, West : 2>
// CHECK:             aie.connect<South : 3, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_385:.*]] = aie.switchbox(%[[VAL_171]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_386:.*]] = aie.shim_mux(%[[VAL_171]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_387:.*]] = aie.switchbox(%[[VAL_208]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, North : 2>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<West : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_388:.*]] = aie.switchbox(%[[VAL_207]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<South : 2, West : 0>
// CHECK:             aie.connect<East : 0, West : 1>
// CHECK:             aie.connect<East : 1, West : 2>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_389:.*]] = aie.tile(11, 3)
// CHECK:           %[[VAL_390:.*]] = aie.switchbox(%[[VAL_389]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_391:.*]] = aie.tile(20, 0)
// CHECK:           %[[VAL_392:.*]] = aie.switchbox(%[[VAL_391]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_393:.*]] = aie.tile(21, 0)
// CHECK:           %[[VAL_394:.*]] = aie.switchbox(%[[VAL_393]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_395:.*]] = aie.tile(22, 0)
// CHECK:           %[[VAL_396:.*]] = aie.switchbox(%[[VAL_395]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_397:.*]] = aie.tile(23, 0)
// CHECK:           %[[VAL_398:.*]] = aie.switchbox(%[[VAL_397]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_399:.*]] = aie.tile(24, 0)
// CHECK:           %[[VAL_400:.*]] = aie.switchbox(%[[VAL_399]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_401:.*]] = aie.tile(25, 0)
// CHECK:           %[[VAL_402:.*]] = aie.switchbox(%[[VAL_401]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_403:.*]] = aie.switchbox(%[[VAL_150]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_404:.*]] = aie.shim_mux(%[[VAL_150]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_405:.*]] = aie.tile(12, 1)
// CHECK:           %[[VAL_406:.*]] = aie.switchbox(%[[VAL_405]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_407:.*]] = aie.tile(13, 1)
// CHECK:           %[[VAL_408:.*]] = aie.switchbox(%[[VAL_407]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_409:.*]] = aie.tile(14, 1)
// CHECK:           %[[VAL_410:.*]] = aie.switchbox(%[[VAL_409]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_411:.*]] = aie.tile(15, 1)
// CHECK:           %[[VAL_412:.*]] = aie.switchbox(%[[VAL_411]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_413:.*]] = aie.tile(16, 1)
// CHECK:           %[[VAL_414:.*]] = aie.switchbox(%[[VAL_413]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_415:.*]] = aie.tile(17, 1)
// CHECK:           %[[VAL_416:.*]] = aie.switchbox(%[[VAL_415]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:             aie.connect<North : 1, East : 1>
// CHECK:             aie.connect<West : 0, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_417:.*]] = aie.switchbox(%[[VAL_189]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<West : 1, South : 1>
// CHECK:             aie.connect<West : 2, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_418:.*]] = aie.switchbox(%[[VAL_153]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<East : 0, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_419:.*]] = aie.switchbox(%[[VAL_132]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<East : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_420:.*]] = aie.switchbox(%[[VAL_129]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_421:.*]] = aie.shim_mux(%[[VAL_129]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_422:.*]] = aie.switchbox(%[[VAL_170]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<West : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_423:.*]] = aie.switchbox(%[[VAL_111]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<South : 2, DMA : 0>
// CHECK:             aie.connect<South : 3, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<West : 0, South : 1>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_424:.*]] = aie.tile(28, 0)
// CHECK:           %[[VAL_425:.*]] = aie.switchbox(%[[VAL_424]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_426:.*]] = aie.tile(29, 0)
// CHECK:           %[[VAL_427:.*]] = aie.switchbox(%[[VAL_426]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_428:.*]] = aie.tile(30, 0)
// CHECK:           %[[VAL_429:.*]] = aie.switchbox(%[[VAL_428]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_430:.*]] = aie.tile(31, 0)
// CHECK:           %[[VAL_431:.*]] = aie.switchbox(%[[VAL_430]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_432:.*]] = aie.tile(32, 0)
// CHECK:           %[[VAL_433:.*]] = aie.switchbox(%[[VAL_432]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_434:.*]] = aie.tile(33, 0)
// CHECK:           %[[VAL_435:.*]] = aie.switchbox(%[[VAL_434]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_436:.*]] = aie.switchbox(%[[VAL_110]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_437:.*]] = aie.shim_mux(%[[VAL_110]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_438:.*]] = aie.tile(20, 1)
// CHECK:           %[[VAL_439:.*]] = aie.switchbox(%[[VAL_438]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:             aie.connect<East : 2, North : 2>
// CHECK:             aie.connect<East : 3, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_440:.*]] = aie.tile(21, 1)
// CHECK:           %[[VAL_441:.*]] = aie.switchbox(%[[VAL_440]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_442:.*]] = aie.tile(22, 1)
// CHECK:           %[[VAL_443:.*]] = aie.switchbox(%[[VAL_442]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_444:.*]] = aie.tile(23, 1)
// CHECK:           %[[VAL_445:.*]] = aie.switchbox(%[[VAL_444]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_446:.*]] = aie.tile(24, 1)
// CHECK:           %[[VAL_447:.*]] = aie.switchbox(%[[VAL_446]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_448:.*]] = aie.tile(25, 1)
// CHECK:           %[[VAL_449:.*]] = aie.switchbox(%[[VAL_448]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_450:.*]] = aie.switchbox(%[[VAL_149]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_451:.*]] = aie.tile(12, 2)
// CHECK:           %[[VAL_452:.*]] = aie.switchbox(%[[VAL_451]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_453:.*]] = aie.tile(13, 2)
// CHECK:           %[[VAL_454:.*]] = aie.switchbox(%[[VAL_453]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_455:.*]] = aie.tile(14, 2)
// CHECK:           %[[VAL_456:.*]] = aie.switchbox(%[[VAL_455]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_457:.*]] = aie.tile(15, 2)
// CHECK:           %[[VAL_458:.*]] = aie.switchbox(%[[VAL_457]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<North : 0, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_459:.*]] = aie.tile(16, 2)
// CHECK:           %[[VAL_460:.*]] = aie.switchbox(%[[VAL_459]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_461:.*]] = aie.tile(17, 2)
// CHECK:           %[[VAL_462:.*]] = aie.switchbox(%[[VAL_461]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:             aie.connect<West : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_463:.*]] = aie.switchbox(%[[VAL_188]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_464:.*]] = aie.switchbox(%[[VAL_169]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_465:.*]] = aie.tile(20, 2)
// CHECK:           %[[VAL_466:.*]] = aie.switchbox(%[[VAL_465]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<South : 2, North : 0>
// CHECK:             aie.connect<South : 3, North : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_467:.*]] = aie.switchbox(%[[VAL_91]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_468:.*]] = aie.shim_mux(%[[VAL_91]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_469:.*]] = aie.switchbox(%[[VAL_128]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_470:.*]] = aie.tile(12, 3)
// CHECK:           %[[VAL_471:.*]] = aie.switchbox(%[[VAL_470]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_472:.*]] = aie.tile(13, 3)
// CHECK:           %[[VAL_473:.*]] = aie.switchbox(%[[VAL_472]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_474:.*]] = aie.tile(14, 3)
// CHECK:           %[[VAL_475:.*]] = aie.switchbox(%[[VAL_474]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_476:.*]] = aie.tile(15, 3)
// CHECK:           %[[VAL_477:.*]] = aie.switchbox(%[[VAL_476]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_478:.*]] = aie.tile(16, 3)
// CHECK:           %[[VAL_479:.*]] = aie.switchbox(%[[VAL_478]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_480:.*]] = aie.tile(17, 3)
// CHECK:           %[[VAL_481:.*]] = aie.switchbox(%[[VAL_480]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_482:.*]] = aie.tile(18, 3)
// CHECK:           %[[VAL_483:.*]] = aie.switchbox(%[[VAL_482]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_484:.*]] = aie.tile(19, 3)
// CHECK:           %[[VAL_485:.*]] = aie.switchbox(%[[VAL_484]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_486:.*]] = aie.tile(20, 3)
// CHECK:           %[[VAL_487:.*]] = aie.switchbox(%[[VAL_486]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_488:.*]] = aie.switchbox(%[[VAL_92]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:             aie.connect<East : 2, West : 0>
// CHECK:             aie.connect<East : 3, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_489:.*]] = aie.tile(36, 0)
// CHECK:           %[[VAL_490:.*]] = aie.switchbox(%[[VAL_489]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_491:.*]] = aie.tile(37, 0)
// CHECK:           %[[VAL_492:.*]] = aie.switchbox(%[[VAL_491]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_493:.*]] = aie.tile(38, 0)
// CHECK:           %[[VAL_494:.*]] = aie.switchbox(%[[VAL_493]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_495:.*]] = aie.tile(39, 0)
// CHECK:           %[[VAL_496:.*]] = aie.switchbox(%[[VAL_495]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_497:.*]] = aie.tile(40, 0)
// CHECK:           %[[VAL_498:.*]] = aie.switchbox(%[[VAL_497]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_499:.*]] = aie.tile(41, 0)
// CHECK:           %[[VAL_500:.*]] = aie.switchbox(%[[VAL_499]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_501:.*]] = aie.switchbox(%[[VAL_70]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_502:.*]] = aie.shim_mux(%[[VAL_70]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_503:.*]] = aie.tile(28, 1)
// CHECK:           %[[VAL_504:.*]] = aie.switchbox(%[[VAL_503]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_505:.*]] = aie.tile(29, 1)
// CHECK:           %[[VAL_506:.*]] = aie.switchbox(%[[VAL_505]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_507:.*]] = aie.tile(30, 1)
// CHECK:           %[[VAL_508:.*]] = aie.switchbox(%[[VAL_507]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_509:.*]] = aie.tile(31, 1)
// CHECK:           %[[VAL_510:.*]] = aie.switchbox(%[[VAL_509]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_511:.*]] = aie.tile(32, 1)
// CHECK:           %[[VAL_512:.*]] = aie.switchbox(%[[VAL_511]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_513:.*]] = aie.tile(33, 1)
// CHECK:           %[[VAL_514:.*]] = aie.switchbox(%[[VAL_513]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_515:.*]] = aie.switchbox(%[[VAL_109]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_516:.*]] = aie.tile(24, 2)
// CHECK:           %[[VAL_517:.*]] = aie.switchbox(%[[VAL_516]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:             aie.connect<East : 2, West : 0>
// CHECK:             aie.connect<East : 3, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_518:.*]] = aie.tile(25, 2)
// CHECK:           %[[VAL_519:.*]] = aie.switchbox(%[[VAL_518]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_520:.*]] = aie.switchbox(%[[VAL_148]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_521:.*]] = aie.tile(21, 3)
// CHECK:           %[[VAL_522:.*]] = aie.switchbox(%[[VAL_521]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_523:.*]] = aie.tile(22, 3)
// CHECK:           %[[VAL_524:.*]] = aie.switchbox(%[[VAL_523]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_525:.*]] = aie.tile(23, 3)
// CHECK:           %[[VAL_526:.*]] = aie.switchbox(%[[VAL_525]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_527:.*]] = aie.tile(24, 3)
// CHECK:           %[[VAL_528:.*]] = aie.switchbox(%[[VAL_527]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_529:.*]] = aie.tile(15, 4)
// CHECK:           %[[VAL_530:.*]] = aie.switchbox(%[[VAL_529]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_531:.*]] = aie.switchbox(%[[VAL_73]]) {
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<East : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_532:.*]] = aie.switchbox(%[[VAL_51]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 2, DMA : 0>
// CHECK:             aie.connect<East : 3, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_533:.*]] = aie.switchbox(%[[VAL_28]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_534:.*]] = aie.switchbox(%[[VAL_7]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<South : 0, West : 2>
// CHECK:             aie.connect<South : 1, West : 3>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 2, DMA : 0>
// CHECK:             aie.connect<East : 3, DMA : 1>
// CHECK:             aie.connect<DMA : 0, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_535:.*]] = aie.tile(11, 5)
// CHECK:           %[[VAL_536:.*]] = aie.switchbox(%[[VAL_535]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<West : 2, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_537:.*]] = aie.tile(12, 5)
// CHECK:           %[[VAL_538:.*]] = aie.switchbox(%[[VAL_537]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<West : 2, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_539:.*]] = aie.tile(13, 5)
// CHECK:           %[[VAL_540:.*]] = aie.switchbox(%[[VAL_539]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<West : 2, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_541:.*]] = aie.tile(14, 5)
// CHECK:           %[[VAL_542:.*]] = aie.switchbox(%[[VAL_541]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_543:.*]] = aie.tile(15, 5)
// CHECK:           %[[VAL_544:.*]] = aie.switchbox(%[[VAL_543]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<West : 1, East : 0>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_545:.*]] = aie.switchbox(%[[VAL_48]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_546:.*]] = aie.shim_mux(%[[VAL_48]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_547:.*]] = aie.switchbox(%[[VAL_90]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_548:.*]] = aie.tile(21, 2)
// CHECK:           %[[VAL_549:.*]] = aie.switchbox(%[[VAL_548]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_550:.*]] = aie.tile(22, 2)
// CHECK:           %[[VAL_551:.*]] = aie.switchbox(%[[VAL_550]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_552:.*]] = aie.tile(23, 2)
// CHECK:           %[[VAL_553:.*]] = aie.switchbox(%[[VAL_552]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_554:.*]] = aie.switchbox(%[[VAL_127]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_555:.*]] = aie.tile(11, 4)
// CHECK:           %[[VAL_556:.*]] = aie.switchbox(%[[VAL_555]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_557:.*]] = aie.tile(12, 4)
// CHECK:           %[[VAL_558:.*]] = aie.switchbox(%[[VAL_557]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_559:.*]] = aie.tile(13, 4)
// CHECK:           %[[VAL_560:.*]] = aie.switchbox(%[[VAL_559]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_561:.*]] = aie.tile(14, 4)
// CHECK:           %[[VAL_562:.*]] = aie.switchbox(%[[VAL_561]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_563:.*]] = aie.tile(16, 4)
// CHECK:           %[[VAL_564:.*]] = aie.switchbox(%[[VAL_563]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_565:.*]] = aie.tile(17, 4)
// CHECK:           %[[VAL_566:.*]] = aie.switchbox(%[[VAL_565]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_567:.*]] = aie.tile(44, 0)
// CHECK:           %[[VAL_568:.*]] = aie.switchbox(%[[VAL_567]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_569:.*]] = aie.tile(45, 0)
// CHECK:           %[[VAL_570:.*]] = aie.switchbox(%[[VAL_569]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_571:.*]] = aie.switchbox(%[[VAL_25]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_572:.*]] = aie.shim_mux(%[[VAL_25]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_573:.*]] = aie.tile(41, 1)
// CHECK:           %[[VAL_574:.*]] = aie.switchbox(%[[VAL_573]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_575:.*]] = aie.switchbox(%[[VAL_69]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_576:.*]] = aie.tile(28, 2)
// CHECK:           %[[VAL_577:.*]] = aie.switchbox(%[[VAL_576]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_578:.*]] = aie.tile(29, 2)
// CHECK:           %[[VAL_579:.*]] = aie.switchbox(%[[VAL_578]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_580:.*]] = aie.tile(30, 2)
// CHECK:           %[[VAL_581:.*]] = aie.switchbox(%[[VAL_580]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_582:.*]] = aie.tile(31, 2)
// CHECK:           %[[VAL_583:.*]] = aie.switchbox(%[[VAL_582]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_584:.*]] = aie.tile(32, 2)
// CHECK:           %[[VAL_585:.*]] = aie.switchbox(%[[VAL_584]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_586:.*]] = aie.tile(33, 2)
// CHECK:           %[[VAL_587:.*]] = aie.switchbox(%[[VAL_586]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_588:.*]] = aie.switchbox(%[[VAL_108]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_589:.*]] = aie.switchbox(%[[VAL_89]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_590:.*]] = aie.tile(36, 2)
// CHECK:           %[[VAL_591:.*]] = aie.switchbox(%[[VAL_590]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_592:.*]] = aie.tile(37, 2)
// CHECK:           %[[VAL_593:.*]] = aie.switchbox(%[[VAL_592]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_594:.*]] = aie.tile(38, 2)
// CHECK:           %[[VAL_595:.*]] = aie.switchbox(%[[VAL_594]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_596:.*]] = aie.tile(39, 2)
// CHECK:           %[[VAL_597:.*]] = aie.switchbox(%[[VAL_596]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_598:.*]] = aie.tile(40, 2)
// CHECK:           %[[VAL_599:.*]] = aie.switchbox(%[[VAL_598]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_600:.*]] = aie.tile(41, 2)
// CHECK:           %[[VAL_601:.*]] = aie.switchbox(%[[VAL_600]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_602:.*]] = aie.tile(25, 3)
// CHECK:           %[[VAL_603:.*]] = aie.switchbox(%[[VAL_602]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_604:.*]] = aie.tile(26, 3)
// CHECK:           %[[VAL_605:.*]] = aie.switchbox(%[[VAL_604]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_606:.*]] = aie.tile(18, 4)
// CHECK:           %[[VAL_607:.*]] = aie.switchbox(%[[VAL_606]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_608:.*]] = aie.tile(19, 4)
// CHECK:           %[[VAL_609:.*]] = aie.switchbox(%[[VAL_608]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_610:.*]] = aie.tile(20, 4)
// CHECK:           %[[VAL_611:.*]] = aie.switchbox(%[[VAL_610]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_612:.*]] = aie.tile(16, 5)
// CHECK:           %[[VAL_613:.*]] = aie.switchbox(%[[VAL_612]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_614:.*]] = aie.switchbox(%[[VAL_4]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_615:.*]] = aie.shim_mux(%[[VAL_4]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_616:.*]] = aie.switchbox(%[[VAL_24]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_617:.*]] = aie.tile(45, 2)
// CHECK:           %[[VAL_618:.*]] = aie.switchbox(%[[VAL_617]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_619:.*]] = aie.switchbox(%[[VAL_23]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_620:.*]] = aie.tile(42, 3)
// CHECK:           %[[VAL_621:.*]] = aie.switchbox(%[[VAL_620]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_622:.*]] = aie.tile(43, 3)
// CHECK:           %[[VAL_623:.*]] = aie.switchbox(%[[VAL_622]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_624:.*]] = aie.tile(44, 3)
// CHECK:           %[[VAL_625:.*]] = aie.switchbox(%[[VAL_624]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_626:.*]] = aie.tile(45, 3)
// CHECK:           %[[VAL_627:.*]] = aie.switchbox(%[[VAL_626]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_628:.*]] = aie.tile(21, 4)
// CHECK:           %[[VAL_629:.*]] = aie.switchbox(%[[VAL_628]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_630:.*]] = aie.tile(22, 4)
// CHECK:           %[[VAL_631:.*]] = aie.switchbox(%[[VAL_630]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_632:.*]] = aie.tile(23, 4)
// CHECK:           %[[VAL_633:.*]] = aie.switchbox(%[[VAL_632]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_634:.*]] = aie.tile(24, 4)
// CHECK:           %[[VAL_635:.*]] = aie.switchbox(%[[VAL_634]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_636:.*]] = aie.tile(25, 4)
// CHECK:           %[[VAL_637:.*]] = aie.switchbox(%[[VAL_636]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_638:.*]] = aie.tile(26, 4)
// CHECK:           %[[VAL_639:.*]] = aie.switchbox(%[[VAL_638]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_640:.*]] = aie.tile(27, 4)
// CHECK:           %[[VAL_641:.*]] = aie.switchbox(%[[VAL_640]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_642:.*]] = aie.tile(28, 4)
// CHECK:           %[[VAL_643:.*]] = aie.switchbox(%[[VAL_642]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_644:.*]] = aie.tile(29, 4)
// CHECK:           %[[VAL_645:.*]] = aie.switchbox(%[[VAL_644]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_646:.*]] = aie.tile(30, 4)
// CHECK:           %[[VAL_647:.*]] = aie.switchbox(%[[VAL_646]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_648:.*]] = aie.tile(31, 4)
// CHECK:           %[[VAL_649:.*]] = aie.switchbox(%[[VAL_648]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_650:.*]] = aie.tile(32, 4)
// CHECK:           %[[VAL_651:.*]] = aie.switchbox(%[[VAL_650]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_652:.*]] = aie.tile(33, 4)
// CHECK:           %[[VAL_653:.*]] = aie.switchbox(%[[VAL_652]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_654:.*]] = aie.tile(34, 4)
// CHECK:           %[[VAL_655:.*]] = aie.switchbox(%[[VAL_654]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_656:.*]] = aie.tile(35, 4)
// CHECK:           %[[VAL_657:.*]] = aie.switchbox(%[[VAL_656]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_658:.*]] = aie.tile(36, 4)
// CHECK:           %[[VAL_659:.*]] = aie.switchbox(%[[VAL_658]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_660:.*]] = aie.tile(37, 4)
// CHECK:           %[[VAL_661:.*]] = aie.switchbox(%[[VAL_660]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_662:.*]] = aie.tile(38, 4)
// CHECK:           %[[VAL_663:.*]] = aie.switchbox(%[[VAL_662]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_664:.*]] = aie.tile(39, 4)
// CHECK:           %[[VAL_665:.*]] = aie.switchbox(%[[VAL_664]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_666:.*]] = aie.tile(40, 4)
// CHECK:           %[[VAL_667:.*]] = aie.switchbox(%[[VAL_666]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_668:.*]] = aie.tile(41, 4)
// CHECK:           %[[VAL_669:.*]] = aie.switchbox(%[[VAL_668]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_670:.*]] = aie.tile(42, 4)
// CHECK:           %[[VAL_671:.*]] = aie.switchbox(%[[VAL_670]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_672:.*]] = aie.tile(17, 5)
// CHECK:           %[[VAL_673:.*]] = aie.switchbox(%[[VAL_672]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_674:.*]] = aie.tile(18, 5)
// CHECK:           %[[VAL_675:.*]] = aie.switchbox(%[[VAL_674]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           aie.wire(%[[VAL_676:.*]] : North, %[[VAL_677:.*]] : South)
// CHECK:           aie.wire(%[[VAL_308]] : DMA, %[[VAL_676]] : DMA)
// CHECK:           aie.wire(%[[VAL_307]] : Core, %[[VAL_678:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_307]] : DMA, %[[VAL_678]] : DMA)
// CHECK:           aie.wire(%[[VAL_677]] : North, %[[VAL_678]] : South)
// CHECK:           aie.wire(%[[VAL_306]] : Core, %[[VAL_679:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_306]] : DMA, %[[VAL_679]] : DMA)
// CHECK:           aie.wire(%[[VAL_678]] : North, %[[VAL_679]] : South)
// CHECK:           aie.wire(%[[VAL_677]] : East, %[[VAL_680:.*]] : West)
// CHECK:           aie.wire(%[[VAL_681:.*]] : North, %[[VAL_680]] : South)
// CHECK:           aie.wire(%[[VAL_287]] : DMA, %[[VAL_681]] : DMA)
// CHECK:           aie.wire(%[[VAL_678]] : East, %[[VAL_682:.*]] : West)
// CHECK:           aie.wire(%[[VAL_286]] : Core, %[[VAL_682]] : Core)
// CHECK:           aie.wire(%[[VAL_286]] : DMA, %[[VAL_682]] : DMA)
// CHECK:           aie.wire(%[[VAL_680]] : North, %[[VAL_682]] : South)
// CHECK:           aie.wire(%[[VAL_679]] : East, %[[VAL_683:.*]] : West)
// CHECK:           aie.wire(%[[VAL_285]] : Core, %[[VAL_683]] : Core)
// CHECK:           aie.wire(%[[VAL_285]] : DMA, %[[VAL_683]] : DMA)
// CHECK:           aie.wire(%[[VAL_682]] : North, %[[VAL_683]] : South)
// CHECK:           aie.wire(%[[VAL_680]] : East, %[[VAL_684:.*]] : West)
// CHECK:           aie.wire(%[[VAL_683]] : East, %[[VAL_685:.*]] : West)
// CHECK:           aie.wire(%[[VAL_332]] : Core, %[[VAL_685]] : Core)
// CHECK:           aie.wire(%[[VAL_332]] : DMA, %[[VAL_685]] : DMA)
// CHECK:           aie.wire(%[[VAL_684]] : East, %[[VAL_686:.*]] : West)
// CHECK:           aie.wire(%[[VAL_685]] : East, %[[VAL_687:.*]] : West)
// CHECK:           aie.wire(%[[VAL_334]] : Core, %[[VAL_687]] : Core)
// CHECK:           aie.wire(%[[VAL_334]] : DMA, %[[VAL_687]] : DMA)
// CHECK:           aie.wire(%[[VAL_686]] : East, %[[VAL_688:.*]] : West)
// CHECK:           aie.wire(%[[VAL_689:.*]] : North, %[[VAL_688]] : South)
// CHECK:           aie.wire(%[[VAL_268]] : DMA, %[[VAL_689]] : DMA)
// CHECK:           aie.wire(%[[VAL_267]] : Core, %[[VAL_690:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_267]] : DMA, %[[VAL_690]] : DMA)
// CHECK:           aie.wire(%[[VAL_688]] : North, %[[VAL_690]] : South)
// CHECK:           aie.wire(%[[VAL_687]] : East, %[[VAL_691:.*]] : West)
// CHECK:           aie.wire(%[[VAL_266]] : Core, %[[VAL_691]] : Core)
// CHECK:           aie.wire(%[[VAL_266]] : DMA, %[[VAL_691]] : DMA)
// CHECK:           aie.wire(%[[VAL_690]] : North, %[[VAL_691]] : South)
// CHECK:           aie.wire(%[[VAL_688]] : East, %[[VAL_692:.*]] : West)
// CHECK:           aie.wire(%[[VAL_693:.*]] : North, %[[VAL_692]] : South)
// CHECK:           aie.wire(%[[VAL_249]] : DMA, %[[VAL_693]] : DMA)
// CHECK:           aie.wire(%[[VAL_690]] : East, %[[VAL_694:.*]] : West)
// CHECK:           aie.wire(%[[VAL_248]] : Core, %[[VAL_694]] : Core)
// CHECK:           aie.wire(%[[VAL_248]] : DMA, %[[VAL_694]] : DMA)
// CHECK:           aie.wire(%[[VAL_692]] : North, %[[VAL_694]] : South)
// CHECK:           aie.wire(%[[VAL_691]] : East, %[[VAL_695:.*]] : West)
// CHECK:           aie.wire(%[[VAL_311]] : Core, %[[VAL_695]] : Core)
// CHECK:           aie.wire(%[[VAL_311]] : DMA, %[[VAL_695]] : DMA)
// CHECK:           aie.wire(%[[VAL_694]] : North, %[[VAL_695]] : South)
// CHECK:           aie.wire(%[[VAL_232]] : Core, %[[VAL_696:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_232]] : DMA, %[[VAL_696]] : DMA)
// CHECK:           aie.wire(%[[VAL_695]] : North, %[[VAL_696]] : South)
// CHECK:           aie.wire(%[[VAL_153]] : Core, %[[VAL_697:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_153]] : DMA, %[[VAL_697]] : DMA)
// CHECK:           aie.wire(%[[VAL_696]] : North, %[[VAL_697]] : South)
// CHECK:           aie.wire(%[[VAL_73]] : Core, %[[VAL_698:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_73]] : DMA, %[[VAL_698]] : DMA)
// CHECK:           aie.wire(%[[VAL_697]] : North, %[[VAL_698]] : South)
// CHECK:           aie.wire(%[[VAL_692]] : East, %[[VAL_699:.*]] : West)
// CHECK:           aie.wire(%[[VAL_694]] : East, %[[VAL_700:.*]] : West)
// CHECK:           aie.wire(%[[VAL_362]] : Core, %[[VAL_700]] : Core)
// CHECK:           aie.wire(%[[VAL_362]] : DMA, %[[VAL_700]] : DMA)
// CHECK:           aie.wire(%[[VAL_699]] : North, %[[VAL_700]] : South)
// CHECK:           aie.wire(%[[VAL_695]] : East, %[[VAL_701:.*]] : West)
// CHECK:           aie.wire(%[[VAL_290]] : Core, %[[VAL_701]] : Core)
// CHECK:           aie.wire(%[[VAL_290]] : DMA, %[[VAL_701]] : DMA)
// CHECK:           aie.wire(%[[VAL_700]] : North, %[[VAL_701]] : South)
// CHECK:           aie.wire(%[[VAL_696]] : East, %[[VAL_702:.*]] : West)
// CHECK:           aie.wire(%[[VAL_212]] : Core, %[[VAL_702]] : Core)
// CHECK:           aie.wire(%[[VAL_212]] : DMA, %[[VAL_702]] : DMA)
// CHECK:           aie.wire(%[[VAL_701]] : North, %[[VAL_702]] : South)
// CHECK:           aie.wire(%[[VAL_697]] : East, %[[VAL_703:.*]] : West)
// CHECK:           aie.wire(%[[VAL_132]] : Core, %[[VAL_703]] : Core)
// CHECK:           aie.wire(%[[VAL_132]] : DMA, %[[VAL_703]] : DMA)
// CHECK:           aie.wire(%[[VAL_702]] : North, %[[VAL_703]] : South)
// CHECK:           aie.wire(%[[VAL_698]] : East, %[[VAL_704:.*]] : West)
// CHECK:           aie.wire(%[[VAL_51]] : Core, %[[VAL_704]] : Core)
// CHECK:           aie.wire(%[[VAL_51]] : DMA, %[[VAL_704]] : DMA)
// CHECK:           aie.wire(%[[VAL_703]] : North, %[[VAL_704]] : South)
// CHECK:           aie.wire(%[[VAL_699]] : East, %[[VAL_705:.*]] : West)
// CHECK:           aie.wire(%[[VAL_700]] : East, %[[VAL_706:.*]] : West)
// CHECK:           aie.wire(%[[VAL_353]] : Core, %[[VAL_706]] : Core)
// CHECK:           aie.wire(%[[VAL_353]] : DMA, %[[VAL_706]] : DMA)
// CHECK:           aie.wire(%[[VAL_705]] : North, %[[VAL_706]] : South)
// CHECK:           aie.wire(%[[VAL_701]] : East, %[[VAL_707:.*]] : West)
// CHECK:           aie.wire(%[[VAL_269]] : Core, %[[VAL_707]] : Core)
// CHECK:           aie.wire(%[[VAL_269]] : DMA, %[[VAL_707]] : DMA)
// CHECK:           aie.wire(%[[VAL_706]] : North, %[[VAL_707]] : South)
// CHECK:           aie.wire(%[[VAL_702]] : East, %[[VAL_708:.*]] : West)
// CHECK:           aie.wire(%[[VAL_191]] : Core, %[[VAL_708]] : Core)
// CHECK:           aie.wire(%[[VAL_191]] : DMA, %[[VAL_708]] : DMA)
// CHECK:           aie.wire(%[[VAL_707]] : North, %[[VAL_708]] : South)
// CHECK:           aie.wire(%[[VAL_703]] : East, %[[VAL_709:.*]] : West)
// CHECK:           aie.wire(%[[VAL_111]] : Core, %[[VAL_709]] : Core)
// CHECK:           aie.wire(%[[VAL_111]] : DMA, %[[VAL_709]] : DMA)
// CHECK:           aie.wire(%[[VAL_708]] : North, %[[VAL_709]] : South)
// CHECK:           aie.wire(%[[VAL_704]] : East, %[[VAL_710:.*]] : West)
// CHECK:           aie.wire(%[[VAL_28]] : Core, %[[VAL_710]] : Core)
// CHECK:           aie.wire(%[[VAL_28]] : DMA, %[[VAL_710]] : DMA)
// CHECK:           aie.wire(%[[VAL_709]] : North, %[[VAL_710]] : South)
// CHECK:           aie.wire(%[[VAL_705]] : East, %[[VAL_711:.*]] : West)
// CHECK:           aie.wire(%[[VAL_712:.*]] : North, %[[VAL_711]] : South)
// CHECK:           aie.wire(%[[VAL_229]] : DMA, %[[VAL_712]] : DMA)
// CHECK:           aie.wire(%[[VAL_706]] : East, %[[VAL_713:.*]] : West)
// CHECK:           aie.wire(%[[VAL_228]] : Core, %[[VAL_713]] : Core)
// CHECK:           aie.wire(%[[VAL_228]] : DMA, %[[VAL_713]] : DMA)
// CHECK:           aie.wire(%[[VAL_711]] : North, %[[VAL_713]] : South)
// CHECK:           aie.wire(%[[VAL_707]] : East, %[[VAL_714:.*]] : West)
// CHECK:           aie.wire(%[[VAL_250]] : Core, %[[VAL_714]] : Core)
// CHECK:           aie.wire(%[[VAL_250]] : DMA, %[[VAL_714]] : DMA)
// CHECK:           aie.wire(%[[VAL_713]] : North, %[[VAL_714]] : South)
// CHECK:           aie.wire(%[[VAL_708]] : East, %[[VAL_715:.*]] : West)
// CHECK:           aie.wire(%[[VAL_172]] : Core, %[[VAL_715]] : Core)
// CHECK:           aie.wire(%[[VAL_172]] : DMA, %[[VAL_715]] : DMA)
// CHECK:           aie.wire(%[[VAL_714]] : North, %[[VAL_715]] : South)
// CHECK:           aie.wire(%[[VAL_709]] : East, %[[VAL_716:.*]] : West)
// CHECK:           aie.wire(%[[VAL_92]] : Core, %[[VAL_716]] : Core)
// CHECK:           aie.wire(%[[VAL_92]] : DMA, %[[VAL_716]] : DMA)
// CHECK:           aie.wire(%[[VAL_715]] : North, %[[VAL_716]] : South)
// CHECK:           aie.wire(%[[VAL_710]] : East, %[[VAL_717:.*]] : West)
// CHECK:           aie.wire(%[[VAL_7]] : Core, %[[VAL_717]] : Core)
// CHECK:           aie.wire(%[[VAL_7]] : DMA, %[[VAL_717]] : DMA)
// CHECK:           aie.wire(%[[VAL_716]] : North, %[[VAL_717]] : South)
// CHECK:           aie.wire(%[[VAL_711]] : East, %[[VAL_718:.*]] : West)
// CHECK:           aie.wire(%[[VAL_719:.*]] : North, %[[VAL_718]] : South)
// CHECK:           aie.wire(%[[VAL_209]] : DMA, %[[VAL_719]] : DMA)
// CHECK:           aie.wire(%[[VAL_713]] : East, %[[VAL_720:.*]] : West)
// CHECK:           aie.wire(%[[VAL_208]] : Core, %[[VAL_720]] : Core)
// CHECK:           aie.wire(%[[VAL_208]] : DMA, %[[VAL_720]] : DMA)
// CHECK:           aie.wire(%[[VAL_718]] : North, %[[VAL_720]] : South)
// CHECK:           aie.wire(%[[VAL_714]] : East, %[[VAL_721:.*]] : West)
// CHECK:           aie.wire(%[[VAL_207]] : Core, %[[VAL_721]] : Core)
// CHECK:           aie.wire(%[[VAL_207]] : DMA, %[[VAL_721]] : DMA)
// CHECK:           aie.wire(%[[VAL_720]] : North, %[[VAL_721]] : South)
// CHECK:           aie.wire(%[[VAL_715]] : East, %[[VAL_722:.*]] : West)
// CHECK:           aie.wire(%[[VAL_389]] : Core, %[[VAL_722]] : Core)
// CHECK:           aie.wire(%[[VAL_389]] : DMA, %[[VAL_722]] : DMA)
// CHECK:           aie.wire(%[[VAL_721]] : North, %[[VAL_722]] : South)
// CHECK:           aie.wire(%[[VAL_716]] : East, %[[VAL_723:.*]] : West)
// CHECK:           aie.wire(%[[VAL_555]] : Core, %[[VAL_723]] : Core)
// CHECK:           aie.wire(%[[VAL_555]] : DMA, %[[VAL_723]] : DMA)
// CHECK:           aie.wire(%[[VAL_722]] : North, %[[VAL_723]] : South)
// CHECK:           aie.wire(%[[VAL_717]] : East, %[[VAL_724:.*]] : West)
// CHECK:           aie.wire(%[[VAL_535]] : Core, %[[VAL_724]] : Core)
// CHECK:           aie.wire(%[[VAL_535]] : DMA, %[[VAL_724]] : DMA)
// CHECK:           aie.wire(%[[VAL_723]] : North, %[[VAL_724]] : South)
// CHECK:           aie.wire(%[[VAL_718]] : East, %[[VAL_725:.*]] : West)
// CHECK:           aie.wire(%[[VAL_720]] : East, %[[VAL_726:.*]] : West)
// CHECK:           aie.wire(%[[VAL_405]] : Core, %[[VAL_726]] : Core)
// CHECK:           aie.wire(%[[VAL_405]] : DMA, %[[VAL_726]] : DMA)
// CHECK:           aie.wire(%[[VAL_725]] : North, %[[VAL_726]] : South)
// CHECK:           aie.wire(%[[VAL_721]] : East, %[[VAL_727:.*]] : West)
// CHECK:           aie.wire(%[[VAL_451]] : Core, %[[VAL_727]] : Core)
// CHECK:           aie.wire(%[[VAL_451]] : DMA, %[[VAL_727]] : DMA)
// CHECK:           aie.wire(%[[VAL_726]] : North, %[[VAL_727]] : South)
// CHECK:           aie.wire(%[[VAL_722]] : East, %[[VAL_728:.*]] : West)
// CHECK:           aie.wire(%[[VAL_470]] : Core, %[[VAL_728]] : Core)
// CHECK:           aie.wire(%[[VAL_470]] : DMA, %[[VAL_728]] : DMA)
// CHECK:           aie.wire(%[[VAL_727]] : North, %[[VAL_728]] : South)
// CHECK:           aie.wire(%[[VAL_723]] : East, %[[VAL_729:.*]] : West)
// CHECK:           aie.wire(%[[VAL_557]] : Core, %[[VAL_729]] : Core)
// CHECK:           aie.wire(%[[VAL_557]] : DMA, %[[VAL_729]] : DMA)
// CHECK:           aie.wire(%[[VAL_728]] : North, %[[VAL_729]] : South)
// CHECK:           aie.wire(%[[VAL_724]] : East, %[[VAL_730:.*]] : West)
// CHECK:           aie.wire(%[[VAL_537]] : Core, %[[VAL_730]] : Core)
// CHECK:           aie.wire(%[[VAL_537]] : DMA, %[[VAL_730]] : DMA)
// CHECK:           aie.wire(%[[VAL_729]] : North, %[[VAL_730]] : South)
// CHECK:           aie.wire(%[[VAL_725]] : East, %[[VAL_731:.*]] : West)
// CHECK:           aie.wire(%[[VAL_726]] : East, %[[VAL_732:.*]] : West)
// CHECK:           aie.wire(%[[VAL_407]] : Core, %[[VAL_732]] : Core)
// CHECK:           aie.wire(%[[VAL_407]] : DMA, %[[VAL_732]] : DMA)
// CHECK:           aie.wire(%[[VAL_731]] : North, %[[VAL_732]] : South)
// CHECK:           aie.wire(%[[VAL_727]] : East, %[[VAL_733:.*]] : West)
// CHECK:           aie.wire(%[[VAL_453]] : Core, %[[VAL_733]] : Core)
// CHECK:           aie.wire(%[[VAL_453]] : DMA, %[[VAL_733]] : DMA)
// CHECK:           aie.wire(%[[VAL_732]] : North, %[[VAL_733]] : South)
// CHECK:           aie.wire(%[[VAL_728]] : East, %[[VAL_734:.*]] : West)
// CHECK:           aie.wire(%[[VAL_472]] : Core, %[[VAL_734]] : Core)
// CHECK:           aie.wire(%[[VAL_472]] : DMA, %[[VAL_734]] : DMA)
// CHECK:           aie.wire(%[[VAL_733]] : North, %[[VAL_734]] : South)
// CHECK:           aie.wire(%[[VAL_729]] : East, %[[VAL_735:.*]] : West)
// CHECK:           aie.wire(%[[VAL_559]] : Core, %[[VAL_735]] : Core)
// CHECK:           aie.wire(%[[VAL_559]] : DMA, %[[VAL_735]] : DMA)
// CHECK:           aie.wire(%[[VAL_734]] : North, %[[VAL_735]] : South)
// CHECK:           aie.wire(%[[VAL_730]] : East, %[[VAL_736:.*]] : West)
// CHECK:           aie.wire(%[[VAL_539]] : Core, %[[VAL_736]] : Core)
// CHECK:           aie.wire(%[[VAL_539]] : DMA, %[[VAL_736]] : DMA)
// CHECK:           aie.wire(%[[VAL_735]] : North, %[[VAL_736]] : South)
// CHECK:           aie.wire(%[[VAL_731]] : East, %[[VAL_737:.*]] : West)
// CHECK:           aie.wire(%[[VAL_732]] : East, %[[VAL_738:.*]] : West)
// CHECK:           aie.wire(%[[VAL_409]] : Core, %[[VAL_738]] : Core)
// CHECK:           aie.wire(%[[VAL_409]] : DMA, %[[VAL_738]] : DMA)
// CHECK:           aie.wire(%[[VAL_737]] : North, %[[VAL_738]] : South)
// CHECK:           aie.wire(%[[VAL_733]] : East, %[[VAL_739:.*]] : West)
// CHECK:           aie.wire(%[[VAL_455]] : Core, %[[VAL_739]] : Core)
// CHECK:           aie.wire(%[[VAL_455]] : DMA, %[[VAL_739]] : DMA)
// CHECK:           aie.wire(%[[VAL_738]] : North, %[[VAL_739]] : South)
// CHECK:           aie.wire(%[[VAL_734]] : East, %[[VAL_740:.*]] : West)
// CHECK:           aie.wire(%[[VAL_474]] : Core, %[[VAL_740]] : Core)
// CHECK:           aie.wire(%[[VAL_474]] : DMA, %[[VAL_740]] : DMA)
// CHECK:           aie.wire(%[[VAL_739]] : North, %[[VAL_740]] : South)
// CHECK:           aie.wire(%[[VAL_735]] : East, %[[VAL_741:.*]] : West)
// CHECK:           aie.wire(%[[VAL_561]] : Core, %[[VAL_741]] : Core)
// CHECK:           aie.wire(%[[VAL_561]] : DMA, %[[VAL_741]] : DMA)
// CHECK:           aie.wire(%[[VAL_740]] : North, %[[VAL_741]] : South)
// CHECK:           aie.wire(%[[VAL_736]] : East, %[[VAL_742:.*]] : West)
// CHECK:           aie.wire(%[[VAL_541]] : Core, %[[VAL_742]] : Core)
// CHECK:           aie.wire(%[[VAL_541]] : DMA, %[[VAL_742]] : DMA)
// CHECK:           aie.wire(%[[VAL_741]] : North, %[[VAL_742]] : South)
// CHECK:           aie.wire(%[[VAL_737]] : East, %[[VAL_743:.*]] : West)
// CHECK:           aie.wire(%[[VAL_738]] : East, %[[VAL_744:.*]] : West)
// CHECK:           aie.wire(%[[VAL_411]] : Core, %[[VAL_744]] : Core)
// CHECK:           aie.wire(%[[VAL_411]] : DMA, %[[VAL_744]] : DMA)
// CHECK:           aie.wire(%[[VAL_743]] : North, %[[VAL_744]] : South)
// CHECK:           aie.wire(%[[VAL_739]] : East, %[[VAL_745:.*]] : West)
// CHECK:           aie.wire(%[[VAL_457]] : Core, %[[VAL_745]] : Core)
// CHECK:           aie.wire(%[[VAL_457]] : DMA, %[[VAL_745]] : DMA)
// CHECK:           aie.wire(%[[VAL_744]] : North, %[[VAL_745]] : South)
// CHECK:           aie.wire(%[[VAL_740]] : East, %[[VAL_746:.*]] : West)
// CHECK:           aie.wire(%[[VAL_476]] : Core, %[[VAL_746]] : Core)
// CHECK:           aie.wire(%[[VAL_476]] : DMA, %[[VAL_746]] : DMA)
// CHECK:           aie.wire(%[[VAL_745]] : North, %[[VAL_746]] : South)
// CHECK:           aie.wire(%[[VAL_741]] : East, %[[VAL_747:.*]] : West)
// CHECK:           aie.wire(%[[VAL_529]] : Core, %[[VAL_747]] : Core)
// CHECK:           aie.wire(%[[VAL_529]] : DMA, %[[VAL_747]] : DMA)
// CHECK:           aie.wire(%[[VAL_746]] : North, %[[VAL_747]] : South)
// CHECK:           aie.wire(%[[VAL_742]] : East, %[[VAL_748:.*]] : West)
// CHECK:           aie.wire(%[[VAL_543]] : Core, %[[VAL_748]] : Core)
// CHECK:           aie.wire(%[[VAL_543]] : DMA, %[[VAL_748]] : DMA)
// CHECK:           aie.wire(%[[VAL_747]] : North, %[[VAL_748]] : South)
// CHECK:           aie.wire(%[[VAL_743]] : East, %[[VAL_749:.*]] : West)
// CHECK:           aie.wire(%[[VAL_744]] : East, %[[VAL_750:.*]] : West)
// CHECK:           aie.wire(%[[VAL_413]] : Core, %[[VAL_750]] : Core)
// CHECK:           aie.wire(%[[VAL_413]] : DMA, %[[VAL_750]] : DMA)
// CHECK:           aie.wire(%[[VAL_749]] : North, %[[VAL_750]] : South)
// CHECK:           aie.wire(%[[VAL_745]] : East, %[[VAL_751:.*]] : West)
// CHECK:           aie.wire(%[[VAL_459]] : Core, %[[VAL_751]] : Core)
// CHECK:           aie.wire(%[[VAL_459]] : DMA, %[[VAL_751]] : DMA)
// CHECK:           aie.wire(%[[VAL_750]] : North, %[[VAL_751]] : South)
// CHECK:           aie.wire(%[[VAL_746]] : East, %[[VAL_752:.*]] : West)
// CHECK:           aie.wire(%[[VAL_478]] : Core, %[[VAL_752]] : Core)
// CHECK:           aie.wire(%[[VAL_478]] : DMA, %[[VAL_752]] : DMA)
// CHECK:           aie.wire(%[[VAL_751]] : North, %[[VAL_752]] : South)
// CHECK:           aie.wire(%[[VAL_747]] : East, %[[VAL_753:.*]] : West)
// CHECK:           aie.wire(%[[VAL_563]] : Core, %[[VAL_753]] : Core)
// CHECK:           aie.wire(%[[VAL_563]] : DMA, %[[VAL_753]] : DMA)
// CHECK:           aie.wire(%[[VAL_752]] : North, %[[VAL_753]] : South)
// CHECK:           aie.wire(%[[VAL_748]] : East, %[[VAL_754:.*]] : West)
// CHECK:           aie.wire(%[[VAL_612]] : Core, %[[VAL_754]] : Core)
// CHECK:           aie.wire(%[[VAL_612]] : DMA, %[[VAL_754]] : DMA)
// CHECK:           aie.wire(%[[VAL_753]] : North, %[[VAL_754]] : South)
// CHECK:           aie.wire(%[[VAL_749]] : East, %[[VAL_755:.*]] : West)
// CHECK:           aie.wire(%[[VAL_750]] : East, %[[VAL_756:.*]] : West)
// CHECK:           aie.wire(%[[VAL_415]] : Core, %[[VAL_756]] : Core)
// CHECK:           aie.wire(%[[VAL_415]] : DMA, %[[VAL_756]] : DMA)
// CHECK:           aie.wire(%[[VAL_755]] : North, %[[VAL_756]] : South)
// CHECK:           aie.wire(%[[VAL_751]] : East, %[[VAL_757:.*]] : West)
// CHECK:           aie.wire(%[[VAL_461]] : Core, %[[VAL_757]] : Core)
// CHECK:           aie.wire(%[[VAL_461]] : DMA, %[[VAL_757]] : DMA)
// CHECK:           aie.wire(%[[VAL_756]] : North, %[[VAL_757]] : South)
// CHECK:           aie.wire(%[[VAL_752]] : East, %[[VAL_758:.*]] : West)
// CHECK:           aie.wire(%[[VAL_480]] : Core, %[[VAL_758]] : Core)
// CHECK:           aie.wire(%[[VAL_480]] : DMA, %[[VAL_758]] : DMA)
// CHECK:           aie.wire(%[[VAL_757]] : North, %[[VAL_758]] : South)
// CHECK:           aie.wire(%[[VAL_753]] : East, %[[VAL_759:.*]] : West)
// CHECK:           aie.wire(%[[VAL_565]] : Core, %[[VAL_759]] : Core)
// CHECK:           aie.wire(%[[VAL_565]] : DMA, %[[VAL_759]] : DMA)
// CHECK:           aie.wire(%[[VAL_758]] : North, %[[VAL_759]] : South)
// CHECK:           aie.wire(%[[VAL_754]] : East, %[[VAL_760:.*]] : West)
// CHECK:           aie.wire(%[[VAL_672]] : Core, %[[VAL_760]] : Core)
// CHECK:           aie.wire(%[[VAL_672]] : DMA, %[[VAL_760]] : DMA)
// CHECK:           aie.wire(%[[VAL_759]] : North, %[[VAL_760]] : South)
// CHECK:           aie.wire(%[[VAL_755]] : East, %[[VAL_761:.*]] : West)
// CHECK:           aie.wire(%[[VAL_762:.*]] : North, %[[VAL_761]] : South)
// CHECK:           aie.wire(%[[VAL_190]] : DMA, %[[VAL_762]] : DMA)
// CHECK:           aie.wire(%[[VAL_756]] : East, %[[VAL_763:.*]] : West)
// CHECK:           aie.wire(%[[VAL_189]] : Core, %[[VAL_763]] : Core)
// CHECK:           aie.wire(%[[VAL_189]] : DMA, %[[VAL_763]] : DMA)
// CHECK:           aie.wire(%[[VAL_761]] : North, %[[VAL_763]] : South)
// CHECK:           aie.wire(%[[VAL_757]] : East, %[[VAL_764:.*]] : West)
// CHECK:           aie.wire(%[[VAL_188]] : Core, %[[VAL_764]] : Core)
// CHECK:           aie.wire(%[[VAL_188]] : DMA, %[[VAL_764]] : DMA)
// CHECK:           aie.wire(%[[VAL_763]] : North, %[[VAL_764]] : South)
// CHECK:           aie.wire(%[[VAL_758]] : East, %[[VAL_765:.*]] : West)
// CHECK:           aie.wire(%[[VAL_482]] : Core, %[[VAL_765]] : Core)
// CHECK:           aie.wire(%[[VAL_482]] : DMA, %[[VAL_765]] : DMA)
// CHECK:           aie.wire(%[[VAL_764]] : North, %[[VAL_765]] : South)
// CHECK:           aie.wire(%[[VAL_759]] : East, %[[VAL_766:.*]] : West)
// CHECK:           aie.wire(%[[VAL_606]] : Core, %[[VAL_766]] : Core)
// CHECK:           aie.wire(%[[VAL_606]] : DMA, %[[VAL_766]] : DMA)
// CHECK:           aie.wire(%[[VAL_765]] : North, %[[VAL_766]] : South)
// CHECK:           aie.wire(%[[VAL_760]] : East, %[[VAL_767:.*]] : West)
// CHECK:           aie.wire(%[[VAL_674]] : Core, %[[VAL_767]] : Core)
// CHECK:           aie.wire(%[[VAL_674]] : DMA, %[[VAL_767]] : DMA)
// CHECK:           aie.wire(%[[VAL_766]] : North, %[[VAL_767]] : South)
// CHECK:           aie.wire(%[[VAL_761]] : East, %[[VAL_768:.*]] : West)
// CHECK:           aie.wire(%[[VAL_769:.*]] : North, %[[VAL_768]] : South)
// CHECK:           aie.wire(%[[VAL_171]] : DMA, %[[VAL_769]] : DMA)
// CHECK:           aie.wire(%[[VAL_763]] : East, %[[VAL_770:.*]] : West)
// CHECK:           aie.wire(%[[VAL_170]] : Core, %[[VAL_770]] : Core)
// CHECK:           aie.wire(%[[VAL_170]] : DMA, %[[VAL_770]] : DMA)
// CHECK:           aie.wire(%[[VAL_768]] : North, %[[VAL_770]] : South)
// CHECK:           aie.wire(%[[VAL_764]] : East, %[[VAL_771:.*]] : West)
// CHECK:           aie.wire(%[[VAL_169]] : Core, %[[VAL_771]] : Core)
// CHECK:           aie.wire(%[[VAL_169]] : DMA, %[[VAL_771]] : DMA)
// CHECK:           aie.wire(%[[VAL_770]] : North, %[[VAL_771]] : South)
// CHECK:           aie.wire(%[[VAL_765]] : East, %[[VAL_772:.*]] : West)
// CHECK:           aie.wire(%[[VAL_484]] : Core, %[[VAL_772]] : Core)
// CHECK:           aie.wire(%[[VAL_484]] : DMA, %[[VAL_772]] : DMA)
// CHECK:           aie.wire(%[[VAL_771]] : North, %[[VAL_772]] : South)
// CHECK:           aie.wire(%[[VAL_766]] : East, %[[VAL_773:.*]] : West)
// CHECK:           aie.wire(%[[VAL_608]] : Core, %[[VAL_773]] : Core)
// CHECK:           aie.wire(%[[VAL_608]] : DMA, %[[VAL_773]] : DMA)
// CHECK:           aie.wire(%[[VAL_772]] : North, %[[VAL_773]] : South)
// CHECK:           aie.wire(%[[VAL_768]] : East, %[[VAL_774:.*]] : West)
// CHECK:           aie.wire(%[[VAL_770]] : East, %[[VAL_775:.*]] : West)
// CHECK:           aie.wire(%[[VAL_438]] : Core, %[[VAL_775]] : Core)
// CHECK:           aie.wire(%[[VAL_438]] : DMA, %[[VAL_775]] : DMA)
// CHECK:           aie.wire(%[[VAL_774]] : North, %[[VAL_775]] : South)
// CHECK:           aie.wire(%[[VAL_771]] : East, %[[VAL_776:.*]] : West)
// CHECK:           aie.wire(%[[VAL_465]] : Core, %[[VAL_776]] : Core)
// CHECK:           aie.wire(%[[VAL_465]] : DMA, %[[VAL_776]] : DMA)
// CHECK:           aie.wire(%[[VAL_775]] : North, %[[VAL_776]] : South)
// CHECK:           aie.wire(%[[VAL_772]] : East, %[[VAL_777:.*]] : West)
// CHECK:           aie.wire(%[[VAL_486]] : Core, %[[VAL_777]] : Core)
// CHECK:           aie.wire(%[[VAL_486]] : DMA, %[[VAL_777]] : DMA)
// CHECK:           aie.wire(%[[VAL_776]] : North, %[[VAL_777]] : South)
// CHECK:           aie.wire(%[[VAL_773]] : East, %[[VAL_778:.*]] : West)
// CHECK:           aie.wire(%[[VAL_610]] : Core, %[[VAL_778]] : Core)
// CHECK:           aie.wire(%[[VAL_610]] : DMA, %[[VAL_778]] : DMA)
// CHECK:           aie.wire(%[[VAL_777]] : North, %[[VAL_778]] : South)
// CHECK:           aie.wire(%[[VAL_774]] : East, %[[VAL_779:.*]] : West)
// CHECK:           aie.wire(%[[VAL_775]] : East, %[[VAL_780:.*]] : West)
// CHECK:           aie.wire(%[[VAL_440]] : Core, %[[VAL_780]] : Core)
// CHECK:           aie.wire(%[[VAL_440]] : DMA, %[[VAL_780]] : DMA)
// CHECK:           aie.wire(%[[VAL_779]] : North, %[[VAL_780]] : South)
// CHECK:           aie.wire(%[[VAL_776]] : East, %[[VAL_781:.*]] : West)
// CHECK:           aie.wire(%[[VAL_548]] : Core, %[[VAL_781]] : Core)
// CHECK:           aie.wire(%[[VAL_548]] : DMA, %[[VAL_781]] : DMA)
// CHECK:           aie.wire(%[[VAL_780]] : North, %[[VAL_781]] : South)
// CHECK:           aie.wire(%[[VAL_777]] : East, %[[VAL_782:.*]] : West)
// CHECK:           aie.wire(%[[VAL_521]] : Core, %[[VAL_782]] : Core)
// CHECK:           aie.wire(%[[VAL_521]] : DMA, %[[VAL_782]] : DMA)
// CHECK:           aie.wire(%[[VAL_781]] : North, %[[VAL_782]] : South)
// CHECK:           aie.wire(%[[VAL_778]] : East, %[[VAL_783:.*]] : West)
// CHECK:           aie.wire(%[[VAL_628]] : Core, %[[VAL_783]] : Core)
// CHECK:           aie.wire(%[[VAL_628]] : DMA, %[[VAL_783]] : DMA)
// CHECK:           aie.wire(%[[VAL_782]] : North, %[[VAL_783]] : South)
// CHECK:           aie.wire(%[[VAL_779]] : East, %[[VAL_784:.*]] : West)
// CHECK:           aie.wire(%[[VAL_780]] : East, %[[VAL_785:.*]] : West)
// CHECK:           aie.wire(%[[VAL_442]] : Core, %[[VAL_785]] : Core)
// CHECK:           aie.wire(%[[VAL_442]] : DMA, %[[VAL_785]] : DMA)
// CHECK:           aie.wire(%[[VAL_784]] : North, %[[VAL_785]] : South)
// CHECK:           aie.wire(%[[VAL_781]] : East, %[[VAL_786:.*]] : West)
// CHECK:           aie.wire(%[[VAL_550]] : Core, %[[VAL_786]] : Core)
// CHECK:           aie.wire(%[[VAL_550]] : DMA, %[[VAL_786]] : DMA)
// CHECK:           aie.wire(%[[VAL_785]] : North, %[[VAL_786]] : South)
// CHECK:           aie.wire(%[[VAL_782]] : East, %[[VAL_787:.*]] : West)
// CHECK:           aie.wire(%[[VAL_523]] : Core, %[[VAL_787]] : Core)
// CHECK:           aie.wire(%[[VAL_523]] : DMA, %[[VAL_787]] : DMA)
// CHECK:           aie.wire(%[[VAL_786]] : North, %[[VAL_787]] : South)
// CHECK:           aie.wire(%[[VAL_783]] : East, %[[VAL_788:.*]] : West)
// CHECK:           aie.wire(%[[VAL_630]] : Core, %[[VAL_788]] : Core)
// CHECK:           aie.wire(%[[VAL_630]] : DMA, %[[VAL_788]] : DMA)
// CHECK:           aie.wire(%[[VAL_787]] : North, %[[VAL_788]] : South)
// CHECK:           aie.wire(%[[VAL_784]] : East, %[[VAL_789:.*]] : West)
// CHECK:           aie.wire(%[[VAL_785]] : East, %[[VAL_790:.*]] : West)
// CHECK:           aie.wire(%[[VAL_444]] : Core, %[[VAL_790]] : Core)
// CHECK:           aie.wire(%[[VAL_444]] : DMA, %[[VAL_790]] : DMA)
// CHECK:           aie.wire(%[[VAL_789]] : North, %[[VAL_790]] : South)
// CHECK:           aie.wire(%[[VAL_786]] : East, %[[VAL_791:.*]] : West)
// CHECK:           aie.wire(%[[VAL_552]] : Core, %[[VAL_791]] : Core)
// CHECK:           aie.wire(%[[VAL_552]] : DMA, %[[VAL_791]] : DMA)
// CHECK:           aie.wire(%[[VAL_790]] : North, %[[VAL_791]] : South)
// CHECK:           aie.wire(%[[VAL_787]] : East, %[[VAL_792:.*]] : West)
// CHECK:           aie.wire(%[[VAL_525]] : Core, %[[VAL_792]] : Core)
// CHECK:           aie.wire(%[[VAL_525]] : DMA, %[[VAL_792]] : DMA)
// CHECK:           aie.wire(%[[VAL_791]] : North, %[[VAL_792]] : South)
// CHECK:           aie.wire(%[[VAL_788]] : East, %[[VAL_793:.*]] : West)
// CHECK:           aie.wire(%[[VAL_632]] : Core, %[[VAL_793]] : Core)
// CHECK:           aie.wire(%[[VAL_632]] : DMA, %[[VAL_793]] : DMA)
// CHECK:           aie.wire(%[[VAL_792]] : North, %[[VAL_793]] : South)
// CHECK:           aie.wire(%[[VAL_789]] : East, %[[VAL_794:.*]] : West)
// CHECK:           aie.wire(%[[VAL_790]] : East, %[[VAL_795:.*]] : West)
// CHECK:           aie.wire(%[[VAL_446]] : Core, %[[VAL_795]] : Core)
// CHECK:           aie.wire(%[[VAL_446]] : DMA, %[[VAL_795]] : DMA)
// CHECK:           aie.wire(%[[VAL_794]] : North, %[[VAL_795]] : South)
// CHECK:           aie.wire(%[[VAL_791]] : East, %[[VAL_796:.*]] : West)
// CHECK:           aie.wire(%[[VAL_516]] : Core, %[[VAL_796]] : Core)
// CHECK:           aie.wire(%[[VAL_516]] : DMA, %[[VAL_796]] : DMA)
// CHECK:           aie.wire(%[[VAL_795]] : North, %[[VAL_796]] : South)
// CHECK:           aie.wire(%[[VAL_792]] : East, %[[VAL_797:.*]] : West)
// CHECK:           aie.wire(%[[VAL_527]] : Core, %[[VAL_797]] : Core)
// CHECK:           aie.wire(%[[VAL_527]] : DMA, %[[VAL_797]] : DMA)
// CHECK:           aie.wire(%[[VAL_796]] : North, %[[VAL_797]] : South)
// CHECK:           aie.wire(%[[VAL_793]] : East, %[[VAL_798:.*]] : West)
// CHECK:           aie.wire(%[[VAL_634]] : Core, %[[VAL_798]] : Core)
// CHECK:           aie.wire(%[[VAL_634]] : DMA, %[[VAL_798]] : DMA)
// CHECK:           aie.wire(%[[VAL_797]] : North, %[[VAL_798]] : South)
// CHECK:           aie.wire(%[[VAL_794]] : East, %[[VAL_799:.*]] : West)
// CHECK:           aie.wire(%[[VAL_795]] : East, %[[VAL_800:.*]] : West)
// CHECK:           aie.wire(%[[VAL_448]] : Core, %[[VAL_800]] : Core)
// CHECK:           aie.wire(%[[VAL_448]] : DMA, %[[VAL_800]] : DMA)
// CHECK:           aie.wire(%[[VAL_799]] : North, %[[VAL_800]] : South)
// CHECK:           aie.wire(%[[VAL_796]] : East, %[[VAL_801:.*]] : West)
// CHECK:           aie.wire(%[[VAL_518]] : Core, %[[VAL_801]] : Core)
// CHECK:           aie.wire(%[[VAL_518]] : DMA, %[[VAL_801]] : DMA)
// CHECK:           aie.wire(%[[VAL_800]] : North, %[[VAL_801]] : South)
// CHECK:           aie.wire(%[[VAL_797]] : East, %[[VAL_802:.*]] : West)
// CHECK:           aie.wire(%[[VAL_602]] : Core, %[[VAL_802]] : Core)
// CHECK:           aie.wire(%[[VAL_602]] : DMA, %[[VAL_802]] : DMA)
// CHECK:           aie.wire(%[[VAL_801]] : North, %[[VAL_802]] : South)
// CHECK:           aie.wire(%[[VAL_798]] : East, %[[VAL_803:.*]] : West)
// CHECK:           aie.wire(%[[VAL_636]] : Core, %[[VAL_803]] : Core)
// CHECK:           aie.wire(%[[VAL_636]] : DMA, %[[VAL_803]] : DMA)
// CHECK:           aie.wire(%[[VAL_802]] : North, %[[VAL_803]] : South)
// CHECK:           aie.wire(%[[VAL_799]] : East, %[[VAL_804:.*]] : West)
// CHECK:           aie.wire(%[[VAL_805:.*]] : North, %[[VAL_804]] : South)
// CHECK:           aie.wire(%[[VAL_150]] : DMA, %[[VAL_805]] : DMA)
// CHECK:           aie.wire(%[[VAL_800]] : East, %[[VAL_806:.*]] : West)
// CHECK:           aie.wire(%[[VAL_149]] : Core, %[[VAL_806]] : Core)
// CHECK:           aie.wire(%[[VAL_149]] : DMA, %[[VAL_806]] : DMA)
// CHECK:           aie.wire(%[[VAL_804]] : North, %[[VAL_806]] : South)
// CHECK:           aie.wire(%[[VAL_801]] : East, %[[VAL_807:.*]] : West)
// CHECK:           aie.wire(%[[VAL_148]] : Core, %[[VAL_807]] : Core)
// CHECK:           aie.wire(%[[VAL_148]] : DMA, %[[VAL_807]] : DMA)
// CHECK:           aie.wire(%[[VAL_806]] : North, %[[VAL_807]] : South)
// CHECK:           aie.wire(%[[VAL_802]] : East, %[[VAL_808:.*]] : West)
// CHECK:           aie.wire(%[[VAL_604]] : Core, %[[VAL_808]] : Core)
// CHECK:           aie.wire(%[[VAL_604]] : DMA, %[[VAL_808]] : DMA)
// CHECK:           aie.wire(%[[VAL_807]] : North, %[[VAL_808]] : South)
// CHECK:           aie.wire(%[[VAL_803]] : East, %[[VAL_809:.*]] : West)
// CHECK:           aie.wire(%[[VAL_638]] : Core, %[[VAL_809]] : Core)
// CHECK:           aie.wire(%[[VAL_638]] : DMA, %[[VAL_809]] : DMA)
// CHECK:           aie.wire(%[[VAL_808]] : North, %[[VAL_809]] : South)
// CHECK:           aie.wire(%[[VAL_804]] : East, %[[VAL_810:.*]] : West)
// CHECK:           aie.wire(%[[VAL_811:.*]] : North, %[[VAL_810]] : South)
// CHECK:           aie.wire(%[[VAL_129]] : DMA, %[[VAL_811]] : DMA)
// CHECK:           aie.wire(%[[VAL_806]] : East, %[[VAL_812:.*]] : West)
// CHECK:           aie.wire(%[[VAL_128]] : Core, %[[VAL_812]] : Core)
// CHECK:           aie.wire(%[[VAL_128]] : DMA, %[[VAL_812]] : DMA)
// CHECK:           aie.wire(%[[VAL_810]] : North, %[[VAL_812]] : South)
// CHECK:           aie.wire(%[[VAL_807]] : East, %[[VAL_813:.*]] : West)
// CHECK:           aie.wire(%[[VAL_127]] : Core, %[[VAL_813]] : Core)
// CHECK:           aie.wire(%[[VAL_127]] : DMA, %[[VAL_813]] : DMA)
// CHECK:           aie.wire(%[[VAL_812]] : North, %[[VAL_813]] : South)
// CHECK:           aie.wire(%[[VAL_809]] : East, %[[VAL_814:.*]] : West)
// CHECK:           aie.wire(%[[VAL_640]] : Core, %[[VAL_814]] : Core)
// CHECK:           aie.wire(%[[VAL_640]] : DMA, %[[VAL_814]] : DMA)
// CHECK:           aie.wire(%[[VAL_810]] : East, %[[VAL_815:.*]] : West)
// CHECK:           aie.wire(%[[VAL_812]] : East, %[[VAL_816:.*]] : West)
// CHECK:           aie.wire(%[[VAL_503]] : Core, %[[VAL_816]] : Core)
// CHECK:           aie.wire(%[[VAL_503]] : DMA, %[[VAL_816]] : DMA)
// CHECK:           aie.wire(%[[VAL_815]] : North, %[[VAL_816]] : South)
// CHECK:           aie.wire(%[[VAL_813]] : East, %[[VAL_817:.*]] : West)
// CHECK:           aie.wire(%[[VAL_576]] : Core, %[[VAL_817]] : Core)
// CHECK:           aie.wire(%[[VAL_576]] : DMA, %[[VAL_817]] : DMA)
// CHECK:           aie.wire(%[[VAL_816]] : North, %[[VAL_817]] : South)
// CHECK:           aie.wire(%[[VAL_814]] : East, %[[VAL_818:.*]] : West)
// CHECK:           aie.wire(%[[VAL_642]] : Core, %[[VAL_818]] : Core)
// CHECK:           aie.wire(%[[VAL_642]] : DMA, %[[VAL_818]] : DMA)
// CHECK:           aie.wire(%[[VAL_815]] : East, %[[VAL_819:.*]] : West)
// CHECK:           aie.wire(%[[VAL_816]] : East, %[[VAL_820:.*]] : West)
// CHECK:           aie.wire(%[[VAL_505]] : Core, %[[VAL_820]] : Core)
// CHECK:           aie.wire(%[[VAL_505]] : DMA, %[[VAL_820]] : DMA)
// CHECK:           aie.wire(%[[VAL_819]] : North, %[[VAL_820]] : South)
// CHECK:           aie.wire(%[[VAL_817]] : East, %[[VAL_821:.*]] : West)
// CHECK:           aie.wire(%[[VAL_578]] : Core, %[[VAL_821]] : Core)
// CHECK:           aie.wire(%[[VAL_578]] : DMA, %[[VAL_821]] : DMA)
// CHECK:           aie.wire(%[[VAL_820]] : North, %[[VAL_821]] : South)
// CHECK:           aie.wire(%[[VAL_818]] : East, %[[VAL_822:.*]] : West)
// CHECK:           aie.wire(%[[VAL_644]] : Core, %[[VAL_822]] : Core)
// CHECK:           aie.wire(%[[VAL_644]] : DMA, %[[VAL_822]] : DMA)
// CHECK:           aie.wire(%[[VAL_819]] : East, %[[VAL_823:.*]] : West)
// CHECK:           aie.wire(%[[VAL_820]] : East, %[[VAL_824:.*]] : West)
// CHECK:           aie.wire(%[[VAL_507]] : Core, %[[VAL_824]] : Core)
// CHECK:           aie.wire(%[[VAL_507]] : DMA, %[[VAL_824]] : DMA)
// CHECK:           aie.wire(%[[VAL_823]] : North, %[[VAL_824]] : South)
// CHECK:           aie.wire(%[[VAL_821]] : East, %[[VAL_825:.*]] : West)
// CHECK:           aie.wire(%[[VAL_580]] : Core, %[[VAL_825]] : Core)
// CHECK:           aie.wire(%[[VAL_580]] : DMA, %[[VAL_825]] : DMA)
// CHECK:           aie.wire(%[[VAL_824]] : North, %[[VAL_825]] : South)
// CHECK:           aie.wire(%[[VAL_822]] : East, %[[VAL_826:.*]] : West)
// CHECK:           aie.wire(%[[VAL_646]] : Core, %[[VAL_826]] : Core)
// CHECK:           aie.wire(%[[VAL_646]] : DMA, %[[VAL_826]] : DMA)
// CHECK:           aie.wire(%[[VAL_823]] : East, %[[VAL_827:.*]] : West)
// CHECK:           aie.wire(%[[VAL_824]] : East, %[[VAL_828:.*]] : West)
// CHECK:           aie.wire(%[[VAL_509]] : Core, %[[VAL_828]] : Core)
// CHECK:           aie.wire(%[[VAL_509]] : DMA, %[[VAL_828]] : DMA)
// CHECK:           aie.wire(%[[VAL_827]] : North, %[[VAL_828]] : South)
// CHECK:           aie.wire(%[[VAL_825]] : East, %[[VAL_829:.*]] : West)
// CHECK:           aie.wire(%[[VAL_582]] : Core, %[[VAL_829]] : Core)
// CHECK:           aie.wire(%[[VAL_582]] : DMA, %[[VAL_829]] : DMA)
// CHECK:           aie.wire(%[[VAL_828]] : North, %[[VAL_829]] : South)
// CHECK:           aie.wire(%[[VAL_826]] : East, %[[VAL_830:.*]] : West)
// CHECK:           aie.wire(%[[VAL_648]] : Core, %[[VAL_830]] : Core)
// CHECK:           aie.wire(%[[VAL_648]] : DMA, %[[VAL_830]] : DMA)
// CHECK:           aie.wire(%[[VAL_827]] : East, %[[VAL_831:.*]] : West)
// CHECK:           aie.wire(%[[VAL_828]] : East, %[[VAL_832:.*]] : West)
// CHECK:           aie.wire(%[[VAL_511]] : Core, %[[VAL_832]] : Core)
// CHECK:           aie.wire(%[[VAL_511]] : DMA, %[[VAL_832]] : DMA)
// CHECK:           aie.wire(%[[VAL_831]] : North, %[[VAL_832]] : South)
// CHECK:           aie.wire(%[[VAL_829]] : East, %[[VAL_833:.*]] : West)
// CHECK:           aie.wire(%[[VAL_584]] : Core, %[[VAL_833]] : Core)
// CHECK:           aie.wire(%[[VAL_584]] : DMA, %[[VAL_833]] : DMA)
// CHECK:           aie.wire(%[[VAL_832]] : North, %[[VAL_833]] : South)
// CHECK:           aie.wire(%[[VAL_830]] : East, %[[VAL_834:.*]] : West)
// CHECK:           aie.wire(%[[VAL_650]] : Core, %[[VAL_834]] : Core)
// CHECK:           aie.wire(%[[VAL_650]] : DMA, %[[VAL_834]] : DMA)
// CHECK:           aie.wire(%[[VAL_831]] : East, %[[VAL_835:.*]] : West)
// CHECK:           aie.wire(%[[VAL_832]] : East, %[[VAL_836:.*]] : West)
// CHECK:           aie.wire(%[[VAL_513]] : Core, %[[VAL_836]] : Core)
// CHECK:           aie.wire(%[[VAL_513]] : DMA, %[[VAL_836]] : DMA)
// CHECK:           aie.wire(%[[VAL_835]] : North, %[[VAL_836]] : South)
// CHECK:           aie.wire(%[[VAL_833]] : East, %[[VAL_837:.*]] : West)
// CHECK:           aie.wire(%[[VAL_586]] : Core, %[[VAL_837]] : Core)
// CHECK:           aie.wire(%[[VAL_586]] : DMA, %[[VAL_837]] : DMA)
// CHECK:           aie.wire(%[[VAL_836]] : North, %[[VAL_837]] : South)
// CHECK:           aie.wire(%[[VAL_834]] : East, %[[VAL_838:.*]] : West)
// CHECK:           aie.wire(%[[VAL_652]] : Core, %[[VAL_838]] : Core)
// CHECK:           aie.wire(%[[VAL_652]] : DMA, %[[VAL_838]] : DMA)
// CHECK:           aie.wire(%[[VAL_835]] : East, %[[VAL_839:.*]] : West)
// CHECK:           aie.wire(%[[VAL_840:.*]] : North, %[[VAL_839]] : South)
// CHECK:           aie.wire(%[[VAL_110]] : DMA, %[[VAL_840]] : DMA)
// CHECK:           aie.wire(%[[VAL_836]] : East, %[[VAL_841:.*]] : West)
// CHECK:           aie.wire(%[[VAL_109]] : Core, %[[VAL_841]] : Core)
// CHECK:           aie.wire(%[[VAL_109]] : DMA, %[[VAL_841]] : DMA)
// CHECK:           aie.wire(%[[VAL_839]] : North, %[[VAL_841]] : South)
// CHECK:           aie.wire(%[[VAL_837]] : East, %[[VAL_842:.*]] : West)
// CHECK:           aie.wire(%[[VAL_108]] : Core, %[[VAL_842]] : Core)
// CHECK:           aie.wire(%[[VAL_108]] : DMA, %[[VAL_842]] : DMA)
// CHECK:           aie.wire(%[[VAL_841]] : North, %[[VAL_842]] : South)
// CHECK:           aie.wire(%[[VAL_838]] : East, %[[VAL_843:.*]] : West)
// CHECK:           aie.wire(%[[VAL_654]] : Core, %[[VAL_843]] : Core)
// CHECK:           aie.wire(%[[VAL_654]] : DMA, %[[VAL_843]] : DMA)
// CHECK:           aie.wire(%[[VAL_839]] : East, %[[VAL_844:.*]] : West)
// CHECK:           aie.wire(%[[VAL_845:.*]] : North, %[[VAL_844]] : South)
// CHECK:           aie.wire(%[[VAL_91]] : DMA, %[[VAL_845]] : DMA)
// CHECK:           aie.wire(%[[VAL_841]] : East, %[[VAL_846:.*]] : West)
// CHECK:           aie.wire(%[[VAL_90]] : Core, %[[VAL_846]] : Core)
// CHECK:           aie.wire(%[[VAL_90]] : DMA, %[[VAL_846]] : DMA)
// CHECK:           aie.wire(%[[VAL_844]] : North, %[[VAL_846]] : South)
// CHECK:           aie.wire(%[[VAL_842]] : East, %[[VAL_847:.*]] : West)
// CHECK:           aie.wire(%[[VAL_89]] : Core, %[[VAL_847]] : Core)
// CHECK:           aie.wire(%[[VAL_89]] : DMA, %[[VAL_847]] : DMA)
// CHECK:           aie.wire(%[[VAL_846]] : North, %[[VAL_847]] : South)
// CHECK:           aie.wire(%[[VAL_843]] : East, %[[VAL_848:.*]] : West)
// CHECK:           aie.wire(%[[VAL_656]] : Core, %[[VAL_848]] : Core)
// CHECK:           aie.wire(%[[VAL_656]] : DMA, %[[VAL_848]] : DMA)
// CHECK:           aie.wire(%[[VAL_844]] : East, %[[VAL_849:.*]] : West)
// CHECK:           aie.wire(%[[VAL_847]] : East, %[[VAL_850:.*]] : West)
// CHECK:           aie.wire(%[[VAL_590]] : Core, %[[VAL_850]] : Core)
// CHECK:           aie.wire(%[[VAL_590]] : DMA, %[[VAL_850]] : DMA)
// CHECK:           aie.wire(%[[VAL_848]] : East, %[[VAL_851:.*]] : West)
// CHECK:           aie.wire(%[[VAL_658]] : Core, %[[VAL_851]] : Core)
// CHECK:           aie.wire(%[[VAL_658]] : DMA, %[[VAL_851]] : DMA)
// CHECK:           aie.wire(%[[VAL_849]] : East, %[[VAL_852:.*]] : West)
// CHECK:           aie.wire(%[[VAL_850]] : East, %[[VAL_853:.*]] : West)
// CHECK:           aie.wire(%[[VAL_592]] : Core, %[[VAL_853]] : Core)
// CHECK:           aie.wire(%[[VAL_592]] : DMA, %[[VAL_853]] : DMA)
// CHECK:           aie.wire(%[[VAL_851]] : East, %[[VAL_854:.*]] : West)
// CHECK:           aie.wire(%[[VAL_660]] : Core, %[[VAL_854]] : Core)
// CHECK:           aie.wire(%[[VAL_660]] : DMA, %[[VAL_854]] : DMA)
// CHECK:           aie.wire(%[[VAL_852]] : East, %[[VAL_855:.*]] : West)
// CHECK:           aie.wire(%[[VAL_853]] : East, %[[VAL_856:.*]] : West)
// CHECK:           aie.wire(%[[VAL_594]] : Core, %[[VAL_856]] : Core)
// CHECK:           aie.wire(%[[VAL_594]] : DMA, %[[VAL_856]] : DMA)
// CHECK:           aie.wire(%[[VAL_854]] : East, %[[VAL_857:.*]] : West)
// CHECK:           aie.wire(%[[VAL_662]] : Core, %[[VAL_857]] : Core)
// CHECK:           aie.wire(%[[VAL_662]] : DMA, %[[VAL_857]] : DMA)
// CHECK:           aie.wire(%[[VAL_855]] : East, %[[VAL_858:.*]] : West)
// CHECK:           aie.wire(%[[VAL_856]] : East, %[[VAL_859:.*]] : West)
// CHECK:           aie.wire(%[[VAL_596]] : Core, %[[VAL_859]] : Core)
// CHECK:           aie.wire(%[[VAL_596]] : DMA, %[[VAL_859]] : DMA)
// CHECK:           aie.wire(%[[VAL_857]] : East, %[[VAL_860:.*]] : West)
// CHECK:           aie.wire(%[[VAL_664]] : Core, %[[VAL_860]] : Core)
// CHECK:           aie.wire(%[[VAL_664]] : DMA, %[[VAL_860]] : DMA)
// CHECK:           aie.wire(%[[VAL_858]] : East, %[[VAL_861:.*]] : West)
// CHECK:           aie.wire(%[[VAL_859]] : East, %[[VAL_862:.*]] : West)
// CHECK:           aie.wire(%[[VAL_598]] : Core, %[[VAL_862]] : Core)
// CHECK:           aie.wire(%[[VAL_598]] : DMA, %[[VAL_862]] : DMA)
// CHECK:           aie.wire(%[[VAL_860]] : East, %[[VAL_863:.*]] : West)
// CHECK:           aie.wire(%[[VAL_666]] : Core, %[[VAL_863]] : Core)
// CHECK:           aie.wire(%[[VAL_666]] : DMA, %[[VAL_863]] : DMA)
// CHECK:           aie.wire(%[[VAL_861]] : East, %[[VAL_864:.*]] : West)
// CHECK:           aie.wire(%[[VAL_573]] : Core, %[[VAL_865:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_573]] : DMA, %[[VAL_865]] : DMA)
// CHECK:           aie.wire(%[[VAL_864]] : North, %[[VAL_865]] : South)
// CHECK:           aie.wire(%[[VAL_862]] : East, %[[VAL_866:.*]] : West)
// CHECK:           aie.wire(%[[VAL_600]] : Core, %[[VAL_866]] : Core)
// CHECK:           aie.wire(%[[VAL_600]] : DMA, %[[VAL_866]] : DMA)
// CHECK:           aie.wire(%[[VAL_865]] : North, %[[VAL_866]] : South)
// CHECK:           aie.wire(%[[VAL_863]] : East, %[[VAL_867:.*]] : West)
// CHECK:           aie.wire(%[[VAL_668]] : Core, %[[VAL_867]] : Core)
// CHECK:           aie.wire(%[[VAL_668]] : DMA, %[[VAL_867]] : DMA)
// CHECK:           aie.wire(%[[VAL_864]] : East, %[[VAL_868:.*]] : West)
// CHECK:           aie.wire(%[[VAL_869:.*]] : North, %[[VAL_868]] : South)
// CHECK:           aie.wire(%[[VAL_70]] : DMA, %[[VAL_869]] : DMA)
// CHECK:           aie.wire(%[[VAL_865]] : East, %[[VAL_870:.*]] : West)
// CHECK:           aie.wire(%[[VAL_69]] : Core, %[[VAL_870]] : Core)
// CHECK:           aie.wire(%[[VAL_69]] : DMA, %[[VAL_870]] : DMA)
// CHECK:           aie.wire(%[[VAL_868]] : North, %[[VAL_870]] : South)
// CHECK:           aie.wire(%[[VAL_620]] : Core, %[[VAL_871:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_620]] : DMA, %[[VAL_871]] : DMA)
// CHECK:           aie.wire(%[[VAL_867]] : East, %[[VAL_872:.*]] : West)
// CHECK:           aie.wire(%[[VAL_670]] : Core, %[[VAL_872]] : Core)
// CHECK:           aie.wire(%[[VAL_670]] : DMA, %[[VAL_872]] : DMA)
// CHECK:           aie.wire(%[[VAL_871]] : North, %[[VAL_872]] : South)
// CHECK:           aie.wire(%[[VAL_868]] : East, %[[VAL_873:.*]] : West)
// CHECK:           aie.wire(%[[VAL_874:.*]] : North, %[[VAL_873]] : South)
// CHECK:           aie.wire(%[[VAL_48]] : DMA, %[[VAL_874]] : DMA)
// CHECK:           aie.wire(%[[VAL_871]] : East, %[[VAL_875:.*]] : West)
// CHECK:           aie.wire(%[[VAL_622]] : Core, %[[VAL_875]] : Core)
// CHECK:           aie.wire(%[[VAL_622]] : DMA, %[[VAL_875]] : DMA)
// CHECK:           aie.wire(%[[VAL_873]] : East, %[[VAL_876:.*]] : West)
// CHECK:           aie.wire(%[[VAL_875]] : East, %[[VAL_877:.*]] : West)
// CHECK:           aie.wire(%[[VAL_624]] : Core, %[[VAL_877]] : Core)
// CHECK:           aie.wire(%[[VAL_624]] : DMA, %[[VAL_877]] : DMA)
// CHECK:           aie.wire(%[[VAL_876]] : East, %[[VAL_878:.*]] : West)
// CHECK:           aie.wire(%[[VAL_617]] : Core, %[[VAL_879:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_617]] : DMA, %[[VAL_879]] : DMA)
// CHECK:           aie.wire(%[[VAL_877]] : East, %[[VAL_880:.*]] : West)
// CHECK:           aie.wire(%[[VAL_626]] : Core, %[[VAL_880]] : Core)
// CHECK:           aie.wire(%[[VAL_626]] : DMA, %[[VAL_880]] : DMA)
// CHECK:           aie.wire(%[[VAL_879]] : North, %[[VAL_880]] : South)
// CHECK:           aie.wire(%[[VAL_878]] : East, %[[VAL_881:.*]] : West)
// CHECK:           aie.wire(%[[VAL_882:.*]] : North, %[[VAL_881]] : South)
// CHECK:           aie.wire(%[[VAL_25]] : DMA, %[[VAL_882]] : DMA)
// CHECK:           aie.wire(%[[VAL_24]] : Core, %[[VAL_883:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_24]] : DMA, %[[VAL_883]] : DMA)
// CHECK:           aie.wire(%[[VAL_881]] : North, %[[VAL_883]] : South)
// CHECK:           aie.wire(%[[VAL_879]] : East, %[[VAL_884:.*]] : West)
// CHECK:           aie.wire(%[[VAL_23]] : Core, %[[VAL_884]] : Core)
// CHECK:           aie.wire(%[[VAL_23]] : DMA, %[[VAL_884]] : DMA)
// CHECK:           aie.wire(%[[VAL_883]] : North, %[[VAL_884]] : South)
// CHECK:           aie.wire(%[[VAL_881]] : East, %[[VAL_885:.*]] : West)
// CHECK:           aie.wire(%[[VAL_886:.*]] : North, %[[VAL_885]] : South)
// CHECK:           aie.wire(%[[VAL_4]] : DMA, %[[VAL_886]] : DMA)
// CHECK:         }

module @vecmul_4x4  {
  aie.device(xcvc1902) {
    %0 = aie.tile(47, 2)
    %1 = aie.tile(47, 1)
    %2 = aie.tile(47, 0)
    %3 = aie.tile(3, 3)
    %4 = aie.tile(10, 5)
    %5 = aie.lock(%4, 2)
    %6 = aie.buffer(%4) {sym_name = "buf47"} : memref<64xi32, 2>
    %7 = aie.lock(%4, 1)
    %8 = aie.buffer(%4) {sym_name = "buf46"} : memref<64xi32, 2>
    %9 = aie.lock(%4, 0)
    %10 = aie.buffer(%4) {sym_name = "buf45"} : memref<64xi32, 2>
    %11 = aie.mem(%4)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%9, Acquire, 0)
      aie.dma_bd(%10 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%9, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%7, Acquire, 0)
      aie.dma_bd(%8 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%7, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%5, Acquire, 1)
      aie.dma_bd(%6 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%5, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %12 = aie.core(%4)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%9, Acquire, 1)
      aie.use_lock(%7, Acquire, 1)
      aie.use_lock(%5, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %10[%arg0] : memref<64xi32, 2>
        %201 = affine.load %8[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %6[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%5, Release, 1)
      aie.use_lock(%7, Release, 0)
      aie.use_lock(%9, Release, 0)
      cf.br ^bb1
    }
    %13 = aie.tile(46, 2)
    %14 = aie.tile(46, 1)
    %15 = aie.tile(46, 0)
    %16 = aie.tile(2, 3)
    %17 = aie.tile(9, 5)
    %18 = aie.lock(%17, 2)
    %19 = aie.buffer(%17) {sym_name = "buf44"} : memref<64xi32, 2>
    %20 = aie.lock(%17, 1)
    %21 = aie.buffer(%17) {sym_name = "buf43"} : memref<64xi32, 2>
    %22 = aie.lock(%17, 0)
    %23 = aie.buffer(%17) {sym_name = "buf42"} : memref<64xi32, 2>
    %24 = aie.mem(%17)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%22, Acquire, 0)
      aie.dma_bd(%23 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%22, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%20, Acquire, 0)
      aie.dma_bd(%21 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%20, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%18, Acquire, 1)
      aie.dma_bd(%19 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%18, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %25 = aie.core(%17)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%22, Acquire, 1)
      aie.use_lock(%20, Acquire, 1)
      aie.use_lock(%18, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %23[%arg0] : memref<64xi32, 2>
        %201 = affine.load %21[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %19[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%18, Release, 1)
      aie.use_lock(%20, Release, 0)
      aie.use_lock(%22, Release, 0)
      cf.br ^bb1
    }
    %26 = aie.tile(43, 2)
    %27 = aie.tile(43, 1)
    %28 = aie.tile(43, 0)
    %29 = aie.tile(1, 3)
    %30 = aie.tile(8, 5)
    %31 = aie.lock(%30, 2)
    %32 = aie.buffer(%30) {sym_name = "buf41"} : memref<64xi32, 2>
    %33 = aie.lock(%30, 1)
    %34 = aie.buffer(%30) {sym_name = "buf40"} : memref<64xi32, 2>
    %35 = aie.lock(%30, 0)
    %36 = aie.buffer(%30) {sym_name = "buf39"} : memref<64xi32, 2>
    %37 = aie.mem(%30)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%35, Acquire, 0)
      aie.dma_bd(%36 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%35, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%33, Acquire, 0)
      aie.dma_bd(%34 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%33, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%31, Acquire, 1)
      aie.dma_bd(%32 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%31, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %38 = aie.core(%30)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%35, Acquire, 1)
      aie.use_lock(%33, Acquire, 1)
      aie.use_lock(%31, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %36[%arg0] : memref<64xi32, 2>
        %201 = affine.load %34[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %32[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%31, Release, 1)
      aie.use_lock(%33, Release, 0)
      aie.use_lock(%35, Release, 0)
      cf.br ^bb1
    }
    %39 = aie.tile(42, 2)
    %40 = aie.tile(42, 1)
    %41 = aie.tile(42, 0)
    %42 = aie.tile(0, 3)
    %43 = aie.tile(7, 5)
    %44 = aie.lock(%43, 2)
    %45 = aie.buffer(%43) {sym_name = "buf38"} : memref<64xi32, 2>
    %46 = aie.lock(%43, 1)
    %47 = aie.buffer(%43) {sym_name = "buf37"} : memref<64xi32, 2>
    %48 = aie.lock(%43, 0)
    %49 = aie.buffer(%43) {sym_name = "buf36"} : memref<64xi32, 2>
    %50 = aie.mem(%43)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%48, Acquire, 0)
      aie.dma_bd(%49 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%48, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%46, Acquire, 0)
      aie.dma_bd(%47 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%46, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%44, Acquire, 1)
      aie.dma_bd(%45 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%44, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %51 = aie.core(%43)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%48, Acquire, 1)
      aie.use_lock(%46, Acquire, 1)
      aie.use_lock(%44, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %49[%arg0] : memref<64xi32, 2>
        %201 = affine.load %47[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %45[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%44, Release, 1)
      aie.use_lock(%46, Release, 0)
      aie.use_lock(%48, Release, 0)
      cf.br ^bb1
    }
    %52 = aie.tile(35, 2)
    %53 = aie.tile(35, 1)
    %54 = aie.tile(35, 0)
    %55 = aie.tile(10, 4)
    %56 = aie.lock(%55, 2)
    %57 = aie.buffer(%55) {sym_name = "buf35"} : memref<64xi32, 2>
    %58 = aie.lock(%55, 1)
    %59 = aie.buffer(%55) {sym_name = "buf34"} : memref<64xi32, 2>
    %60 = aie.lock(%55, 0)
    %61 = aie.buffer(%55) {sym_name = "buf33"} : memref<64xi32, 2>
    %62 = aie.mem(%55)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%60, Acquire, 0)
      aie.dma_bd(%61 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%60, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%58, Acquire, 0)
      aie.dma_bd(%59 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%58, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%56, Acquire, 1)
      aie.dma_bd(%57 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%56, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %63 = aie.core(%55)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%60, Acquire, 1)
      aie.use_lock(%58, Acquire, 1)
      aie.use_lock(%56, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %61[%arg0] : memref<64xi32, 2>
        %201 = affine.load %59[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %57[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%56, Release, 1)
      aie.use_lock(%58, Release, 0)
      aie.use_lock(%60, Release, 0)
      cf.br ^bb1
    }
    %64 = aie.tile(34, 2)
    %65 = aie.tile(34, 1)
    %66 = aie.tile(34, 0)
    %67 = aie.tile(9, 4)
    %68 = aie.lock(%67, 2)
    %69 = aie.buffer(%67) {sym_name = "buf32"} : memref<64xi32, 2>
    %70 = aie.lock(%67, 1)
    %71 = aie.buffer(%67) {sym_name = "buf31"} : memref<64xi32, 2>
    %72 = aie.lock(%67, 0)
    %73 = aie.buffer(%67) {sym_name = "buf30"} : memref<64xi32, 2>
    %74 = aie.mem(%67)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%72, Acquire, 0)
      aie.dma_bd(%73 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%72, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%70, Acquire, 0)
      aie.dma_bd(%71 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%70, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%68, Acquire, 1)
      aie.dma_bd(%69 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%68, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %75 = aie.core(%67)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%72, Acquire, 1)
      aie.use_lock(%70, Acquire, 1)
      aie.use_lock(%68, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %73[%arg0] : memref<64xi32, 2>
        %201 = affine.load %71[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %69[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%68, Release, 1)
      aie.use_lock(%70, Release, 0)
      aie.use_lock(%72, Release, 0)
      cf.br ^bb1
    }
    %76 = aie.tile(27, 2)
    %77 = aie.tile(27, 1)
    %78 = aie.tile(27, 0)
    %79 = aie.tile(1, 2)
    %80 = aie.tile(8, 4)
    %81 = aie.lock(%80, 2)
    %82 = aie.buffer(%80) {sym_name = "buf29"} : memref<64xi32, 2>
    %83 = aie.lock(%80, 1)
    %84 = aie.buffer(%80) {sym_name = "buf28"} : memref<64xi32, 2>
    %85 = aie.lock(%80, 0)
    %86 = aie.buffer(%80) {sym_name = "buf27"} : memref<64xi32, 2>
    %87 = aie.mem(%80)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%85, Acquire, 0)
      aie.dma_bd(%86 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%85, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%83, Acquire, 0)
      aie.dma_bd(%84 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%83, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%81, Acquire, 1)
      aie.dma_bd(%82 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%81, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %88 = aie.core(%80)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%85, Acquire, 1)
      aie.use_lock(%83, Acquire, 1)
      aie.use_lock(%81, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %86[%arg0] : memref<64xi32, 2>
        %201 = affine.load %84[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %82[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%81, Release, 1)
      aie.use_lock(%83, Release, 0)
      aie.use_lock(%85, Release, 0)
      cf.br ^bb1
    }
    %89 = aie.tile(26, 2)
    %90 = aie.tile(26, 1)
    %91 = aie.tile(26, 0)
    %92 = aie.tile(0, 2)
    %93 = aie.tile(7, 4)
    %94 = aie.lock(%93, 2)
    %95 = aie.buffer(%93) {sym_name = "buf26"} : memref<64xi32, 2>
    %96 = aie.lock(%93, 1)
    %97 = aie.buffer(%93) {sym_name = "buf25"} : memref<64xi32, 2>
    %98 = aie.lock(%93, 0)
    %99 = aie.buffer(%93) {sym_name = "buf24"} : memref<64xi32, 2>
    %100 = aie.mem(%93)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%98, Acquire, 0)
      aie.dma_bd(%99 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%98, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%96, Acquire, 0)
      aie.dma_bd(%97 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%96, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%94, Acquire, 1)
      aie.dma_bd(%95 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%94, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %101 = aie.core(%93)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%98, Acquire, 1)
      aie.use_lock(%96, Acquire, 1)
      aie.use_lock(%94, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %99[%arg0] : memref<64xi32, 2>
        %201 = affine.load %97[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %95[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%94, Release, 1)
      aie.use_lock(%96, Release, 0)
      aie.use_lock(%98, Release, 0)
      cf.br ^bb1
    }
    %102 = aie.tile(19, 2)
    %103 = aie.tile(19, 1)
    %104 = aie.tile(19, 0)
    %105 = aie.tile(10, 3)
    %106 = aie.lock(%105, 2)
    %107 = aie.buffer(%105) {sym_name = "buf23"} : memref<64xi32, 2>
    %108 = aie.lock(%105, 1)
    %109 = aie.buffer(%105) {sym_name = "buf22"} : memref<64xi32, 2>
    %110 = aie.lock(%105, 0)
    %111 = aie.buffer(%105) {sym_name = "buf21"} : memref<64xi32, 2>
    %112 = aie.mem(%105)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%110, Acquire, 0)
      aie.dma_bd(%111 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%110, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%108, Acquire, 0)
      aie.dma_bd(%109 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%108, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%106, Acquire, 1)
      aie.dma_bd(%107 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%106, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %113 = aie.core(%105)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%110, Acquire, 1)
      aie.use_lock(%108, Acquire, 1)
      aie.use_lock(%106, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %111[%arg0] : memref<64xi32, 2>
        %201 = affine.load %109[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %107[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%106, Release, 1)
      aie.use_lock(%108, Release, 0)
      aie.use_lock(%110, Release, 0)
      cf.br ^bb1
    }
    %114 = aie.tile(18, 2)
    %115 = aie.tile(18, 1)
    %116 = aie.tile(18, 0)
    %117 = aie.tile(9, 3)
    %118 = aie.lock(%117, 2)
    %119 = aie.buffer(%117) {sym_name = "buf20"} : memref<64xi32, 2>
    %120 = aie.lock(%117, 1)
    %121 = aie.buffer(%117) {sym_name = "buf19"} : memref<64xi32, 2>
    %122 = aie.lock(%117, 0)
    %123 = aie.buffer(%117) {sym_name = "buf18"} : memref<64xi32, 2>
    %124 = aie.mem(%117)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%122, Acquire, 0)
      aie.dma_bd(%123 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%122, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%120, Acquire, 0)
      aie.dma_bd(%121 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%120, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%118, Acquire, 1)
      aie.dma_bd(%119 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%118, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %125 = aie.core(%117)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%122, Acquire, 1)
      aie.use_lock(%120, Acquire, 1)
      aie.use_lock(%118, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %123[%arg0] : memref<64xi32, 2>
        %201 = affine.load %121[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %119[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%118, Release, 1)
      aie.use_lock(%120, Release, 0)
      aie.use_lock(%122, Release, 0)
      cf.br ^bb1
    }
    %126 = aie.tile(11, 2)
    %127 = aie.tile(11, 1)
    %128 = aie.tile(11, 0)
    %129 = aie.tile(1, 1)
    %130 = aie.tile(8, 3)
    %131 = aie.lock(%130, 2)
    %132 = aie.buffer(%130) {sym_name = "buf17"} : memref<64xi32, 2>
    %133 = aie.lock(%130, 1)
    %134 = aie.buffer(%130) {sym_name = "buf16"} : memref<64xi32, 2>
    %135 = aie.lock(%130, 0)
    %136 = aie.buffer(%130) {sym_name = "buf15"} : memref<64xi32, 2>
    %137 = aie.mem(%130)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%135, Acquire, 0)
      aie.dma_bd(%136 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%135, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%133, Acquire, 0)
      aie.dma_bd(%134 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%133, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%131, Acquire, 1)
      aie.dma_bd(%132 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%131, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %138 = aie.core(%130)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%135, Acquire, 1)
      aie.use_lock(%133, Acquire, 1)
      aie.use_lock(%131, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %136[%arg0] : memref<64xi32, 2>
        %201 = affine.load %134[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %132[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%131, Release, 1)
      aie.use_lock(%133, Release, 0)
      aie.use_lock(%135, Release, 0)
      cf.br ^bb1
    }
    %139 = aie.tile(10, 1)
    %140 = aie.tile(10, 0)
    %141 = aie.tile(0, 1)
    %142 = aie.tile(7, 3)
    %143 = aie.lock(%142, 2)
    %144 = aie.buffer(%142) {sym_name = "buf14"} : memref<64xi32, 2>
    %145 = aie.lock(%142, 1)
    %146 = aie.buffer(%142) {sym_name = "buf13"} : memref<64xi32, 2>
    %147 = aie.lock(%142, 0)
    %148 = aie.buffer(%142) {sym_name = "buf12"} : memref<64xi32, 2>
    %149 = aie.mem(%142)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%147, Acquire, 0)
      aie.dma_bd(%148 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%147, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%145, Acquire, 0)
      aie.dma_bd(%146 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%145, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%143, Acquire, 1)
      aie.dma_bd(%144 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%143, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %150 = aie.core(%142)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%147, Acquire, 1)
      aie.use_lock(%145, Acquire, 1)
      aie.use_lock(%143, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %148[%arg0] : memref<64xi32, 2>
        %201 = affine.load %146[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %144[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%143, Release, 1)
      aie.use_lock(%145, Release, 0)
      aie.use_lock(%147, Release, 0)
      cf.br ^bb1
    }
    %151 = aie.tile(7, 1)
    %152 = aie.tile(7, 0)
    %153 = aie.tile(10, 2)
    %154 = aie.lock(%153, 2)
    %155 = aie.buffer(%153) {sym_name = "buf11"} : memref<64xi32, 2>
    %156 = aie.lock(%153, 1)
    %157 = aie.buffer(%153) {sym_name = "buf10"} : memref<64xi32, 2>
    %158 = aie.lock(%153, 0)
    %159 = aie.buffer(%153) {sym_name = "buf9"} : memref<64xi32, 2>
    %160 = aie.mem(%153)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%158, Acquire, 0)
      aie.dma_bd(%159 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%158, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%156, Acquire, 0)
      aie.dma_bd(%157 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%156, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%154, Acquire, 1)
      aie.dma_bd(%155 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%154, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %161 = aie.core(%153)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%158, Acquire, 1)
      aie.use_lock(%156, Acquire, 1)
      aie.use_lock(%154, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %159[%arg0] : memref<64xi32, 2>
        %201 = affine.load %157[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %155[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%154, Release, 1)
      aie.use_lock(%156, Release, 0)
      aie.use_lock(%158, Release, 0)
      cf.br ^bb1
    }
    %162 = aie.tile(6, 2)
    %163 = aie.tile(6, 1)
    %164 = aie.tile(6, 0)
    %165 = aie.tile(9, 2)
    %166 = aie.lock(%165, 2)
    %167 = aie.buffer(%165) {sym_name = "buf8"} : memref<64xi32, 2>
    %168 = aie.lock(%165, 1)
    %169 = aie.buffer(%165) {sym_name = "buf7"} : memref<64xi32, 2>
    %170 = aie.lock(%165, 0)
    %171 = aie.buffer(%165) {sym_name = "buf6"} : memref<64xi32, 2>
    %172 = aie.mem(%165)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%170, Acquire, 0)
      aie.dma_bd(%171 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%170, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%168, Acquire, 0)
      aie.dma_bd(%169 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%168, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%166, Acquire, 1)
      aie.dma_bd(%167 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%166, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %173 = aie.core(%165)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%170, Acquire, 1)
      aie.use_lock(%168, Acquire, 1)
      aie.use_lock(%166, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %171[%arg0] : memref<64xi32, 2>
        %201 = affine.load %169[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %167[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%166, Release, 1)
      aie.use_lock(%168, Release, 0)
      aie.use_lock(%170, Release, 0)
      cf.br ^bb1
    }
    %174 = aie.tile(3, 2)
    %175 = aie.tile(3, 1)
    %176 = aie.tile(3, 0)
    %177 = aie.tile(1, 0)
    %178 = aie.tile(8, 2)
    %179 = aie.lock(%178, 2)
    %180 = aie.buffer(%178) {sym_name = "buf5"} : memref<64xi32, 2>
    %181 = aie.lock(%178, 1)
    %182 = aie.buffer(%178) {sym_name = "buf4"} : memref<64xi32, 2>
    %183 = aie.lock(%178, 0)
    %184 = aie.buffer(%178) {sym_name = "buf3"} : memref<64xi32, 2>
    %185 = aie.mem(%178)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%183, Acquire, 0)
      aie.dma_bd(%184 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%183, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%181, Acquire, 0)
      aie.dma_bd(%182 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%181, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%179, Acquire, 1)
      aie.dma_bd(%180 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%179, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %186 = aie.core(%178)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%183, Acquire, 1)
      aie.use_lock(%181, Acquire, 1)
      aie.use_lock(%179, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %184[%arg0] : memref<64xi32, 2>
        %201 = affine.load %182[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %180[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%179, Release, 1)
      aie.use_lock(%181, Release, 0)
      aie.use_lock(%183, Release, 0)
      cf.br ^bb1
    }
    %187 = aie.tile(2, 2)
    %188 = aie.tile(2, 1)
    %189 = aie.tile(2, 0)
    %190 = aie.tile(0, 0)
    %191 = aie.tile(7, 2)
    %192 = aie.lock(%191, 2)
    %193 = aie.buffer(%191) {sym_name = "buf2"} : memref<64xi32, 2>
    %194 = aie.lock(%191, 1)
    %195 = aie.buffer(%191) {sym_name = "buf1"} : memref<64xi32, 2>
    %196 = aie.lock(%191, 0)
    %197 = aie.buffer(%191) {sym_name = "buf0"} : memref<64xi32, 2>
    %198 = aie.mem(%191)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%196, Acquire, 0)
      aie.dma_bd(%197 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%196, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%194, Acquire, 0)
      aie.dma_bd(%195 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%194, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%192, Acquire, 1)
      aie.dma_bd(%193 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%192, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %199 = aie.core(%191)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%196, Acquire, 1)
      aie.use_lock(%194, Acquire, 1)
      aie.use_lock(%192, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %197[%arg0] : memref<64xi32, 2>
        %201 = affine.load %195[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %193[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%192, Release, 1)
      aie.use_lock(%194, Release, 0)
      aie.use_lock(%196, Release, 0)
      cf.br ^bb1
    }
    aie.flow(%189, DMA : 0, %191, DMA : 0)
    aie.flow(%189, DMA : 1, %191, DMA : 1)
    aie.flow(%191, DMA : 0, %189, DMA : 0)
    aie.flow(%176, DMA : 0, %178, DMA : 0)
    aie.flow(%176, DMA : 1, %178, DMA : 1)
    aie.flow(%178, DMA : 0, %189, DMA : 1)
    aie.flow(%164, DMA : 0, %165, DMA : 0)
    aie.flow(%164, DMA : 1, %165, DMA : 1)
    aie.flow(%165, DMA : 0, %176, DMA : 0)
    aie.flow(%152, DMA : 0, %153, DMA : 0)
    aie.flow(%152, DMA : 1, %153, DMA : 1)
    aie.flow(%153, DMA : 0, %176, DMA : 1)
    aie.flow(%140, DMA : 0, %142, DMA : 0)
    aie.flow(%140, DMA : 1, %142, DMA : 1)
    aie.flow(%142, DMA : 0, %164, DMA : 0)
    aie.flow(%128, DMA : 0, %130, DMA : 0)
    aie.flow(%128, DMA : 1, %130, DMA : 1)
    aie.flow(%130, DMA : 0, %164, DMA : 1)
    aie.flow(%116, DMA : 0, %117, DMA : 0)
    aie.flow(%116, DMA : 1, %117, DMA : 1)
    aie.flow(%117, DMA : 0, %152, DMA : 0)
    aie.flow(%104, DMA : 0, %105, DMA : 0)
    aie.flow(%104, DMA : 1, %105, DMA : 1)
    aie.flow(%105, DMA : 0, %152, DMA : 1)
    aie.flow(%91, DMA : 0, %93, DMA : 0)
    aie.flow(%91, DMA : 1, %93, DMA : 1)
    aie.flow(%93, DMA : 0, %140, DMA : 0)
    aie.flow(%78, DMA : 0, %80, DMA : 0)
    aie.flow(%78, DMA : 1, %80, DMA : 1)
    aie.flow(%80, DMA : 0, %140, DMA : 1)
    aie.flow(%66, DMA : 0, %67, DMA : 0)
    aie.flow(%66, DMA : 1, %67, DMA : 1)
    aie.flow(%67, DMA : 0, %128, DMA : 0)
    aie.flow(%54, DMA : 0, %55, DMA : 0)
    aie.flow(%54, DMA : 1, %55, DMA : 1)
    aie.flow(%55, DMA : 0, %128, DMA : 1)
    aie.flow(%41, DMA : 0, %43, DMA : 0)
    aie.flow(%41, DMA : 1, %43, DMA : 1)
    aie.flow(%43, DMA : 0, %116, DMA : 0)
    aie.flow(%28, DMA : 0, %30, DMA : 0)
    aie.flow(%28, DMA : 1, %30, DMA : 1)
    aie.flow(%30, DMA : 0, %116, DMA : 1)
    aie.flow(%15, DMA : 0, %17, DMA : 0)
    aie.flow(%15, DMA : 1, %17, DMA : 1)
    aie.flow(%17, DMA : 0, %104, DMA : 0)
    aie.flow(%2, DMA : 0, %4, DMA : 0)
    aie.flow(%2, DMA : 1, %4, DMA : 1)
    aie.flow(%4, DMA : 0, %104, DMA : 1)
  }
}
