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
// CHECK:           %[[VAL_1:.*]] = aie.tile(47, 1)
// CHECK:           %[[VAL_2:.*]] = aie.tile(47, 0)
// CHECK:           %[[VAL_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_4:.*]] = aie.tile(10, 5)
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_4]], 2)
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "buf47"} : memref<64xi32, 2>
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_4]], 1)
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "buf46"} : memref<64xi32, 2>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_4]], 0)
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "buf45"} : memref<64xi32, 2>
// CHECK:           %[[VAL_11:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_12:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_9]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_9]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_13:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_7]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_14:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_5]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_5]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = aie.core(%[[VAL_4]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_9]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_7]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_5]], Acquire, 0)
// CHECK:             affine.for %[[VAL_16:.*]] = 0 to 64 {
// CHECK:               %[[VAL_17:.*]] = affine.load %[[VAL_10]]{{\[}}%[[VAL_16]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_18:.*]] = affine.load %[[VAL_8]]{{\[}}%[[VAL_16]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_19:.*]] = arith.muli %[[VAL_17]], %[[VAL_18]] : i32
// CHECK:               affine.store %[[VAL_19]], %[[VAL_6]]{{\[}}%[[VAL_16]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = aie.tile(46, 2)
// CHECK:           %[[VAL_21:.*]] = aie.tile(46, 1)
// CHECK:           %[[VAL_22:.*]] = aie.tile(46, 0)
// CHECK:           %[[VAL_23:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_24:.*]] = aie.tile(9, 5)
// CHECK:           %[[VAL_25:.*]] = aie.lock(%[[VAL_24]], 2)
// CHECK:           %[[VAL_26:.*]] = aie.buffer(%[[VAL_24]]) {sym_name = "buf44"} : memref<64xi32, 2>
// CHECK:           %[[VAL_27:.*]] = aie.lock(%[[VAL_24]], 1)
// CHECK:           %[[VAL_28:.*]] = aie.buffer(%[[VAL_24]]) {sym_name = "buf43"} : memref<64xi32, 2>
// CHECK:           %[[VAL_29:.*]] = aie.lock(%[[VAL_24]], 0)
// CHECK:           %[[VAL_30:.*]] = aie.buffer(%[[VAL_24]]) {sym_name = "buf42"} : memref<64xi32, 2>
// CHECK:           %[[VAL_31:.*]] = aie.mem(%[[VAL_24]]) {
// CHECK:             %[[VAL_32:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_29]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_30]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_29]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_33:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_27]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_28]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_27]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_34:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_25]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_26]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_25]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = aie.core(%[[VAL_24]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_29]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_27]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_25]], Acquire, 0)
// CHECK:             affine.for %[[VAL_36:.*]] = 0 to 64 {
// CHECK:               %[[VAL_37:.*]] = affine.load %[[VAL_30]]{{\[}}%[[VAL_36]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_38:.*]] = affine.load %[[VAL_28]]{{\[}}%[[VAL_36]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_39:.*]] = arith.muli %[[VAL_37]], %[[VAL_38]] : i32
// CHECK:               affine.store %[[VAL_39]], %[[VAL_26]]{{\[}}%[[VAL_36]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_25]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_27]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_29]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_40:.*]] = aie.tile(43, 2)
// CHECK:           %[[VAL_41:.*]] = aie.tile(43, 1)
// CHECK:           %[[VAL_42:.*]] = aie.tile(43, 0)
// CHECK:           %[[VAL_43:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_44:.*]] = aie.tile(8, 5)
// CHECK:           %[[VAL_45:.*]] = aie.lock(%[[VAL_44]], 2)
// CHECK:           %[[VAL_46:.*]] = aie.buffer(%[[VAL_44]]) {sym_name = "buf41"} : memref<64xi32, 2>
// CHECK:           %[[VAL_47:.*]] = aie.lock(%[[VAL_44]], 1)
// CHECK:           %[[VAL_48:.*]] = aie.buffer(%[[VAL_44]]) {sym_name = "buf40"} : memref<64xi32, 2>
// CHECK:           %[[VAL_49:.*]] = aie.lock(%[[VAL_44]], 0)
// CHECK:           %[[VAL_50:.*]] = aie.buffer(%[[VAL_44]]) {sym_name = "buf39"} : memref<64xi32, 2>
// CHECK:           %[[VAL_51:.*]] = aie.mem(%[[VAL_44]]) {
// CHECK:             %[[VAL_52:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_49]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_50]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_49]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_53:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_47]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_48]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_47]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_54:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_45]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_46]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_45]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_55:.*]] = aie.core(%[[VAL_44]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_49]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_47]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_45]], Acquire, 0)
// CHECK:             affine.for %[[VAL_56:.*]] = 0 to 64 {
// CHECK:               %[[VAL_57:.*]] = affine.load %[[VAL_50]]{{\[}}%[[VAL_56]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_58:.*]] = affine.load %[[VAL_48]]{{\[}}%[[VAL_56]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_59:.*]] = arith.muli %[[VAL_57]], %[[VAL_58]] : i32
// CHECK:               affine.store %[[VAL_59]], %[[VAL_46]]{{\[}}%[[VAL_56]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_45]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_47]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_49]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_60:.*]] = aie.tile(42, 2)
// CHECK:           %[[VAL_61:.*]] = aie.tile(42, 1)
// CHECK:           %[[VAL_62:.*]] = aie.tile(42, 0)
// CHECK:           %[[VAL_63:.*]] = aie.tile(0, 3)
// CHECK:           %[[VAL_64:.*]] = aie.tile(7, 5)
// CHECK:           %[[VAL_65:.*]] = aie.lock(%[[VAL_64]], 2)
// CHECK:           %[[VAL_66:.*]] = aie.buffer(%[[VAL_64]]) {sym_name = "buf38"} : memref<64xi32, 2>
// CHECK:           %[[VAL_67:.*]] = aie.lock(%[[VAL_64]], 1)
// CHECK:           %[[VAL_68:.*]] = aie.buffer(%[[VAL_64]]) {sym_name = "buf37"} : memref<64xi32, 2>
// CHECK:           %[[VAL_69:.*]] = aie.lock(%[[VAL_64]], 0)
// CHECK:           %[[VAL_70:.*]] = aie.buffer(%[[VAL_64]]) {sym_name = "buf36"} : memref<64xi32, 2>
// CHECK:           %[[VAL_71:.*]] = aie.mem(%[[VAL_64]]) {
// CHECK:             %[[VAL_72:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_69]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_70]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_69]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_73:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_67]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_68]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_67]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_74:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_65]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_66]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_65]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_75:.*]] = aie.core(%[[VAL_64]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_69]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_67]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_65]], Acquire, 0)
// CHECK:             affine.for %[[VAL_76:.*]] = 0 to 64 {
// CHECK:               %[[VAL_77:.*]] = affine.load %[[VAL_70]]{{\[}}%[[VAL_76]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_78:.*]] = affine.load %[[VAL_68]]{{\[}}%[[VAL_76]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_79:.*]] = arith.muli %[[VAL_77]], %[[VAL_78]] : i32
// CHECK:               affine.store %[[VAL_79]], %[[VAL_66]]{{\[}}%[[VAL_76]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_65]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_67]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_69]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_80:.*]] = aie.tile(35, 2)
// CHECK:           %[[VAL_81:.*]] = aie.tile(35, 1)
// CHECK:           %[[VAL_82:.*]] = aie.tile(35, 0)
// CHECK:           %[[VAL_83:.*]] = aie.tile(10, 4)
// CHECK:           %[[VAL_84:.*]] = aie.lock(%[[VAL_83]], 2)
// CHECK:           %[[VAL_85:.*]] = aie.buffer(%[[VAL_83]]) {sym_name = "buf35"} : memref<64xi32, 2>
// CHECK:           %[[VAL_86:.*]] = aie.lock(%[[VAL_83]], 1)
// CHECK:           %[[VAL_87:.*]] = aie.buffer(%[[VAL_83]]) {sym_name = "buf34"} : memref<64xi32, 2>
// CHECK:           %[[VAL_88:.*]] = aie.lock(%[[VAL_83]], 0)
// CHECK:           %[[VAL_89:.*]] = aie.buffer(%[[VAL_83]]) {sym_name = "buf33"} : memref<64xi32, 2>
// CHECK:           %[[VAL_90:.*]] = aie.mem(%[[VAL_83]]) {
// CHECK:             %[[VAL_91:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_88]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_89]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_88]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_92:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_86]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_87]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_86]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_93:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_84]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_85]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_84]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_94:.*]] = aie.core(%[[VAL_83]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_88]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_86]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_84]], Acquire, 0)
// CHECK:             affine.for %[[VAL_95:.*]] = 0 to 64 {
// CHECK:               %[[VAL_96:.*]] = affine.load %[[VAL_89]]{{\[}}%[[VAL_95]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_97:.*]] = affine.load %[[VAL_87]]{{\[}}%[[VAL_95]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_98:.*]] = arith.muli %[[VAL_96]], %[[VAL_97]] : i32
// CHECK:               affine.store %[[VAL_98]], %[[VAL_85]]{{\[}}%[[VAL_95]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_84]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_86]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_88]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_99:.*]] = aie.tile(34, 2)
// CHECK:           %[[VAL_100:.*]] = aie.tile(34, 1)
// CHECK:           %[[VAL_101:.*]] = aie.tile(34, 0)
// CHECK:           %[[VAL_102:.*]] = aie.tile(9, 4)
// CHECK:           %[[VAL_103:.*]] = aie.lock(%[[VAL_102]], 2)
// CHECK:           %[[VAL_104:.*]] = aie.buffer(%[[VAL_102]]) {sym_name = "buf32"} : memref<64xi32, 2>
// CHECK:           %[[VAL_105:.*]] = aie.lock(%[[VAL_102]], 1)
// CHECK:           %[[VAL_106:.*]] = aie.buffer(%[[VAL_102]]) {sym_name = "buf31"} : memref<64xi32, 2>
// CHECK:           %[[VAL_107:.*]] = aie.lock(%[[VAL_102]], 0)
// CHECK:           %[[VAL_108:.*]] = aie.buffer(%[[VAL_102]]) {sym_name = "buf30"} : memref<64xi32, 2>
// CHECK:           %[[VAL_109:.*]] = aie.mem(%[[VAL_102]]) {
// CHECK:             %[[VAL_110:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_107]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_108]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_107]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_111:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_105]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_106]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_105]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_112:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_103]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_104]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_103]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_113:.*]] = aie.core(%[[VAL_102]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_107]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_105]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_103]], Acquire, 0)
// CHECK:             affine.for %[[VAL_114:.*]] = 0 to 64 {
// CHECK:               %[[VAL_115:.*]] = affine.load %[[VAL_108]]{{\[}}%[[VAL_114]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_116:.*]] = affine.load %[[VAL_106]]{{\[}}%[[VAL_114]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_117:.*]] = arith.muli %[[VAL_115]], %[[VAL_116]] : i32
// CHECK:               affine.store %[[VAL_117]], %[[VAL_104]]{{\[}}%[[VAL_114]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_103]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_105]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_107]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_118:.*]] = aie.tile(27, 2)
// CHECK:           %[[VAL_119:.*]] = aie.tile(27, 1)
// CHECK:           %[[VAL_120:.*]] = aie.tile(27, 0)
// CHECK:           %[[VAL_121:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_122:.*]] = aie.tile(8, 4)
// CHECK:           %[[VAL_123:.*]] = aie.lock(%[[VAL_122]], 2)
// CHECK:           %[[VAL_124:.*]] = aie.buffer(%[[VAL_122]]) {sym_name = "buf29"} : memref<64xi32, 2>
// CHECK:           %[[VAL_125:.*]] = aie.lock(%[[VAL_122]], 1)
// CHECK:           %[[VAL_126:.*]] = aie.buffer(%[[VAL_122]]) {sym_name = "buf28"} : memref<64xi32, 2>
// CHECK:           %[[VAL_127:.*]] = aie.lock(%[[VAL_122]], 0)
// CHECK:           %[[VAL_128:.*]] = aie.buffer(%[[VAL_122]]) {sym_name = "buf27"} : memref<64xi32, 2>
// CHECK:           %[[VAL_129:.*]] = aie.mem(%[[VAL_122]]) {
// CHECK:             %[[VAL_130:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_127]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_128]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_127]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_131:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_125]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_126]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_125]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_132:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_123]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_124]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_123]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_133:.*]] = aie.core(%[[VAL_122]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_127]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_125]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_123]], Acquire, 0)
// CHECK:             affine.for %[[VAL_134:.*]] = 0 to 64 {
// CHECK:               %[[VAL_135:.*]] = affine.load %[[VAL_128]]{{\[}}%[[VAL_134]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_136:.*]] = affine.load %[[VAL_126]]{{\[}}%[[VAL_134]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_137:.*]] = arith.muli %[[VAL_135]], %[[VAL_136]] : i32
// CHECK:               affine.store %[[VAL_137]], %[[VAL_124]]{{\[}}%[[VAL_134]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_123]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_125]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_127]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_138:.*]] = aie.tile(26, 2)
// CHECK:           %[[VAL_139:.*]] = aie.tile(26, 1)
// CHECK:           %[[VAL_140:.*]] = aie.tile(26, 0)
// CHECK:           %[[VAL_141:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_142:.*]] = aie.tile(7, 4)
// CHECK:           %[[VAL_143:.*]] = aie.lock(%[[VAL_142]], 2)
// CHECK:           %[[VAL_144:.*]] = aie.buffer(%[[VAL_142]]) {sym_name = "buf26"} : memref<64xi32, 2>
// CHECK:           %[[VAL_145:.*]] = aie.lock(%[[VAL_142]], 1)
// CHECK:           %[[VAL_146:.*]] = aie.buffer(%[[VAL_142]]) {sym_name = "buf25"} : memref<64xi32, 2>
// CHECK:           %[[VAL_147:.*]] = aie.lock(%[[VAL_142]], 0)
// CHECK:           %[[VAL_148:.*]] = aie.buffer(%[[VAL_142]]) {sym_name = "buf24"} : memref<64xi32, 2>
// CHECK:           %[[VAL_149:.*]] = aie.mem(%[[VAL_142]]) {
// CHECK:             %[[VAL_150:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_147]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_148]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_147]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_151:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_145]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_146]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_145]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_152:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_143]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_144]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_143]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_153:.*]] = aie.core(%[[VAL_142]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_147]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_145]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_143]], Acquire, 0)
// CHECK:             affine.for %[[VAL_154:.*]] = 0 to 64 {
// CHECK:               %[[VAL_155:.*]] = affine.load %[[VAL_148]]{{\[}}%[[VAL_154]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_156:.*]] = affine.load %[[VAL_146]]{{\[}}%[[VAL_154]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_157:.*]] = arith.muli %[[VAL_155]], %[[VAL_156]] : i32
// CHECK:               affine.store %[[VAL_157]], %[[VAL_144]]{{\[}}%[[VAL_154]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_143]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_145]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_147]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_158:.*]] = aie.tile(19, 2)
// CHECK:           %[[VAL_159:.*]] = aie.tile(19, 1)
// CHECK:           %[[VAL_160:.*]] = aie.tile(19, 0)
// CHECK:           %[[VAL_161:.*]] = aie.tile(10, 3)
// CHECK:           %[[VAL_162:.*]] = aie.lock(%[[VAL_161]], 2)
// CHECK:           %[[VAL_163:.*]] = aie.buffer(%[[VAL_161]]) {sym_name = "buf23"} : memref<64xi32, 2>
// CHECK:           %[[VAL_164:.*]] = aie.lock(%[[VAL_161]], 1)
// CHECK:           %[[VAL_165:.*]] = aie.buffer(%[[VAL_161]]) {sym_name = "buf22"} : memref<64xi32, 2>
// CHECK:           %[[VAL_166:.*]] = aie.lock(%[[VAL_161]], 0)
// CHECK:           %[[VAL_167:.*]] = aie.buffer(%[[VAL_161]]) {sym_name = "buf21"} : memref<64xi32, 2>
// CHECK:           %[[VAL_168:.*]] = aie.mem(%[[VAL_161]]) {
// CHECK:             %[[VAL_169:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_166]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_167]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_166]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_170:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_164]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_165]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_164]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_171:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_162]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_163]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_162]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_172:.*]] = aie.core(%[[VAL_161]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_166]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_164]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_162]], Acquire, 0)
// CHECK:             affine.for %[[VAL_173:.*]] = 0 to 64 {
// CHECK:               %[[VAL_174:.*]] = affine.load %[[VAL_167]]{{\[}}%[[VAL_173]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_175:.*]] = affine.load %[[VAL_165]]{{\[}}%[[VAL_173]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_176:.*]] = arith.muli %[[VAL_174]], %[[VAL_175]] : i32
// CHECK:               affine.store %[[VAL_176]], %[[VAL_163]]{{\[}}%[[VAL_173]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_162]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_164]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_166]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_177:.*]] = aie.tile(18, 2)
// CHECK:           %[[VAL_178:.*]] = aie.tile(18, 1)
// CHECK:           %[[VAL_179:.*]] = aie.tile(18, 0)
// CHECK:           %[[VAL_180:.*]] = aie.tile(9, 3)
// CHECK:           %[[VAL_181:.*]] = aie.lock(%[[VAL_180]], 2)
// CHECK:           %[[VAL_182:.*]] = aie.buffer(%[[VAL_180]]) {sym_name = "buf20"} : memref<64xi32, 2>
// CHECK:           %[[VAL_183:.*]] = aie.lock(%[[VAL_180]], 1)
// CHECK:           %[[VAL_184:.*]] = aie.buffer(%[[VAL_180]]) {sym_name = "buf19"} : memref<64xi32, 2>
// CHECK:           %[[VAL_185:.*]] = aie.lock(%[[VAL_180]], 0)
// CHECK:           %[[VAL_186:.*]] = aie.buffer(%[[VAL_180]]) {sym_name = "buf18"} : memref<64xi32, 2>
// CHECK:           %[[VAL_187:.*]] = aie.mem(%[[VAL_180]]) {
// CHECK:             %[[VAL_188:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_185]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_186]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_185]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_189:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_183]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_184]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_183]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_190:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_181]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_182]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_181]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_191:.*]] = aie.core(%[[VAL_180]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_185]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_183]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_181]], Acquire, 0)
// CHECK:             affine.for %[[VAL_192:.*]] = 0 to 64 {
// CHECK:               %[[VAL_193:.*]] = affine.load %[[VAL_186]]{{\[}}%[[VAL_192]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_194:.*]] = affine.load %[[VAL_184]]{{\[}}%[[VAL_192]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_195:.*]] = arith.muli %[[VAL_193]], %[[VAL_194]] : i32
// CHECK:               affine.store %[[VAL_195]], %[[VAL_182]]{{\[}}%[[VAL_192]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_181]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_183]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_185]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_196:.*]] = aie.tile(11, 2)
// CHECK:           %[[VAL_197:.*]] = aie.tile(11, 1)
// CHECK:           %[[VAL_198:.*]] = aie.tile(11, 0)
// CHECK:           %[[VAL_199:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_200:.*]] = aie.tile(8, 3)
// CHECK:           %[[VAL_201:.*]] = aie.lock(%[[VAL_200]], 2)
// CHECK:           %[[VAL_202:.*]] = aie.buffer(%[[VAL_200]]) {sym_name = "buf17"} : memref<64xi32, 2>
// CHECK:           %[[VAL_203:.*]] = aie.lock(%[[VAL_200]], 1)
// CHECK:           %[[VAL_204:.*]] = aie.buffer(%[[VAL_200]]) {sym_name = "buf16"} : memref<64xi32, 2>
// CHECK:           %[[VAL_205:.*]] = aie.lock(%[[VAL_200]], 0)
// CHECK:           %[[VAL_206:.*]] = aie.buffer(%[[VAL_200]]) {sym_name = "buf15"} : memref<64xi32, 2>
// CHECK:           %[[VAL_207:.*]] = aie.mem(%[[VAL_200]]) {
// CHECK:             %[[VAL_208:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_205]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_206]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_205]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_209:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_203]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_204]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_203]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_210:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_201]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_202]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_201]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_211:.*]] = aie.core(%[[VAL_200]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_205]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_203]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_201]], Acquire, 0)
// CHECK:             affine.for %[[VAL_212:.*]] = 0 to 64 {
// CHECK:               %[[VAL_213:.*]] = affine.load %[[VAL_206]]{{\[}}%[[VAL_212]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_214:.*]] = affine.load %[[VAL_204]]{{\[}}%[[VAL_212]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_215:.*]] = arith.muli %[[VAL_213]], %[[VAL_214]] : i32
// CHECK:               affine.store %[[VAL_215]], %[[VAL_202]]{{\[}}%[[VAL_212]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_201]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_203]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_205]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_216:.*]] = aie.tile(10, 1)
// CHECK:           %[[VAL_217:.*]] = aie.tile(10, 0)
// CHECK:           %[[VAL_218:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_219:.*]] = aie.tile(7, 3)
// CHECK:           %[[VAL_220:.*]] = aie.lock(%[[VAL_219]], 2)
// CHECK:           %[[VAL_221:.*]] = aie.buffer(%[[VAL_219]]) {sym_name = "buf14"} : memref<64xi32, 2>
// CHECK:           %[[VAL_222:.*]] = aie.lock(%[[VAL_219]], 1)
// CHECK:           %[[VAL_223:.*]] = aie.buffer(%[[VAL_219]]) {sym_name = "buf13"} : memref<64xi32, 2>
// CHECK:           %[[VAL_224:.*]] = aie.lock(%[[VAL_219]], 0)
// CHECK:           %[[VAL_225:.*]] = aie.buffer(%[[VAL_219]]) {sym_name = "buf12"} : memref<64xi32, 2>
// CHECK:           %[[VAL_226:.*]] = aie.mem(%[[VAL_219]]) {
// CHECK:             %[[VAL_227:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_224]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_225]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_224]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_228:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_222]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_223]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_222]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_229:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_220]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_221]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_220]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_230:.*]] = aie.core(%[[VAL_219]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_224]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_222]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_220]], Acquire, 0)
// CHECK:             affine.for %[[VAL_231:.*]] = 0 to 64 {
// CHECK:               %[[VAL_232:.*]] = affine.load %[[VAL_225]]{{\[}}%[[VAL_231]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_233:.*]] = affine.load %[[VAL_223]]{{\[}}%[[VAL_231]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_234:.*]] = arith.muli %[[VAL_232]], %[[VAL_233]] : i32
// CHECK:               affine.store %[[VAL_234]], %[[VAL_221]]{{\[}}%[[VAL_231]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_220]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_222]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_224]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_235:.*]] = aie.tile(7, 1)
// CHECK:           %[[VAL_236:.*]] = aie.tile(7, 0)
// CHECK:           %[[VAL_237:.*]] = aie.tile(10, 2)
// CHECK:           %[[VAL_238:.*]] = aie.lock(%[[VAL_237]], 2)
// CHECK:           %[[VAL_239:.*]] = aie.buffer(%[[VAL_237]]) {sym_name = "buf11"} : memref<64xi32, 2>
// CHECK:           %[[VAL_240:.*]] = aie.lock(%[[VAL_237]], 1)
// CHECK:           %[[VAL_241:.*]] = aie.buffer(%[[VAL_237]]) {sym_name = "buf10"} : memref<64xi32, 2>
// CHECK:           %[[VAL_242:.*]] = aie.lock(%[[VAL_237]], 0)
// CHECK:           %[[VAL_243:.*]] = aie.buffer(%[[VAL_237]]) {sym_name = "buf9"} : memref<64xi32, 2>
// CHECK:           %[[VAL_244:.*]] = aie.mem(%[[VAL_237]]) {
// CHECK:             %[[VAL_245:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_242]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_243]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_242]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_246:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_240]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_241]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_240]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_247:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_238]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_239]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_238]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_248:.*]] = aie.core(%[[VAL_237]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_242]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_240]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_238]], Acquire, 0)
// CHECK:             affine.for %[[VAL_249:.*]] = 0 to 64 {
// CHECK:               %[[VAL_250:.*]] = affine.load %[[VAL_243]]{{\[}}%[[VAL_249]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_251:.*]] = affine.load %[[VAL_241]]{{\[}}%[[VAL_249]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_252:.*]] = arith.muli %[[VAL_250]], %[[VAL_251]] : i32
// CHECK:               affine.store %[[VAL_252]], %[[VAL_239]]{{\[}}%[[VAL_249]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_238]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_240]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_242]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_253:.*]] = aie.tile(6, 2)
// CHECK:           %[[VAL_254:.*]] = aie.tile(6, 1)
// CHECK:           %[[VAL_255:.*]] = aie.tile(6, 0)
// CHECK:           %[[VAL_256:.*]] = aie.tile(9, 2)
// CHECK:           %[[VAL_257:.*]] = aie.lock(%[[VAL_256]], 2)
// CHECK:           %[[VAL_258:.*]] = aie.buffer(%[[VAL_256]]) {sym_name = "buf8"} : memref<64xi32, 2>
// CHECK:           %[[VAL_259:.*]] = aie.lock(%[[VAL_256]], 1)
// CHECK:           %[[VAL_260:.*]] = aie.buffer(%[[VAL_256]]) {sym_name = "buf7"} : memref<64xi32, 2>
// CHECK:           %[[VAL_261:.*]] = aie.lock(%[[VAL_256]], 0)
// CHECK:           %[[VAL_262:.*]] = aie.buffer(%[[VAL_256]]) {sym_name = "buf6"} : memref<64xi32, 2>
// CHECK:           %[[VAL_263:.*]] = aie.mem(%[[VAL_256]]) {
// CHECK:             %[[VAL_264:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_261]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_262]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_261]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_265:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_259]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_260]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_259]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_266:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_257]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_258]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_257]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_267:.*]] = aie.core(%[[VAL_256]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_261]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_259]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_257]], Acquire, 0)
// CHECK:             affine.for %[[VAL_268:.*]] = 0 to 64 {
// CHECK:               %[[VAL_269:.*]] = affine.load %[[VAL_262]]{{\[}}%[[VAL_268]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_270:.*]] = affine.load %[[VAL_260]]{{\[}}%[[VAL_268]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_271:.*]] = arith.muli %[[VAL_269]], %[[VAL_270]] : i32
// CHECK:               affine.store %[[VAL_271]], %[[VAL_258]]{{\[}}%[[VAL_268]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_257]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_259]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_261]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_272:.*]] = aie.tile(3, 2)
// CHECK:           %[[VAL_273:.*]] = aie.tile(3, 1)
// CHECK:           %[[VAL_274:.*]] = aie.tile(3, 0)
// CHECK:           %[[VAL_275:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_276:.*]] = aie.tile(8, 2)
// CHECK:           %[[VAL_277:.*]] = aie.lock(%[[VAL_276]], 2)
// CHECK:           %[[VAL_278:.*]] = aie.buffer(%[[VAL_276]]) {sym_name = "buf5"} : memref<64xi32, 2>
// CHECK:           %[[VAL_279:.*]] = aie.lock(%[[VAL_276]], 1)
// CHECK:           %[[VAL_280:.*]] = aie.buffer(%[[VAL_276]]) {sym_name = "buf4"} : memref<64xi32, 2>
// CHECK:           %[[VAL_281:.*]] = aie.lock(%[[VAL_276]], 0)
// CHECK:           %[[VAL_282:.*]] = aie.buffer(%[[VAL_276]]) {sym_name = "buf3"} : memref<64xi32, 2>
// CHECK:           %[[VAL_283:.*]] = aie.mem(%[[VAL_276]]) {
// CHECK:             %[[VAL_284:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_281]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_282]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_281]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_285:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_279]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_280]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_279]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_286:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_277]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_278]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_277]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_287:.*]] = aie.core(%[[VAL_276]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_281]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_279]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_277]], Acquire, 0)
// CHECK:             affine.for %[[VAL_288:.*]] = 0 to 64 {
// CHECK:               %[[VAL_289:.*]] = affine.load %[[VAL_282]]{{\[}}%[[VAL_288]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_290:.*]] = affine.load %[[VAL_280]]{{\[}}%[[VAL_288]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_291:.*]] = arith.muli %[[VAL_289]], %[[VAL_290]] : i32
// CHECK:               affine.store %[[VAL_291]], %[[VAL_278]]{{\[}}%[[VAL_288]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_277]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_279]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_281]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_292:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_293:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_294:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_295:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_296:.*]] = aie.tile(7, 2)
// CHECK:           %[[VAL_297:.*]] = aie.lock(%[[VAL_296]], 2)
// CHECK:           %[[VAL_298:.*]] = aie.buffer(%[[VAL_296]]) {sym_name = "buf2"} : memref<64xi32, 2>
// CHECK:           %[[VAL_299:.*]] = aie.lock(%[[VAL_296]], 1)
// CHECK:           %[[VAL_300:.*]] = aie.buffer(%[[VAL_296]]) {sym_name = "buf1"} : memref<64xi32, 2>
// CHECK:           %[[VAL_301:.*]] = aie.lock(%[[VAL_296]], 0)
// CHECK:           %[[VAL_302:.*]] = aie.buffer(%[[VAL_296]]) {sym_name = "buf0"} : memref<64xi32, 2>
// CHECK:           %[[VAL_303:.*]] = aie.mem(%[[VAL_296]]) {
// CHECK:             %[[VAL_304:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_301]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_302]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_301]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_305:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_299]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_300]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_299]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_306:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_297]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_298]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_297]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_307:.*]] = aie.core(%[[VAL_296]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_301]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_299]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_297]], Acquire, 0)
// CHECK:             affine.for %[[VAL_308:.*]] = 0 to 64 {
// CHECK:               %[[VAL_309:.*]] = affine.load %[[VAL_302]]{{\[}}%[[VAL_308]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_310:.*]] = affine.load %[[VAL_300]]{{\[}}%[[VAL_308]]] : memref<64xi32, 2>
// CHECK:               %[[VAL_311:.*]] = arith.muli %[[VAL_309]], %[[VAL_310]] : i32
// CHECK:               affine.store %[[VAL_311]], %[[VAL_298]]{{\[}}%[[VAL_308]]] : memref<64xi32, 2>
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_297]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_299]], Release, 0)
// CHECK:             aie.use_lock(%[[VAL_301]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[VAL_312:.*]] = aie.switchbox(%[[VAL_294]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_313:.*]] = aie.shim_mux(%[[VAL_294]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_314:.*]] = aie.switchbox(%[[VAL_293]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_315:.*]] = aie.switchbox(%[[VAL_292]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:             aie.connect<East : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_316:.*]] = aie.switchbox(%[[VAL_272]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, South : 0>
// CHECK:             aie.connect<East : 3, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_317:.*]] = aie.tile(4, 2)
// CHECK:           %[[VAL_318:.*]] = aie.switchbox(%[[VAL_317]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_319:.*]] = aie.tile(5, 2)
// CHECK:           %[[VAL_320:.*]] = aie.switchbox(%[[VAL_319]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_321:.*]] = aie.switchbox(%[[VAL_253]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<South : 0, East : 2>
// CHECK:             aie.connect<South : 1, East : 3>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_322:.*]] = aie.switchbox(%[[VAL_296]]) {
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
// CHECK:           %[[VAL_323:.*]] = aie.switchbox(%[[VAL_274]]) {
// CHECK:             aie.connect<South : 3, East : 0>
// CHECK:             aie.connect<South : 7, East : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_324:.*]] = aie.shim_mux(%[[VAL_274]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_325:.*]] = aie.tile(4, 0)
// CHECK:           %[[VAL_326:.*]] = aie.switchbox(%[[VAL_325]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_327:.*]] = aie.tile(5, 0)
// CHECK:           %[[VAL_328:.*]] = aie.switchbox(%[[VAL_327]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_329:.*]] = aie.switchbox(%[[VAL_255]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, North : 1>
// CHECK:             aie.connect<South : 3, East : 0>
// CHECK:             aie.connect<South : 7, East : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_330:.*]] = aie.switchbox(%[[VAL_254]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:             aie.connect<East : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_331:.*]] = aie.switchbox(%[[VAL_276]]) {
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
// CHECK:           %[[VAL_332:.*]] = aie.shim_mux(%[[VAL_255]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_333:.*]] = aie.switchbox(%[[VAL_236]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<South : 3, East : 2>
// CHECK:             aie.connect<South : 7, East : 3>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_334:.*]] = aie.tile(8, 0)
// CHECK:           %[[VAL_335:.*]] = aie.switchbox(%[[VAL_334]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<West : 2, East : 2>
// CHECK:             aie.connect<West : 3, East : 3>
// CHECK:           }
// CHECK:           %[[VAL_336:.*]] = aie.tile(9, 0)
// CHECK:           %[[VAL_337:.*]] = aie.switchbox(%[[VAL_336]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, North : 1>
// CHECK:             aie.connect<West : 2, East : 0>
// CHECK:             aie.connect<West : 3, East : 1>
// CHECK:             aie.connect<East : 0, North : 2>
// CHECK:             aie.connect<East : 1, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_338:.*]] = aie.tile(9, 1)
// CHECK:           %[[VAL_339:.*]] = aie.switchbox(%[[VAL_338]]) {
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
// CHECK:           %[[VAL_340:.*]] = aie.switchbox(%[[VAL_256]]) {
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
// CHECK:           %[[VAL_341:.*]] = aie.switchbox(%[[VAL_273]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_342:.*]] = aie.shim_mux(%[[VAL_236]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_343:.*]] = aie.switchbox(%[[VAL_217]]) {
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
// CHECK:           %[[VAL_344:.*]] = aie.switchbox(%[[VAL_216]]) {
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
// CHECK:           %[[VAL_345:.*]] = aie.switchbox(%[[VAL_237]]) {
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
// CHECK:           %[[VAL_346:.*]] = aie.switchbox(%[[VAL_219]]) {
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<East : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_347:.*]] = aie.tile(8, 1)
// CHECK:           %[[VAL_348:.*]] = aie.switchbox(%[[VAL_347]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<East : 2, West : 1>
// CHECK:             aie.connect<East : 3, West : 2>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<North : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_349:.*]] = aie.switchbox(%[[VAL_200]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<East : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<South : 2, West : 2>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_350:.*]] = aie.shim_mux(%[[VAL_217]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_351:.*]] = aie.switchbox(%[[VAL_235]]) {
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<East : 0, West : 1>
// CHECK:             aie.connect<East : 1, South : 0>
// CHECK:             aie.connect<East : 2, South : 1>
// CHECK:             aie.connect<North : 1, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_352:.*]] = aie.switchbox(%[[VAL_180]]) {
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
// CHECK:           %[[VAL_353:.*]] = aie.switchbox(%[[VAL_198]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_354:.*]] = aie.shim_mux(%[[VAL_198]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_355:.*]] = aie.switchbox(%[[VAL_161]]) {
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
// CHECK:           %[[VAL_356:.*]] = aie.tile(12, 0)
// CHECK:           %[[VAL_357:.*]] = aie.switchbox(%[[VAL_356]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_358:.*]] = aie.tile(13, 0)
// CHECK:           %[[VAL_359:.*]] = aie.switchbox(%[[VAL_358]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_360:.*]] = aie.tile(14, 0)
// CHECK:           %[[VAL_361:.*]] = aie.switchbox(%[[VAL_360]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_362:.*]] = aie.tile(15, 0)
// CHECK:           %[[VAL_363:.*]] = aie.switchbox(%[[VAL_362]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_364:.*]] = aie.tile(16, 0)
// CHECK:           %[[VAL_365:.*]] = aie.switchbox(%[[VAL_364]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_366:.*]] = aie.tile(17, 0)
// CHECK:           %[[VAL_367:.*]] = aie.switchbox(%[[VAL_366]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_368:.*]] = aie.switchbox(%[[VAL_179]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_369:.*]] = aie.shim_mux(%[[VAL_179]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_370:.*]] = aie.switchbox(%[[VAL_197]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, North : 2>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<West : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_371:.*]] = aie.switchbox(%[[VAL_196]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<South : 2, West : 0>
// CHECK:             aie.connect<East : 0, West : 1>
// CHECK:             aie.connect<East : 1, West : 2>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_372:.*]] = aie.tile(11, 3)
// CHECK:           %[[VAL_373:.*]] = aie.switchbox(%[[VAL_372]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_374:.*]] = aie.switchbox(%[[VAL_160]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_375:.*]] = aie.shim_mux(%[[VAL_160]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_376:.*]] = aie.switchbox(%[[VAL_142]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<East : 0, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_377:.*]] = aie.tile(12, 1)
// CHECK:           %[[VAL_378:.*]] = aie.switchbox(%[[VAL_377]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_379:.*]] = aie.tile(13, 1)
// CHECK:           %[[VAL_380:.*]] = aie.switchbox(%[[VAL_379]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_381:.*]] = aie.tile(14, 1)
// CHECK:           %[[VAL_382:.*]] = aie.switchbox(%[[VAL_381]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_383:.*]] = aie.tile(15, 1)
// CHECK:           %[[VAL_384:.*]] = aie.switchbox(%[[VAL_383]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_385:.*]] = aie.tile(16, 1)
// CHECK:           %[[VAL_386:.*]] = aie.switchbox(%[[VAL_385]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_387:.*]] = aie.tile(17, 1)
// CHECK:           %[[VAL_388:.*]] = aie.switchbox(%[[VAL_387]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:             aie.connect<North : 1, East : 1>
// CHECK:             aie.connect<West : 0, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_389:.*]] = aie.switchbox(%[[VAL_178]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<West : 1, South : 1>
// CHECK:             aie.connect<West : 2, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_390:.*]] = aie.tile(20, 0)
// CHECK:           %[[VAL_391:.*]] = aie.switchbox(%[[VAL_390]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_392:.*]] = aie.tile(21, 0)
// CHECK:           %[[VAL_393:.*]] = aie.switchbox(%[[VAL_392]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_394:.*]] = aie.tile(22, 0)
// CHECK:           %[[VAL_395:.*]] = aie.switchbox(%[[VAL_394]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_396:.*]] = aie.tile(23, 0)
// CHECK:           %[[VAL_397:.*]] = aie.switchbox(%[[VAL_396]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_398:.*]] = aie.tile(24, 0)
// CHECK:           %[[VAL_399:.*]] = aie.switchbox(%[[VAL_398]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_400:.*]] = aie.tile(25, 0)
// CHECK:           %[[VAL_401:.*]] = aie.switchbox(%[[VAL_400]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_402:.*]] = aie.switchbox(%[[VAL_140]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_403:.*]] = aie.shim_mux(%[[VAL_140]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_404:.*]] = aie.switchbox(%[[VAL_122]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<East : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_405:.*]] = aie.switchbox(%[[VAL_102]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<South : 2, DMA : 0>
// CHECK:             aie.connect<South : 3, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<West : 0, South : 1>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_406:.*]] = aie.switchbox(%[[VAL_159]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<West : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_407:.*]] = aie.switchbox(%[[VAL_120]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_408:.*]] = aie.shim_mux(%[[VAL_120]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_409:.*]] = aie.tile(12, 2)
// CHECK:           %[[VAL_410:.*]] = aie.switchbox(%[[VAL_409]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_411:.*]] = aie.tile(13, 2)
// CHECK:           %[[VAL_412:.*]] = aie.switchbox(%[[VAL_411]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_413:.*]] = aie.tile(14, 2)
// CHECK:           %[[VAL_414:.*]] = aie.switchbox(%[[VAL_413]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_415:.*]] = aie.tile(15, 2)
// CHECK:           %[[VAL_416:.*]] = aie.switchbox(%[[VAL_415]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<North : 0, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_417:.*]] = aie.tile(16, 2)
// CHECK:           %[[VAL_418:.*]] = aie.switchbox(%[[VAL_417]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_419:.*]] = aie.tile(17, 2)
// CHECK:           %[[VAL_420:.*]] = aie.switchbox(%[[VAL_419]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:             aie.connect<West : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_421:.*]] = aie.switchbox(%[[VAL_177]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_422:.*]] = aie.switchbox(%[[VAL_158]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_423:.*]] = aie.tile(20, 1)
// CHECK:           %[[VAL_424:.*]] = aie.switchbox(%[[VAL_423]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:             aie.connect<East : 2, North : 2>
// CHECK:             aie.connect<East : 3, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_425:.*]] = aie.tile(20, 2)
// CHECK:           %[[VAL_426:.*]] = aie.switchbox(%[[VAL_425]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<South : 2, North : 0>
// CHECK:             aie.connect<South : 3, North : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_427:.*]] = aie.tile(21, 1)
// CHECK:           %[[VAL_428:.*]] = aie.switchbox(%[[VAL_427]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_429:.*]] = aie.tile(22, 1)
// CHECK:           %[[VAL_430:.*]] = aie.switchbox(%[[VAL_429]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_431:.*]] = aie.tile(23, 1)
// CHECK:           %[[VAL_432:.*]] = aie.switchbox(%[[VAL_431]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_433:.*]] = aie.tile(24, 1)
// CHECK:           %[[VAL_434:.*]] = aie.switchbox(%[[VAL_433]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_435:.*]] = aie.tile(25, 1)
// CHECK:           %[[VAL_436:.*]] = aie.switchbox(%[[VAL_435]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_437:.*]] = aie.switchbox(%[[VAL_139]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_438:.*]] = aie.tile(28, 0)
// CHECK:           %[[VAL_439:.*]] = aie.switchbox(%[[VAL_438]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_440:.*]] = aie.tile(29, 0)
// CHECK:           %[[VAL_441:.*]] = aie.switchbox(%[[VAL_440]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_442:.*]] = aie.tile(30, 0)
// CHECK:           %[[VAL_443:.*]] = aie.switchbox(%[[VAL_442]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_444:.*]] = aie.tile(31, 0)
// CHECK:           %[[VAL_445:.*]] = aie.switchbox(%[[VAL_444]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_446:.*]] = aie.tile(32, 0)
// CHECK:           %[[VAL_447:.*]] = aie.switchbox(%[[VAL_446]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_448:.*]] = aie.tile(33, 0)
// CHECK:           %[[VAL_449:.*]] = aie.switchbox(%[[VAL_448]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_450:.*]] = aie.switchbox(%[[VAL_101]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_451:.*]] = aie.shim_mux(%[[VAL_101]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_452:.*]] = aie.switchbox(%[[VAL_83]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:             aie.connect<East : 2, West : 0>
// CHECK:             aie.connect<East : 3, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_453:.*]] = aie.tile(12, 3)
// CHECK:           %[[VAL_454:.*]] = aie.switchbox(%[[VAL_453]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_455:.*]] = aie.tile(13, 3)
// CHECK:           %[[VAL_456:.*]] = aie.switchbox(%[[VAL_455]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_457:.*]] = aie.tile(14, 3)
// CHECK:           %[[VAL_458:.*]] = aie.switchbox(%[[VAL_457]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_459:.*]] = aie.tile(15, 3)
// CHECK:           %[[VAL_460:.*]] = aie.switchbox(%[[VAL_459]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_461:.*]] = aie.tile(16, 3)
// CHECK:           %[[VAL_462:.*]] = aie.switchbox(%[[VAL_461]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_463:.*]] = aie.tile(17, 3)
// CHECK:           %[[VAL_464:.*]] = aie.switchbox(%[[VAL_463]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_465:.*]] = aie.tile(18, 3)
// CHECK:           %[[VAL_466:.*]] = aie.switchbox(%[[VAL_465]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_467:.*]] = aie.tile(19, 3)
// CHECK:           %[[VAL_468:.*]] = aie.switchbox(%[[VAL_467]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_469:.*]] = aie.tile(20, 3)
// CHECK:           %[[VAL_470:.*]] = aie.switchbox(%[[VAL_469]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_471:.*]] = aie.switchbox(%[[VAL_119]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_472:.*]] = aie.switchbox(%[[VAL_82]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_473:.*]] = aie.shim_mux(%[[VAL_82]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_474:.*]] = aie.switchbox(%[[VAL_64]]) {
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<East : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_475:.*]] = aie.switchbox(%[[VAL_44]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 2, DMA : 0>
// CHECK:             aie.connect<East : 3, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_476:.*]] = aie.switchbox(%[[VAL_24]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_477:.*]] = aie.switchbox(%[[VAL_4]]) {
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
// CHECK:           %[[VAL_478:.*]] = aie.tile(11, 5)
// CHECK:           %[[VAL_479:.*]] = aie.switchbox(%[[VAL_478]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<West : 2, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_480:.*]] = aie.tile(12, 5)
// CHECK:           %[[VAL_481:.*]] = aie.switchbox(%[[VAL_480]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<West : 2, East : 2>
// CHECK:           }
// CHECK:           %[[VAL_482:.*]] = aie.tile(13, 5)
// CHECK:           %[[VAL_483:.*]] = aie.switchbox(%[[VAL_482]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<West : 2, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_484:.*]] = aie.tile(14, 5)
// CHECK:           %[[VAL_485:.*]] = aie.switchbox(%[[VAL_484]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_486:.*]] = aie.tile(15, 4)
// CHECK:           %[[VAL_487:.*]] = aie.switchbox(%[[VAL_486]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_488:.*]] = aie.tile(15, 5)
// CHECK:           %[[VAL_489:.*]] = aie.switchbox(%[[VAL_488]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<West : 1, East : 0>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_490:.*]] = aie.tile(21, 3)
// CHECK:           %[[VAL_491:.*]] = aie.switchbox(%[[VAL_490]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_492:.*]] = aie.tile(22, 3)
// CHECK:           %[[VAL_493:.*]] = aie.switchbox(%[[VAL_492]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_494:.*]] = aie.tile(23, 3)
// CHECK:           %[[VAL_495:.*]] = aie.switchbox(%[[VAL_494]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_496:.*]] = aie.tile(24, 2)
// CHECK:           %[[VAL_497:.*]] = aie.switchbox(%[[VAL_496]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:             aie.connect<East : 2, West : 0>
// CHECK:             aie.connect<East : 3, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_498:.*]] = aie.tile(24, 3)
// CHECK:           %[[VAL_499:.*]] = aie.switchbox(%[[VAL_498]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_500:.*]] = aie.tile(25, 2)
// CHECK:           %[[VAL_501:.*]] = aie.switchbox(%[[VAL_500]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_502:.*]] = aie.switchbox(%[[VAL_138]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
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
// CHECK:           %[[VAL_515:.*]] = aie.switchbox(%[[VAL_100]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_516:.*]] = aie.tile(36, 0)
// CHECK:           %[[VAL_517:.*]] = aie.switchbox(%[[VAL_516]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_518:.*]] = aie.tile(37, 0)
// CHECK:           %[[VAL_519:.*]] = aie.switchbox(%[[VAL_518]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_520:.*]] = aie.tile(38, 0)
// CHECK:           %[[VAL_521:.*]] = aie.switchbox(%[[VAL_520]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_522:.*]] = aie.tile(39, 0)
// CHECK:           %[[VAL_523:.*]] = aie.switchbox(%[[VAL_522]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_524:.*]] = aie.tile(40, 0)
// CHECK:           %[[VAL_525:.*]] = aie.switchbox(%[[VAL_524]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_526:.*]] = aie.tile(41, 0)
// CHECK:           %[[VAL_527:.*]] = aie.switchbox(%[[VAL_526]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_528:.*]] = aie.switchbox(%[[VAL_62]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_529:.*]] = aie.shim_mux(%[[VAL_62]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_530:.*]] = aie.tile(11, 4)
// CHECK:           %[[VAL_531:.*]] = aie.switchbox(%[[VAL_530]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_532:.*]] = aie.tile(12, 4)
// CHECK:           %[[VAL_533:.*]] = aie.switchbox(%[[VAL_532]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_534:.*]] = aie.tile(13, 4)
// CHECK:           %[[VAL_535:.*]] = aie.switchbox(%[[VAL_534]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_536:.*]] = aie.tile(14, 4)
// CHECK:           %[[VAL_537:.*]] = aie.switchbox(%[[VAL_536]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_538:.*]] = aie.tile(16, 4)
// CHECK:           %[[VAL_539:.*]] = aie.switchbox(%[[VAL_538]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_540:.*]] = aie.tile(17, 4)
// CHECK:           %[[VAL_541:.*]] = aie.switchbox(%[[VAL_540]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_542:.*]] = aie.tile(21, 2)
// CHECK:           %[[VAL_543:.*]] = aie.switchbox(%[[VAL_542]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_544:.*]] = aie.tile(22, 2)
// CHECK:           %[[VAL_545:.*]] = aie.switchbox(%[[VAL_544]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_546:.*]] = aie.tile(23, 2)
// CHECK:           %[[VAL_547:.*]] = aie.switchbox(%[[VAL_546]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_548:.*]] = aie.switchbox(%[[VAL_118]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_549:.*]] = aie.switchbox(%[[VAL_81]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_550:.*]] = aie.switchbox(%[[VAL_42]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_551:.*]] = aie.shim_mux(%[[VAL_42]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_552:.*]] = aie.tile(18, 4)
// CHECK:           %[[VAL_553:.*]] = aie.switchbox(%[[VAL_552]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_554:.*]] = aie.tile(19, 4)
// CHECK:           %[[VAL_555:.*]] = aie.switchbox(%[[VAL_554]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_556:.*]] = aie.tile(20, 4)
// CHECK:           %[[VAL_557:.*]] = aie.switchbox(%[[VAL_556]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_558:.*]] = aie.tile(25, 3)
// CHECK:           %[[VAL_559:.*]] = aie.switchbox(%[[VAL_558]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_560:.*]] = aie.tile(26, 3)
// CHECK:           %[[VAL_561:.*]] = aie.switchbox(%[[VAL_560]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_562:.*]] = aie.tile(28, 2)
// CHECK:           %[[VAL_563:.*]] = aie.switchbox(%[[VAL_562]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_564:.*]] = aie.tile(29, 2)
// CHECK:           %[[VAL_565:.*]] = aie.switchbox(%[[VAL_564]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_566:.*]] = aie.tile(30, 2)
// CHECK:           %[[VAL_567:.*]] = aie.switchbox(%[[VAL_566]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_568:.*]] = aie.tile(31, 2)
// CHECK:           %[[VAL_569:.*]] = aie.switchbox(%[[VAL_568]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_570:.*]] = aie.tile(32, 2)
// CHECK:           %[[VAL_571:.*]] = aie.switchbox(%[[VAL_570]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_572:.*]] = aie.tile(33, 2)
// CHECK:           %[[VAL_573:.*]] = aie.switchbox(%[[VAL_572]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_574:.*]] = aie.switchbox(%[[VAL_99]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_575:.*]] = aie.switchbox(%[[VAL_80]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_576:.*]] = aie.tile(36, 2)
// CHECK:           %[[VAL_577:.*]] = aie.switchbox(%[[VAL_576]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_578:.*]] = aie.tile(37, 2)
// CHECK:           %[[VAL_579:.*]] = aie.switchbox(%[[VAL_578]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_580:.*]] = aie.tile(38, 2)
// CHECK:           %[[VAL_581:.*]] = aie.switchbox(%[[VAL_580]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_582:.*]] = aie.tile(39, 2)
// CHECK:           %[[VAL_583:.*]] = aie.switchbox(%[[VAL_582]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_584:.*]] = aie.tile(40, 2)
// CHECK:           %[[VAL_585:.*]] = aie.switchbox(%[[VAL_584]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_586:.*]] = aie.tile(41, 1)
// CHECK:           %[[VAL_587:.*]] = aie.switchbox(%[[VAL_586]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_588:.*]] = aie.tile(41, 2)
// CHECK:           %[[VAL_589:.*]] = aie.switchbox(%[[VAL_588]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_590:.*]] = aie.switchbox(%[[VAL_61]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_591:.*]] = aie.tile(44, 0)
// CHECK:           %[[VAL_592:.*]] = aie.switchbox(%[[VAL_591]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_593:.*]] = aie.tile(45, 0)
// CHECK:           %[[VAL_594:.*]] = aie.switchbox(%[[VAL_593]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_595:.*]] = aie.switchbox(%[[VAL_22]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_596:.*]] = aie.shim_mux(%[[VAL_22]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_597:.*]] = aie.tile(16, 5)
// CHECK:           %[[VAL_598:.*]] = aie.switchbox(%[[VAL_597]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_599:.*]] = aie.tile(17, 5)
// CHECK:           %[[VAL_600:.*]] = aie.switchbox(%[[VAL_599]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_601:.*]] = aie.tile(18, 5)
// CHECK:           %[[VAL_602:.*]] = aie.switchbox(%[[VAL_601]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_603:.*]] = aie.tile(21, 4)
// CHECK:           %[[VAL_604:.*]] = aie.switchbox(%[[VAL_603]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_605:.*]] = aie.tile(22, 4)
// CHECK:           %[[VAL_606:.*]] = aie.switchbox(%[[VAL_605]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_607:.*]] = aie.tile(23, 4)
// CHECK:           %[[VAL_608:.*]] = aie.switchbox(%[[VAL_607]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_609:.*]] = aie.tile(24, 4)
// CHECK:           %[[VAL_610:.*]] = aie.switchbox(%[[VAL_609]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_611:.*]] = aie.tile(25, 4)
// CHECK:           %[[VAL_612:.*]] = aie.switchbox(%[[VAL_611]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_613:.*]] = aie.tile(26, 4)
// CHECK:           %[[VAL_614:.*]] = aie.switchbox(%[[VAL_613]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_615:.*]] = aie.tile(27, 4)
// CHECK:           %[[VAL_616:.*]] = aie.switchbox(%[[VAL_615]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_617:.*]] = aie.tile(28, 4)
// CHECK:           %[[VAL_618:.*]] = aie.switchbox(%[[VAL_617]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_619:.*]] = aie.tile(29, 4)
// CHECK:           %[[VAL_620:.*]] = aie.switchbox(%[[VAL_619]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_621:.*]] = aie.tile(30, 4)
// CHECK:           %[[VAL_622:.*]] = aie.switchbox(%[[VAL_621]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_623:.*]] = aie.tile(31, 4)
// CHECK:           %[[VAL_624:.*]] = aie.switchbox(%[[VAL_623]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_625:.*]] = aie.tile(32, 4)
// CHECK:           %[[VAL_626:.*]] = aie.switchbox(%[[VAL_625]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_627:.*]] = aie.tile(33, 4)
// CHECK:           %[[VAL_628:.*]] = aie.switchbox(%[[VAL_627]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_629:.*]] = aie.tile(34, 4)
// CHECK:           %[[VAL_630:.*]] = aie.switchbox(%[[VAL_629]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_631:.*]] = aie.tile(35, 4)
// CHECK:           %[[VAL_632:.*]] = aie.switchbox(%[[VAL_631]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_633:.*]] = aie.tile(36, 4)
// CHECK:           %[[VAL_634:.*]] = aie.switchbox(%[[VAL_633]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_635:.*]] = aie.tile(37, 4)
// CHECK:           %[[VAL_636:.*]] = aie.switchbox(%[[VAL_635]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_637:.*]] = aie.tile(38, 4)
// CHECK:           %[[VAL_638:.*]] = aie.switchbox(%[[VAL_637]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_639:.*]] = aie.tile(39, 4)
// CHECK:           %[[VAL_640:.*]] = aie.switchbox(%[[VAL_639]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_641:.*]] = aie.tile(40, 4)
// CHECK:           %[[VAL_642:.*]] = aie.switchbox(%[[VAL_641]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_643:.*]] = aie.tile(41, 4)
// CHECK:           %[[VAL_644:.*]] = aie.switchbox(%[[VAL_643]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_645:.*]] = aie.tile(42, 3)
// CHECK:           %[[VAL_646:.*]] = aie.switchbox(%[[VAL_645]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_647:.*]] = aie.tile(42, 4)
// CHECK:           %[[VAL_648:.*]] = aie.switchbox(%[[VAL_647]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_649:.*]] = aie.tile(43, 3)
// CHECK:           %[[VAL_650:.*]] = aie.switchbox(%[[VAL_649]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_651:.*]] = aie.tile(44, 3)
// CHECK:           %[[VAL_652:.*]] = aie.switchbox(%[[VAL_651]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_653:.*]] = aie.tile(45, 2)
// CHECK:           %[[VAL_654:.*]] = aie.switchbox(%[[VAL_653]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_655:.*]] = aie.tile(45, 3)
// CHECK:           %[[VAL_656:.*]] = aie.switchbox(%[[VAL_655]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_657:.*]] = aie.switchbox(%[[VAL_21]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_658:.*]] = aie.switchbox(%[[VAL_20]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_659:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_660:.*]] = aie.shim_mux(%[[VAL_2]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           aie.wire(%[[VAL_661:.*]] : North, %[[VAL_662:.*]] : South)
// CHECK:           aie.wire(%[[VAL_294]] : DMA, %[[VAL_661]] : DMA)
// CHECK:           aie.wire(%[[VAL_293]] : Core, %[[VAL_663:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_293]] : DMA, %[[VAL_663]] : DMA)
// CHECK:           aie.wire(%[[VAL_662]] : North, %[[VAL_663]] : South)
// CHECK:           aie.wire(%[[VAL_292]] : Core, %[[VAL_664:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_292]] : DMA, %[[VAL_664]] : DMA)
// CHECK:           aie.wire(%[[VAL_663]] : North, %[[VAL_664]] : South)
// CHECK:           aie.wire(%[[VAL_662]] : East, %[[VAL_665:.*]] : West)
// CHECK:           aie.wire(%[[VAL_666:.*]] : North, %[[VAL_665]] : South)
// CHECK:           aie.wire(%[[VAL_274]] : DMA, %[[VAL_666]] : DMA)
// CHECK:           aie.wire(%[[VAL_663]] : East, %[[VAL_667:.*]] : West)
// CHECK:           aie.wire(%[[VAL_273]] : Core, %[[VAL_667]] : Core)
// CHECK:           aie.wire(%[[VAL_273]] : DMA, %[[VAL_667]] : DMA)
// CHECK:           aie.wire(%[[VAL_665]] : North, %[[VAL_667]] : South)
// CHECK:           aie.wire(%[[VAL_664]] : East, %[[VAL_668:.*]] : West)
// CHECK:           aie.wire(%[[VAL_272]] : Core, %[[VAL_668]] : Core)
// CHECK:           aie.wire(%[[VAL_272]] : DMA, %[[VAL_668]] : DMA)
// CHECK:           aie.wire(%[[VAL_667]] : North, %[[VAL_668]] : South)
// CHECK:           aie.wire(%[[VAL_665]] : East, %[[VAL_669:.*]] : West)
// CHECK:           aie.wire(%[[VAL_668]] : East, %[[VAL_670:.*]] : West)
// CHECK:           aie.wire(%[[VAL_317]] : Core, %[[VAL_670]] : Core)
// CHECK:           aie.wire(%[[VAL_317]] : DMA, %[[VAL_670]] : DMA)
// CHECK:           aie.wire(%[[VAL_669]] : East, %[[VAL_671:.*]] : West)
// CHECK:           aie.wire(%[[VAL_670]] : East, %[[VAL_672:.*]] : West)
// CHECK:           aie.wire(%[[VAL_319]] : Core, %[[VAL_672]] : Core)
// CHECK:           aie.wire(%[[VAL_319]] : DMA, %[[VAL_672]] : DMA)
// CHECK:           aie.wire(%[[VAL_671]] : East, %[[VAL_673:.*]] : West)
// CHECK:           aie.wire(%[[VAL_674:.*]] : North, %[[VAL_673]] : South)
// CHECK:           aie.wire(%[[VAL_255]] : DMA, %[[VAL_674]] : DMA)
// CHECK:           aie.wire(%[[VAL_254]] : Core, %[[VAL_675:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_254]] : DMA, %[[VAL_675]] : DMA)
// CHECK:           aie.wire(%[[VAL_673]] : North, %[[VAL_675]] : South)
// CHECK:           aie.wire(%[[VAL_672]] : East, %[[VAL_676:.*]] : West)
// CHECK:           aie.wire(%[[VAL_253]] : Core, %[[VAL_676]] : Core)
// CHECK:           aie.wire(%[[VAL_253]] : DMA, %[[VAL_676]] : DMA)
// CHECK:           aie.wire(%[[VAL_675]] : North, %[[VAL_676]] : South)
// CHECK:           aie.wire(%[[VAL_673]] : East, %[[VAL_677:.*]] : West)
// CHECK:           aie.wire(%[[VAL_678:.*]] : North, %[[VAL_677]] : South)
// CHECK:           aie.wire(%[[VAL_236]] : DMA, %[[VAL_678]] : DMA)
// CHECK:           aie.wire(%[[VAL_675]] : East, %[[VAL_679:.*]] : West)
// CHECK:           aie.wire(%[[VAL_235]] : Core, %[[VAL_679]] : Core)
// CHECK:           aie.wire(%[[VAL_235]] : DMA, %[[VAL_679]] : DMA)
// CHECK:           aie.wire(%[[VAL_677]] : North, %[[VAL_679]] : South)
// CHECK:           aie.wire(%[[VAL_676]] : East, %[[VAL_680:.*]] : West)
// CHECK:           aie.wire(%[[VAL_296]] : Core, %[[VAL_680]] : Core)
// CHECK:           aie.wire(%[[VAL_296]] : DMA, %[[VAL_680]] : DMA)
// CHECK:           aie.wire(%[[VAL_679]] : North, %[[VAL_680]] : South)
// CHECK:           aie.wire(%[[VAL_219]] : Core, %[[VAL_681:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_219]] : DMA, %[[VAL_681]] : DMA)
// CHECK:           aie.wire(%[[VAL_680]] : North, %[[VAL_681]] : South)
// CHECK:           aie.wire(%[[VAL_142]] : Core, %[[VAL_682:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_142]] : DMA, %[[VAL_682]] : DMA)
// CHECK:           aie.wire(%[[VAL_681]] : North, %[[VAL_682]] : South)
// CHECK:           aie.wire(%[[VAL_64]] : Core, %[[VAL_683:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_64]] : DMA, %[[VAL_683]] : DMA)
// CHECK:           aie.wire(%[[VAL_682]] : North, %[[VAL_683]] : South)
// CHECK:           aie.wire(%[[VAL_677]] : East, %[[VAL_684:.*]] : West)
// CHECK:           aie.wire(%[[VAL_679]] : East, %[[VAL_685:.*]] : West)
// CHECK:           aie.wire(%[[VAL_347]] : Core, %[[VAL_685]] : Core)
// CHECK:           aie.wire(%[[VAL_347]] : DMA, %[[VAL_685]] : DMA)
// CHECK:           aie.wire(%[[VAL_684]] : North, %[[VAL_685]] : South)
// CHECK:           aie.wire(%[[VAL_680]] : East, %[[VAL_686:.*]] : West)
// CHECK:           aie.wire(%[[VAL_276]] : Core, %[[VAL_686]] : Core)
// CHECK:           aie.wire(%[[VAL_276]] : DMA, %[[VAL_686]] : DMA)
// CHECK:           aie.wire(%[[VAL_685]] : North, %[[VAL_686]] : South)
// CHECK:           aie.wire(%[[VAL_681]] : East, %[[VAL_687:.*]] : West)
// CHECK:           aie.wire(%[[VAL_200]] : Core, %[[VAL_687]] : Core)
// CHECK:           aie.wire(%[[VAL_200]] : DMA, %[[VAL_687]] : DMA)
// CHECK:           aie.wire(%[[VAL_686]] : North, %[[VAL_687]] : South)
// CHECK:           aie.wire(%[[VAL_682]] : East, %[[VAL_688:.*]] : West)
// CHECK:           aie.wire(%[[VAL_122]] : Core, %[[VAL_688]] : Core)
// CHECK:           aie.wire(%[[VAL_122]] : DMA, %[[VAL_688]] : DMA)
// CHECK:           aie.wire(%[[VAL_687]] : North, %[[VAL_688]] : South)
// CHECK:           aie.wire(%[[VAL_683]] : East, %[[VAL_689:.*]] : West)
// CHECK:           aie.wire(%[[VAL_44]] : Core, %[[VAL_689]] : Core)
// CHECK:           aie.wire(%[[VAL_44]] : DMA, %[[VAL_689]] : DMA)
// CHECK:           aie.wire(%[[VAL_688]] : North, %[[VAL_689]] : South)
// CHECK:           aie.wire(%[[VAL_684]] : East, %[[VAL_690:.*]] : West)
// CHECK:           aie.wire(%[[VAL_685]] : East, %[[VAL_691:.*]] : West)
// CHECK:           aie.wire(%[[VAL_338]] : Core, %[[VAL_691]] : Core)
// CHECK:           aie.wire(%[[VAL_338]] : DMA, %[[VAL_691]] : DMA)
// CHECK:           aie.wire(%[[VAL_690]] : North, %[[VAL_691]] : South)
// CHECK:           aie.wire(%[[VAL_686]] : East, %[[VAL_692:.*]] : West)
// CHECK:           aie.wire(%[[VAL_256]] : Core, %[[VAL_692]] : Core)
// CHECK:           aie.wire(%[[VAL_256]] : DMA, %[[VAL_692]] : DMA)
// CHECK:           aie.wire(%[[VAL_691]] : North, %[[VAL_692]] : South)
// CHECK:           aie.wire(%[[VAL_687]] : East, %[[VAL_693:.*]] : West)
// CHECK:           aie.wire(%[[VAL_180]] : Core, %[[VAL_693]] : Core)
// CHECK:           aie.wire(%[[VAL_180]] : DMA, %[[VAL_693]] : DMA)
// CHECK:           aie.wire(%[[VAL_692]] : North, %[[VAL_693]] : South)
// CHECK:           aie.wire(%[[VAL_688]] : East, %[[VAL_694:.*]] : West)
// CHECK:           aie.wire(%[[VAL_102]] : Core, %[[VAL_694]] : Core)
// CHECK:           aie.wire(%[[VAL_102]] : DMA, %[[VAL_694]] : DMA)
// CHECK:           aie.wire(%[[VAL_693]] : North, %[[VAL_694]] : South)
// CHECK:           aie.wire(%[[VAL_689]] : East, %[[VAL_695:.*]] : West)
// CHECK:           aie.wire(%[[VAL_24]] : Core, %[[VAL_695]] : Core)
// CHECK:           aie.wire(%[[VAL_24]] : DMA, %[[VAL_695]] : DMA)
// CHECK:           aie.wire(%[[VAL_694]] : North, %[[VAL_695]] : South)
// CHECK:           aie.wire(%[[VAL_690]] : East, %[[VAL_696:.*]] : West)
// CHECK:           aie.wire(%[[VAL_697:.*]] : North, %[[VAL_696]] : South)
// CHECK:           aie.wire(%[[VAL_217]] : DMA, %[[VAL_697]] : DMA)
// CHECK:           aie.wire(%[[VAL_691]] : East, %[[VAL_698:.*]] : West)
// CHECK:           aie.wire(%[[VAL_216]] : Core, %[[VAL_698]] : Core)
// CHECK:           aie.wire(%[[VAL_216]] : DMA, %[[VAL_698]] : DMA)
// CHECK:           aie.wire(%[[VAL_696]] : North, %[[VAL_698]] : South)
// CHECK:           aie.wire(%[[VAL_692]] : East, %[[VAL_699:.*]] : West)
// CHECK:           aie.wire(%[[VAL_237]] : Core, %[[VAL_699]] : Core)
// CHECK:           aie.wire(%[[VAL_237]] : DMA, %[[VAL_699]] : DMA)
// CHECK:           aie.wire(%[[VAL_698]] : North, %[[VAL_699]] : South)
// CHECK:           aie.wire(%[[VAL_693]] : East, %[[VAL_700:.*]] : West)
// CHECK:           aie.wire(%[[VAL_161]] : Core, %[[VAL_700]] : Core)
// CHECK:           aie.wire(%[[VAL_161]] : DMA, %[[VAL_700]] : DMA)
// CHECK:           aie.wire(%[[VAL_699]] : North, %[[VAL_700]] : South)
// CHECK:           aie.wire(%[[VAL_694]] : East, %[[VAL_701:.*]] : West)
// CHECK:           aie.wire(%[[VAL_83]] : Core, %[[VAL_701]] : Core)
// CHECK:           aie.wire(%[[VAL_83]] : DMA, %[[VAL_701]] : DMA)
// CHECK:           aie.wire(%[[VAL_700]] : North, %[[VAL_701]] : South)
// CHECK:           aie.wire(%[[VAL_695]] : East, %[[VAL_702:.*]] : West)
// CHECK:           aie.wire(%[[VAL_4]] : Core, %[[VAL_702]] : Core)
// CHECK:           aie.wire(%[[VAL_4]] : DMA, %[[VAL_702]] : DMA)
// CHECK:           aie.wire(%[[VAL_701]] : North, %[[VAL_702]] : South)
// CHECK:           aie.wire(%[[VAL_696]] : East, %[[VAL_703:.*]] : West)
// CHECK:           aie.wire(%[[VAL_704:.*]] : North, %[[VAL_703]] : South)
// CHECK:           aie.wire(%[[VAL_198]] : DMA, %[[VAL_704]] : DMA)
// CHECK:           aie.wire(%[[VAL_698]] : East, %[[VAL_705:.*]] : West)
// CHECK:           aie.wire(%[[VAL_197]] : Core, %[[VAL_705]] : Core)
// CHECK:           aie.wire(%[[VAL_197]] : DMA, %[[VAL_705]] : DMA)
// CHECK:           aie.wire(%[[VAL_703]] : North, %[[VAL_705]] : South)
// CHECK:           aie.wire(%[[VAL_699]] : East, %[[VAL_706:.*]] : West)
// CHECK:           aie.wire(%[[VAL_196]] : Core, %[[VAL_706]] : Core)
// CHECK:           aie.wire(%[[VAL_196]] : DMA, %[[VAL_706]] : DMA)
// CHECK:           aie.wire(%[[VAL_705]] : North, %[[VAL_706]] : South)
// CHECK:           aie.wire(%[[VAL_700]] : East, %[[VAL_707:.*]] : West)
// CHECK:           aie.wire(%[[VAL_372]] : Core, %[[VAL_707]] : Core)
// CHECK:           aie.wire(%[[VAL_372]] : DMA, %[[VAL_707]] : DMA)
// CHECK:           aie.wire(%[[VAL_706]] : North, %[[VAL_707]] : South)
// CHECK:           aie.wire(%[[VAL_701]] : East, %[[VAL_708:.*]] : West)
// CHECK:           aie.wire(%[[VAL_530]] : Core, %[[VAL_708]] : Core)
// CHECK:           aie.wire(%[[VAL_530]] : DMA, %[[VAL_708]] : DMA)
// CHECK:           aie.wire(%[[VAL_707]] : North, %[[VAL_708]] : South)
// CHECK:           aie.wire(%[[VAL_702]] : East, %[[VAL_709:.*]] : West)
// CHECK:           aie.wire(%[[VAL_478]] : Core, %[[VAL_709]] : Core)
// CHECK:           aie.wire(%[[VAL_478]] : DMA, %[[VAL_709]] : DMA)
// CHECK:           aie.wire(%[[VAL_708]] : North, %[[VAL_709]] : South)
// CHECK:           aie.wire(%[[VAL_703]] : East, %[[VAL_710:.*]] : West)
// CHECK:           aie.wire(%[[VAL_705]] : East, %[[VAL_711:.*]] : West)
// CHECK:           aie.wire(%[[VAL_377]] : Core, %[[VAL_711]] : Core)
// CHECK:           aie.wire(%[[VAL_377]] : DMA, %[[VAL_711]] : DMA)
// CHECK:           aie.wire(%[[VAL_710]] : North, %[[VAL_711]] : South)
// CHECK:           aie.wire(%[[VAL_706]] : East, %[[VAL_712:.*]] : West)
// CHECK:           aie.wire(%[[VAL_409]] : Core, %[[VAL_712]] : Core)
// CHECK:           aie.wire(%[[VAL_409]] : DMA, %[[VAL_712]] : DMA)
// CHECK:           aie.wire(%[[VAL_711]] : North, %[[VAL_712]] : South)
// CHECK:           aie.wire(%[[VAL_707]] : East, %[[VAL_713:.*]] : West)
// CHECK:           aie.wire(%[[VAL_453]] : Core, %[[VAL_713]] : Core)
// CHECK:           aie.wire(%[[VAL_453]] : DMA, %[[VAL_713]] : DMA)
// CHECK:           aie.wire(%[[VAL_712]] : North, %[[VAL_713]] : South)
// CHECK:           aie.wire(%[[VAL_708]] : East, %[[VAL_714:.*]] : West)
// CHECK:           aie.wire(%[[VAL_532]] : Core, %[[VAL_714]] : Core)
// CHECK:           aie.wire(%[[VAL_532]] : DMA, %[[VAL_714]] : DMA)
// CHECK:           aie.wire(%[[VAL_713]] : North, %[[VAL_714]] : South)
// CHECK:           aie.wire(%[[VAL_709]] : East, %[[VAL_715:.*]] : West)
// CHECK:           aie.wire(%[[VAL_480]] : Core, %[[VAL_715]] : Core)
// CHECK:           aie.wire(%[[VAL_480]] : DMA, %[[VAL_715]] : DMA)
// CHECK:           aie.wire(%[[VAL_714]] : North, %[[VAL_715]] : South)
// CHECK:           aie.wire(%[[VAL_710]] : East, %[[VAL_716:.*]] : West)
// CHECK:           aie.wire(%[[VAL_711]] : East, %[[VAL_717:.*]] : West)
// CHECK:           aie.wire(%[[VAL_379]] : Core, %[[VAL_717]] : Core)
// CHECK:           aie.wire(%[[VAL_379]] : DMA, %[[VAL_717]] : DMA)
// CHECK:           aie.wire(%[[VAL_716]] : North, %[[VAL_717]] : South)
// CHECK:           aie.wire(%[[VAL_712]] : East, %[[VAL_718:.*]] : West)
// CHECK:           aie.wire(%[[VAL_411]] : Core, %[[VAL_718]] : Core)
// CHECK:           aie.wire(%[[VAL_411]] : DMA, %[[VAL_718]] : DMA)
// CHECK:           aie.wire(%[[VAL_717]] : North, %[[VAL_718]] : South)
// CHECK:           aie.wire(%[[VAL_713]] : East, %[[VAL_719:.*]] : West)
// CHECK:           aie.wire(%[[VAL_455]] : Core, %[[VAL_719]] : Core)
// CHECK:           aie.wire(%[[VAL_455]] : DMA, %[[VAL_719]] : DMA)
// CHECK:           aie.wire(%[[VAL_718]] : North, %[[VAL_719]] : South)
// CHECK:           aie.wire(%[[VAL_714]] : East, %[[VAL_720:.*]] : West)
// CHECK:           aie.wire(%[[VAL_534]] : Core, %[[VAL_720]] : Core)
// CHECK:           aie.wire(%[[VAL_534]] : DMA, %[[VAL_720]] : DMA)
// CHECK:           aie.wire(%[[VAL_719]] : North, %[[VAL_720]] : South)
// CHECK:           aie.wire(%[[VAL_715]] : East, %[[VAL_721:.*]] : West)
// CHECK:           aie.wire(%[[VAL_482]] : Core, %[[VAL_721]] : Core)
// CHECK:           aie.wire(%[[VAL_482]] : DMA, %[[VAL_721]] : DMA)
// CHECK:           aie.wire(%[[VAL_720]] : North, %[[VAL_721]] : South)
// CHECK:           aie.wire(%[[VAL_716]] : East, %[[VAL_722:.*]] : West)
// CHECK:           aie.wire(%[[VAL_717]] : East, %[[VAL_723:.*]] : West)
// CHECK:           aie.wire(%[[VAL_381]] : Core, %[[VAL_723]] : Core)
// CHECK:           aie.wire(%[[VAL_381]] : DMA, %[[VAL_723]] : DMA)
// CHECK:           aie.wire(%[[VAL_722]] : North, %[[VAL_723]] : South)
// CHECK:           aie.wire(%[[VAL_718]] : East, %[[VAL_724:.*]] : West)
// CHECK:           aie.wire(%[[VAL_413]] : Core, %[[VAL_724]] : Core)
// CHECK:           aie.wire(%[[VAL_413]] : DMA, %[[VAL_724]] : DMA)
// CHECK:           aie.wire(%[[VAL_723]] : North, %[[VAL_724]] : South)
// CHECK:           aie.wire(%[[VAL_719]] : East, %[[VAL_725:.*]] : West)
// CHECK:           aie.wire(%[[VAL_457]] : Core, %[[VAL_725]] : Core)
// CHECK:           aie.wire(%[[VAL_457]] : DMA, %[[VAL_725]] : DMA)
// CHECK:           aie.wire(%[[VAL_724]] : North, %[[VAL_725]] : South)
// CHECK:           aie.wire(%[[VAL_720]] : East, %[[VAL_726:.*]] : West)
// CHECK:           aie.wire(%[[VAL_536]] : Core, %[[VAL_726]] : Core)
// CHECK:           aie.wire(%[[VAL_536]] : DMA, %[[VAL_726]] : DMA)
// CHECK:           aie.wire(%[[VAL_725]] : North, %[[VAL_726]] : South)
// CHECK:           aie.wire(%[[VAL_721]] : East, %[[VAL_727:.*]] : West)
// CHECK:           aie.wire(%[[VAL_484]] : Core, %[[VAL_727]] : Core)
// CHECK:           aie.wire(%[[VAL_484]] : DMA, %[[VAL_727]] : DMA)
// CHECK:           aie.wire(%[[VAL_726]] : North, %[[VAL_727]] : South)
// CHECK:           aie.wire(%[[VAL_722]] : East, %[[VAL_728:.*]] : West)
// CHECK:           aie.wire(%[[VAL_723]] : East, %[[VAL_729:.*]] : West)
// CHECK:           aie.wire(%[[VAL_383]] : Core, %[[VAL_729]] : Core)
// CHECK:           aie.wire(%[[VAL_383]] : DMA, %[[VAL_729]] : DMA)
// CHECK:           aie.wire(%[[VAL_728]] : North, %[[VAL_729]] : South)
// CHECK:           aie.wire(%[[VAL_724]] : East, %[[VAL_730:.*]] : West)
// CHECK:           aie.wire(%[[VAL_415]] : Core, %[[VAL_730]] : Core)
// CHECK:           aie.wire(%[[VAL_415]] : DMA, %[[VAL_730]] : DMA)
// CHECK:           aie.wire(%[[VAL_729]] : North, %[[VAL_730]] : South)
// CHECK:           aie.wire(%[[VAL_725]] : East, %[[VAL_731:.*]] : West)
// CHECK:           aie.wire(%[[VAL_459]] : Core, %[[VAL_731]] : Core)
// CHECK:           aie.wire(%[[VAL_459]] : DMA, %[[VAL_731]] : DMA)
// CHECK:           aie.wire(%[[VAL_730]] : North, %[[VAL_731]] : South)
// CHECK:           aie.wire(%[[VAL_726]] : East, %[[VAL_732:.*]] : West)
// CHECK:           aie.wire(%[[VAL_486]] : Core, %[[VAL_732]] : Core)
// CHECK:           aie.wire(%[[VAL_486]] : DMA, %[[VAL_732]] : DMA)
// CHECK:           aie.wire(%[[VAL_731]] : North, %[[VAL_732]] : South)
// CHECK:           aie.wire(%[[VAL_727]] : East, %[[VAL_733:.*]] : West)
// CHECK:           aie.wire(%[[VAL_488]] : Core, %[[VAL_733]] : Core)
// CHECK:           aie.wire(%[[VAL_488]] : DMA, %[[VAL_733]] : DMA)
// CHECK:           aie.wire(%[[VAL_732]] : North, %[[VAL_733]] : South)
// CHECK:           aie.wire(%[[VAL_728]] : East, %[[VAL_734:.*]] : West)
// CHECK:           aie.wire(%[[VAL_729]] : East, %[[VAL_735:.*]] : West)
// CHECK:           aie.wire(%[[VAL_385]] : Core, %[[VAL_735]] : Core)
// CHECK:           aie.wire(%[[VAL_385]] : DMA, %[[VAL_735]] : DMA)
// CHECK:           aie.wire(%[[VAL_734]] : North, %[[VAL_735]] : South)
// CHECK:           aie.wire(%[[VAL_730]] : East, %[[VAL_736:.*]] : West)
// CHECK:           aie.wire(%[[VAL_417]] : Core, %[[VAL_736]] : Core)
// CHECK:           aie.wire(%[[VAL_417]] : DMA, %[[VAL_736]] : DMA)
// CHECK:           aie.wire(%[[VAL_735]] : North, %[[VAL_736]] : South)
// CHECK:           aie.wire(%[[VAL_731]] : East, %[[VAL_737:.*]] : West)
// CHECK:           aie.wire(%[[VAL_461]] : Core, %[[VAL_737]] : Core)
// CHECK:           aie.wire(%[[VAL_461]] : DMA, %[[VAL_737]] : DMA)
// CHECK:           aie.wire(%[[VAL_736]] : North, %[[VAL_737]] : South)
// CHECK:           aie.wire(%[[VAL_732]] : East, %[[VAL_738:.*]] : West)
// CHECK:           aie.wire(%[[VAL_538]] : Core, %[[VAL_738]] : Core)
// CHECK:           aie.wire(%[[VAL_538]] : DMA, %[[VAL_738]] : DMA)
// CHECK:           aie.wire(%[[VAL_737]] : North, %[[VAL_738]] : South)
// CHECK:           aie.wire(%[[VAL_733]] : East, %[[VAL_739:.*]] : West)
// CHECK:           aie.wire(%[[VAL_597]] : Core, %[[VAL_739]] : Core)
// CHECK:           aie.wire(%[[VAL_597]] : DMA, %[[VAL_739]] : DMA)
// CHECK:           aie.wire(%[[VAL_738]] : North, %[[VAL_739]] : South)
// CHECK:           aie.wire(%[[VAL_734]] : East, %[[VAL_740:.*]] : West)
// CHECK:           aie.wire(%[[VAL_735]] : East, %[[VAL_741:.*]] : West)
// CHECK:           aie.wire(%[[VAL_387]] : Core, %[[VAL_741]] : Core)
// CHECK:           aie.wire(%[[VAL_387]] : DMA, %[[VAL_741]] : DMA)
// CHECK:           aie.wire(%[[VAL_740]] : North, %[[VAL_741]] : South)
// CHECK:           aie.wire(%[[VAL_736]] : East, %[[VAL_742:.*]] : West)
// CHECK:           aie.wire(%[[VAL_419]] : Core, %[[VAL_742]] : Core)
// CHECK:           aie.wire(%[[VAL_419]] : DMA, %[[VAL_742]] : DMA)
// CHECK:           aie.wire(%[[VAL_741]] : North, %[[VAL_742]] : South)
// CHECK:           aie.wire(%[[VAL_737]] : East, %[[VAL_743:.*]] : West)
// CHECK:           aie.wire(%[[VAL_463]] : Core, %[[VAL_743]] : Core)
// CHECK:           aie.wire(%[[VAL_463]] : DMA, %[[VAL_743]] : DMA)
// CHECK:           aie.wire(%[[VAL_742]] : North, %[[VAL_743]] : South)
// CHECK:           aie.wire(%[[VAL_738]] : East, %[[VAL_744:.*]] : West)
// CHECK:           aie.wire(%[[VAL_540]] : Core, %[[VAL_744]] : Core)
// CHECK:           aie.wire(%[[VAL_540]] : DMA, %[[VAL_744]] : DMA)
// CHECK:           aie.wire(%[[VAL_743]] : North, %[[VAL_744]] : South)
// CHECK:           aie.wire(%[[VAL_739]] : East, %[[VAL_745:.*]] : West)
// CHECK:           aie.wire(%[[VAL_599]] : Core, %[[VAL_745]] : Core)
// CHECK:           aie.wire(%[[VAL_599]] : DMA, %[[VAL_745]] : DMA)
// CHECK:           aie.wire(%[[VAL_744]] : North, %[[VAL_745]] : South)
// CHECK:           aie.wire(%[[VAL_740]] : East, %[[VAL_746:.*]] : West)
// CHECK:           aie.wire(%[[VAL_747:.*]] : North, %[[VAL_746]] : South)
// CHECK:           aie.wire(%[[VAL_179]] : DMA, %[[VAL_747]] : DMA)
// CHECK:           aie.wire(%[[VAL_741]] : East, %[[VAL_748:.*]] : West)
// CHECK:           aie.wire(%[[VAL_178]] : Core, %[[VAL_748]] : Core)
// CHECK:           aie.wire(%[[VAL_178]] : DMA, %[[VAL_748]] : DMA)
// CHECK:           aie.wire(%[[VAL_746]] : North, %[[VAL_748]] : South)
// CHECK:           aie.wire(%[[VAL_742]] : East, %[[VAL_749:.*]] : West)
// CHECK:           aie.wire(%[[VAL_177]] : Core, %[[VAL_749]] : Core)
// CHECK:           aie.wire(%[[VAL_177]] : DMA, %[[VAL_749]] : DMA)
// CHECK:           aie.wire(%[[VAL_748]] : North, %[[VAL_749]] : South)
// CHECK:           aie.wire(%[[VAL_743]] : East, %[[VAL_750:.*]] : West)
// CHECK:           aie.wire(%[[VAL_465]] : Core, %[[VAL_750]] : Core)
// CHECK:           aie.wire(%[[VAL_465]] : DMA, %[[VAL_750]] : DMA)
// CHECK:           aie.wire(%[[VAL_749]] : North, %[[VAL_750]] : South)
// CHECK:           aie.wire(%[[VAL_744]] : East, %[[VAL_751:.*]] : West)
// CHECK:           aie.wire(%[[VAL_552]] : Core, %[[VAL_751]] : Core)
// CHECK:           aie.wire(%[[VAL_552]] : DMA, %[[VAL_751]] : DMA)
// CHECK:           aie.wire(%[[VAL_750]] : North, %[[VAL_751]] : South)
// CHECK:           aie.wire(%[[VAL_745]] : East, %[[VAL_752:.*]] : West)
// CHECK:           aie.wire(%[[VAL_601]] : Core, %[[VAL_752]] : Core)
// CHECK:           aie.wire(%[[VAL_601]] : DMA, %[[VAL_752]] : DMA)
// CHECK:           aie.wire(%[[VAL_751]] : North, %[[VAL_752]] : South)
// CHECK:           aie.wire(%[[VAL_746]] : East, %[[VAL_753:.*]] : West)
// CHECK:           aie.wire(%[[VAL_754:.*]] : North, %[[VAL_753]] : South)
// CHECK:           aie.wire(%[[VAL_160]] : DMA, %[[VAL_754]] : DMA)
// CHECK:           aie.wire(%[[VAL_748]] : East, %[[VAL_755:.*]] : West)
// CHECK:           aie.wire(%[[VAL_159]] : Core, %[[VAL_755]] : Core)
// CHECK:           aie.wire(%[[VAL_159]] : DMA, %[[VAL_755]] : DMA)
// CHECK:           aie.wire(%[[VAL_753]] : North, %[[VAL_755]] : South)
// CHECK:           aie.wire(%[[VAL_749]] : East, %[[VAL_756:.*]] : West)
// CHECK:           aie.wire(%[[VAL_158]] : Core, %[[VAL_756]] : Core)
// CHECK:           aie.wire(%[[VAL_158]] : DMA, %[[VAL_756]] : DMA)
// CHECK:           aie.wire(%[[VAL_755]] : North, %[[VAL_756]] : South)
// CHECK:           aie.wire(%[[VAL_750]] : East, %[[VAL_757:.*]] : West)
// CHECK:           aie.wire(%[[VAL_467]] : Core, %[[VAL_757]] : Core)
// CHECK:           aie.wire(%[[VAL_467]] : DMA, %[[VAL_757]] : DMA)
// CHECK:           aie.wire(%[[VAL_756]] : North, %[[VAL_757]] : South)
// CHECK:           aie.wire(%[[VAL_751]] : East, %[[VAL_758:.*]] : West)
// CHECK:           aie.wire(%[[VAL_554]] : Core, %[[VAL_758]] : Core)
// CHECK:           aie.wire(%[[VAL_554]] : DMA, %[[VAL_758]] : DMA)
// CHECK:           aie.wire(%[[VAL_757]] : North, %[[VAL_758]] : South)
// CHECK:           aie.wire(%[[VAL_753]] : East, %[[VAL_759:.*]] : West)
// CHECK:           aie.wire(%[[VAL_755]] : East, %[[VAL_760:.*]] : West)
// CHECK:           aie.wire(%[[VAL_423]] : Core, %[[VAL_760]] : Core)
// CHECK:           aie.wire(%[[VAL_423]] : DMA, %[[VAL_760]] : DMA)
// CHECK:           aie.wire(%[[VAL_759]] : North, %[[VAL_760]] : South)
// CHECK:           aie.wire(%[[VAL_756]] : East, %[[VAL_761:.*]] : West)
// CHECK:           aie.wire(%[[VAL_425]] : Core, %[[VAL_761]] : Core)
// CHECK:           aie.wire(%[[VAL_425]] : DMA, %[[VAL_761]] : DMA)
// CHECK:           aie.wire(%[[VAL_760]] : North, %[[VAL_761]] : South)
// CHECK:           aie.wire(%[[VAL_757]] : East, %[[VAL_762:.*]] : West)
// CHECK:           aie.wire(%[[VAL_469]] : Core, %[[VAL_762]] : Core)
// CHECK:           aie.wire(%[[VAL_469]] : DMA, %[[VAL_762]] : DMA)
// CHECK:           aie.wire(%[[VAL_761]] : North, %[[VAL_762]] : South)
// CHECK:           aie.wire(%[[VAL_758]] : East, %[[VAL_763:.*]] : West)
// CHECK:           aie.wire(%[[VAL_556]] : Core, %[[VAL_763]] : Core)
// CHECK:           aie.wire(%[[VAL_556]] : DMA, %[[VAL_763]] : DMA)
// CHECK:           aie.wire(%[[VAL_762]] : North, %[[VAL_763]] : South)
// CHECK:           aie.wire(%[[VAL_759]] : East, %[[VAL_764:.*]] : West)
// CHECK:           aie.wire(%[[VAL_760]] : East, %[[VAL_765:.*]] : West)
// CHECK:           aie.wire(%[[VAL_427]] : Core, %[[VAL_765]] : Core)
// CHECK:           aie.wire(%[[VAL_427]] : DMA, %[[VAL_765]] : DMA)
// CHECK:           aie.wire(%[[VAL_764]] : North, %[[VAL_765]] : South)
// CHECK:           aie.wire(%[[VAL_761]] : East, %[[VAL_766:.*]] : West)
// CHECK:           aie.wire(%[[VAL_542]] : Core, %[[VAL_766]] : Core)
// CHECK:           aie.wire(%[[VAL_542]] : DMA, %[[VAL_766]] : DMA)
// CHECK:           aie.wire(%[[VAL_765]] : North, %[[VAL_766]] : South)
// CHECK:           aie.wire(%[[VAL_762]] : East, %[[VAL_767:.*]] : West)
// CHECK:           aie.wire(%[[VAL_490]] : Core, %[[VAL_767]] : Core)
// CHECK:           aie.wire(%[[VAL_490]] : DMA, %[[VAL_767]] : DMA)
// CHECK:           aie.wire(%[[VAL_766]] : North, %[[VAL_767]] : South)
// CHECK:           aie.wire(%[[VAL_763]] : East, %[[VAL_768:.*]] : West)
// CHECK:           aie.wire(%[[VAL_603]] : Core, %[[VAL_768]] : Core)
// CHECK:           aie.wire(%[[VAL_603]] : DMA, %[[VAL_768]] : DMA)
// CHECK:           aie.wire(%[[VAL_767]] : North, %[[VAL_768]] : South)
// CHECK:           aie.wire(%[[VAL_764]] : East, %[[VAL_769:.*]] : West)
// CHECK:           aie.wire(%[[VAL_765]] : East, %[[VAL_770:.*]] : West)
// CHECK:           aie.wire(%[[VAL_429]] : Core, %[[VAL_770]] : Core)
// CHECK:           aie.wire(%[[VAL_429]] : DMA, %[[VAL_770]] : DMA)
// CHECK:           aie.wire(%[[VAL_769]] : North, %[[VAL_770]] : South)
// CHECK:           aie.wire(%[[VAL_766]] : East, %[[VAL_771:.*]] : West)
// CHECK:           aie.wire(%[[VAL_544]] : Core, %[[VAL_771]] : Core)
// CHECK:           aie.wire(%[[VAL_544]] : DMA, %[[VAL_771]] : DMA)
// CHECK:           aie.wire(%[[VAL_770]] : North, %[[VAL_771]] : South)
// CHECK:           aie.wire(%[[VAL_767]] : East, %[[VAL_772:.*]] : West)
// CHECK:           aie.wire(%[[VAL_492]] : Core, %[[VAL_772]] : Core)
// CHECK:           aie.wire(%[[VAL_492]] : DMA, %[[VAL_772]] : DMA)
// CHECK:           aie.wire(%[[VAL_771]] : North, %[[VAL_772]] : South)
// CHECK:           aie.wire(%[[VAL_768]] : East, %[[VAL_773:.*]] : West)
// CHECK:           aie.wire(%[[VAL_605]] : Core, %[[VAL_773]] : Core)
// CHECK:           aie.wire(%[[VAL_605]] : DMA, %[[VAL_773]] : DMA)
// CHECK:           aie.wire(%[[VAL_772]] : North, %[[VAL_773]] : South)
// CHECK:           aie.wire(%[[VAL_769]] : East, %[[VAL_774:.*]] : West)
// CHECK:           aie.wire(%[[VAL_770]] : East, %[[VAL_775:.*]] : West)
// CHECK:           aie.wire(%[[VAL_431]] : Core, %[[VAL_775]] : Core)
// CHECK:           aie.wire(%[[VAL_431]] : DMA, %[[VAL_775]] : DMA)
// CHECK:           aie.wire(%[[VAL_774]] : North, %[[VAL_775]] : South)
// CHECK:           aie.wire(%[[VAL_771]] : East, %[[VAL_776:.*]] : West)
// CHECK:           aie.wire(%[[VAL_546]] : Core, %[[VAL_776]] : Core)
// CHECK:           aie.wire(%[[VAL_546]] : DMA, %[[VAL_776]] : DMA)
// CHECK:           aie.wire(%[[VAL_775]] : North, %[[VAL_776]] : South)
// CHECK:           aie.wire(%[[VAL_772]] : East, %[[VAL_777:.*]] : West)
// CHECK:           aie.wire(%[[VAL_494]] : Core, %[[VAL_777]] : Core)
// CHECK:           aie.wire(%[[VAL_494]] : DMA, %[[VAL_777]] : DMA)
// CHECK:           aie.wire(%[[VAL_776]] : North, %[[VAL_777]] : South)
// CHECK:           aie.wire(%[[VAL_773]] : East, %[[VAL_778:.*]] : West)
// CHECK:           aie.wire(%[[VAL_607]] : Core, %[[VAL_778]] : Core)
// CHECK:           aie.wire(%[[VAL_607]] : DMA, %[[VAL_778]] : DMA)
// CHECK:           aie.wire(%[[VAL_777]] : North, %[[VAL_778]] : South)
// CHECK:           aie.wire(%[[VAL_774]] : East, %[[VAL_779:.*]] : West)
// CHECK:           aie.wire(%[[VAL_775]] : East, %[[VAL_780:.*]] : West)
// CHECK:           aie.wire(%[[VAL_433]] : Core, %[[VAL_780]] : Core)
// CHECK:           aie.wire(%[[VAL_433]] : DMA, %[[VAL_780]] : DMA)
// CHECK:           aie.wire(%[[VAL_779]] : North, %[[VAL_780]] : South)
// CHECK:           aie.wire(%[[VAL_776]] : East, %[[VAL_781:.*]] : West)
// CHECK:           aie.wire(%[[VAL_496]] : Core, %[[VAL_781]] : Core)
// CHECK:           aie.wire(%[[VAL_496]] : DMA, %[[VAL_781]] : DMA)
// CHECK:           aie.wire(%[[VAL_780]] : North, %[[VAL_781]] : South)
// CHECK:           aie.wire(%[[VAL_777]] : East, %[[VAL_782:.*]] : West)
// CHECK:           aie.wire(%[[VAL_498]] : Core, %[[VAL_782]] : Core)
// CHECK:           aie.wire(%[[VAL_498]] : DMA, %[[VAL_782]] : DMA)
// CHECK:           aie.wire(%[[VAL_781]] : North, %[[VAL_782]] : South)
// CHECK:           aie.wire(%[[VAL_778]] : East, %[[VAL_783:.*]] : West)
// CHECK:           aie.wire(%[[VAL_609]] : Core, %[[VAL_783]] : Core)
// CHECK:           aie.wire(%[[VAL_609]] : DMA, %[[VAL_783]] : DMA)
// CHECK:           aie.wire(%[[VAL_782]] : North, %[[VAL_783]] : South)
// CHECK:           aie.wire(%[[VAL_779]] : East, %[[VAL_784:.*]] : West)
// CHECK:           aie.wire(%[[VAL_780]] : East, %[[VAL_785:.*]] : West)
// CHECK:           aie.wire(%[[VAL_435]] : Core, %[[VAL_785]] : Core)
// CHECK:           aie.wire(%[[VAL_435]] : DMA, %[[VAL_785]] : DMA)
// CHECK:           aie.wire(%[[VAL_784]] : North, %[[VAL_785]] : South)
// CHECK:           aie.wire(%[[VAL_781]] : East, %[[VAL_786:.*]] : West)
// CHECK:           aie.wire(%[[VAL_500]] : Core, %[[VAL_786]] : Core)
// CHECK:           aie.wire(%[[VAL_500]] : DMA, %[[VAL_786]] : DMA)
// CHECK:           aie.wire(%[[VAL_785]] : North, %[[VAL_786]] : South)
// CHECK:           aie.wire(%[[VAL_782]] : East, %[[VAL_787:.*]] : West)
// CHECK:           aie.wire(%[[VAL_558]] : Core, %[[VAL_787]] : Core)
// CHECK:           aie.wire(%[[VAL_558]] : DMA, %[[VAL_787]] : DMA)
// CHECK:           aie.wire(%[[VAL_786]] : North, %[[VAL_787]] : South)
// CHECK:           aie.wire(%[[VAL_783]] : East, %[[VAL_788:.*]] : West)
// CHECK:           aie.wire(%[[VAL_611]] : Core, %[[VAL_788]] : Core)
// CHECK:           aie.wire(%[[VAL_611]] : DMA, %[[VAL_788]] : DMA)
// CHECK:           aie.wire(%[[VAL_787]] : North, %[[VAL_788]] : South)
// CHECK:           aie.wire(%[[VAL_784]] : East, %[[VAL_789:.*]] : West)
// CHECK:           aie.wire(%[[VAL_790:.*]] : North, %[[VAL_789]] : South)
// CHECK:           aie.wire(%[[VAL_140]] : DMA, %[[VAL_790]] : DMA)
// CHECK:           aie.wire(%[[VAL_785]] : East, %[[VAL_791:.*]] : West)
// CHECK:           aie.wire(%[[VAL_139]] : Core, %[[VAL_791]] : Core)
// CHECK:           aie.wire(%[[VAL_139]] : DMA, %[[VAL_791]] : DMA)
// CHECK:           aie.wire(%[[VAL_789]] : North, %[[VAL_791]] : South)
// CHECK:           aie.wire(%[[VAL_786]] : East, %[[VAL_792:.*]] : West)
// CHECK:           aie.wire(%[[VAL_138]] : Core, %[[VAL_792]] : Core)
// CHECK:           aie.wire(%[[VAL_138]] : DMA, %[[VAL_792]] : DMA)
// CHECK:           aie.wire(%[[VAL_791]] : North, %[[VAL_792]] : South)
// CHECK:           aie.wire(%[[VAL_787]] : East, %[[VAL_793:.*]] : West)
// CHECK:           aie.wire(%[[VAL_560]] : Core, %[[VAL_793]] : Core)
// CHECK:           aie.wire(%[[VAL_560]] : DMA, %[[VAL_793]] : DMA)
// CHECK:           aie.wire(%[[VAL_792]] : North, %[[VAL_793]] : South)
// CHECK:           aie.wire(%[[VAL_788]] : East, %[[VAL_794:.*]] : West)
// CHECK:           aie.wire(%[[VAL_613]] : Core, %[[VAL_794]] : Core)
// CHECK:           aie.wire(%[[VAL_613]] : DMA, %[[VAL_794]] : DMA)
// CHECK:           aie.wire(%[[VAL_793]] : North, %[[VAL_794]] : South)
// CHECK:           aie.wire(%[[VAL_789]] : East, %[[VAL_795:.*]] : West)
// CHECK:           aie.wire(%[[VAL_796:.*]] : North, %[[VAL_795]] : South)
// CHECK:           aie.wire(%[[VAL_120]] : DMA, %[[VAL_796]] : DMA)
// CHECK:           aie.wire(%[[VAL_791]] : East, %[[VAL_797:.*]] : West)
// CHECK:           aie.wire(%[[VAL_119]] : Core, %[[VAL_797]] : Core)
// CHECK:           aie.wire(%[[VAL_119]] : DMA, %[[VAL_797]] : DMA)
// CHECK:           aie.wire(%[[VAL_795]] : North, %[[VAL_797]] : South)
// CHECK:           aie.wire(%[[VAL_792]] : East, %[[VAL_798:.*]] : West)
// CHECK:           aie.wire(%[[VAL_118]] : Core, %[[VAL_798]] : Core)
// CHECK:           aie.wire(%[[VAL_118]] : DMA, %[[VAL_798]] : DMA)
// CHECK:           aie.wire(%[[VAL_797]] : North, %[[VAL_798]] : South)
// CHECK:           aie.wire(%[[VAL_794]] : East, %[[VAL_799:.*]] : West)
// CHECK:           aie.wire(%[[VAL_615]] : Core, %[[VAL_799]] : Core)
// CHECK:           aie.wire(%[[VAL_615]] : DMA, %[[VAL_799]] : DMA)
// CHECK:           aie.wire(%[[VAL_795]] : East, %[[VAL_800:.*]] : West)
// CHECK:           aie.wire(%[[VAL_797]] : East, %[[VAL_801:.*]] : West)
// CHECK:           aie.wire(%[[VAL_503]] : Core, %[[VAL_801]] : Core)
// CHECK:           aie.wire(%[[VAL_503]] : DMA, %[[VAL_801]] : DMA)
// CHECK:           aie.wire(%[[VAL_800]] : North, %[[VAL_801]] : South)
// CHECK:           aie.wire(%[[VAL_798]] : East, %[[VAL_802:.*]] : West)
// CHECK:           aie.wire(%[[VAL_562]] : Core, %[[VAL_802]] : Core)
// CHECK:           aie.wire(%[[VAL_562]] : DMA, %[[VAL_802]] : DMA)
// CHECK:           aie.wire(%[[VAL_801]] : North, %[[VAL_802]] : South)
// CHECK:           aie.wire(%[[VAL_799]] : East, %[[VAL_803:.*]] : West)
// CHECK:           aie.wire(%[[VAL_617]] : Core, %[[VAL_803]] : Core)
// CHECK:           aie.wire(%[[VAL_617]] : DMA, %[[VAL_803]] : DMA)
// CHECK:           aie.wire(%[[VAL_800]] : East, %[[VAL_804:.*]] : West)
// CHECK:           aie.wire(%[[VAL_801]] : East, %[[VAL_805:.*]] : West)
// CHECK:           aie.wire(%[[VAL_505]] : Core, %[[VAL_805]] : Core)
// CHECK:           aie.wire(%[[VAL_505]] : DMA, %[[VAL_805]] : DMA)
// CHECK:           aie.wire(%[[VAL_804]] : North, %[[VAL_805]] : South)
// CHECK:           aie.wire(%[[VAL_802]] : East, %[[VAL_806:.*]] : West)
// CHECK:           aie.wire(%[[VAL_564]] : Core, %[[VAL_806]] : Core)
// CHECK:           aie.wire(%[[VAL_564]] : DMA, %[[VAL_806]] : DMA)
// CHECK:           aie.wire(%[[VAL_805]] : North, %[[VAL_806]] : South)
// CHECK:           aie.wire(%[[VAL_803]] : East, %[[VAL_807:.*]] : West)
// CHECK:           aie.wire(%[[VAL_619]] : Core, %[[VAL_807]] : Core)
// CHECK:           aie.wire(%[[VAL_619]] : DMA, %[[VAL_807]] : DMA)
// CHECK:           aie.wire(%[[VAL_804]] : East, %[[VAL_808:.*]] : West)
// CHECK:           aie.wire(%[[VAL_805]] : East, %[[VAL_809:.*]] : West)
// CHECK:           aie.wire(%[[VAL_507]] : Core, %[[VAL_809]] : Core)
// CHECK:           aie.wire(%[[VAL_507]] : DMA, %[[VAL_809]] : DMA)
// CHECK:           aie.wire(%[[VAL_808]] : North, %[[VAL_809]] : South)
// CHECK:           aie.wire(%[[VAL_806]] : East, %[[VAL_810:.*]] : West)
// CHECK:           aie.wire(%[[VAL_566]] : Core, %[[VAL_810]] : Core)
// CHECK:           aie.wire(%[[VAL_566]] : DMA, %[[VAL_810]] : DMA)
// CHECK:           aie.wire(%[[VAL_809]] : North, %[[VAL_810]] : South)
// CHECK:           aie.wire(%[[VAL_807]] : East, %[[VAL_811:.*]] : West)
// CHECK:           aie.wire(%[[VAL_621]] : Core, %[[VAL_811]] : Core)
// CHECK:           aie.wire(%[[VAL_621]] : DMA, %[[VAL_811]] : DMA)
// CHECK:           aie.wire(%[[VAL_808]] : East, %[[VAL_812:.*]] : West)
// CHECK:           aie.wire(%[[VAL_809]] : East, %[[VAL_813:.*]] : West)
// CHECK:           aie.wire(%[[VAL_509]] : Core, %[[VAL_813]] : Core)
// CHECK:           aie.wire(%[[VAL_509]] : DMA, %[[VAL_813]] : DMA)
// CHECK:           aie.wire(%[[VAL_812]] : North, %[[VAL_813]] : South)
// CHECK:           aie.wire(%[[VAL_810]] : East, %[[VAL_814:.*]] : West)
// CHECK:           aie.wire(%[[VAL_568]] : Core, %[[VAL_814]] : Core)
// CHECK:           aie.wire(%[[VAL_568]] : DMA, %[[VAL_814]] : DMA)
// CHECK:           aie.wire(%[[VAL_813]] : North, %[[VAL_814]] : South)
// CHECK:           aie.wire(%[[VAL_811]] : East, %[[VAL_815:.*]] : West)
// CHECK:           aie.wire(%[[VAL_623]] : Core, %[[VAL_815]] : Core)
// CHECK:           aie.wire(%[[VAL_623]] : DMA, %[[VAL_815]] : DMA)
// CHECK:           aie.wire(%[[VAL_812]] : East, %[[VAL_816:.*]] : West)
// CHECK:           aie.wire(%[[VAL_813]] : East, %[[VAL_817:.*]] : West)
// CHECK:           aie.wire(%[[VAL_511]] : Core, %[[VAL_817]] : Core)
// CHECK:           aie.wire(%[[VAL_511]] : DMA, %[[VAL_817]] : DMA)
// CHECK:           aie.wire(%[[VAL_816]] : North, %[[VAL_817]] : South)
// CHECK:           aie.wire(%[[VAL_814]] : East, %[[VAL_818:.*]] : West)
// CHECK:           aie.wire(%[[VAL_570]] : Core, %[[VAL_818]] : Core)
// CHECK:           aie.wire(%[[VAL_570]] : DMA, %[[VAL_818]] : DMA)
// CHECK:           aie.wire(%[[VAL_817]] : North, %[[VAL_818]] : South)
// CHECK:           aie.wire(%[[VAL_815]] : East, %[[VAL_819:.*]] : West)
// CHECK:           aie.wire(%[[VAL_625]] : Core, %[[VAL_819]] : Core)
// CHECK:           aie.wire(%[[VAL_625]] : DMA, %[[VAL_819]] : DMA)
// CHECK:           aie.wire(%[[VAL_816]] : East, %[[VAL_820:.*]] : West)
// CHECK:           aie.wire(%[[VAL_817]] : East, %[[VAL_821:.*]] : West)
// CHECK:           aie.wire(%[[VAL_513]] : Core, %[[VAL_821]] : Core)
// CHECK:           aie.wire(%[[VAL_513]] : DMA, %[[VAL_821]] : DMA)
// CHECK:           aie.wire(%[[VAL_820]] : North, %[[VAL_821]] : South)
// CHECK:           aie.wire(%[[VAL_818]] : East, %[[VAL_822:.*]] : West)
// CHECK:           aie.wire(%[[VAL_572]] : Core, %[[VAL_822]] : Core)
// CHECK:           aie.wire(%[[VAL_572]] : DMA, %[[VAL_822]] : DMA)
// CHECK:           aie.wire(%[[VAL_821]] : North, %[[VAL_822]] : South)
// CHECK:           aie.wire(%[[VAL_819]] : East, %[[VAL_823:.*]] : West)
// CHECK:           aie.wire(%[[VAL_627]] : Core, %[[VAL_823]] : Core)
// CHECK:           aie.wire(%[[VAL_627]] : DMA, %[[VAL_823]] : DMA)
// CHECK:           aie.wire(%[[VAL_820]] : East, %[[VAL_824:.*]] : West)
// CHECK:           aie.wire(%[[VAL_825:.*]] : North, %[[VAL_824]] : South)
// CHECK:           aie.wire(%[[VAL_101]] : DMA, %[[VAL_825]] : DMA)
// CHECK:           aie.wire(%[[VAL_821]] : East, %[[VAL_826:.*]] : West)
// CHECK:           aie.wire(%[[VAL_100]] : Core, %[[VAL_826]] : Core)
// CHECK:           aie.wire(%[[VAL_100]] : DMA, %[[VAL_826]] : DMA)
// CHECK:           aie.wire(%[[VAL_824]] : North, %[[VAL_826]] : South)
// CHECK:           aie.wire(%[[VAL_822]] : East, %[[VAL_827:.*]] : West)
// CHECK:           aie.wire(%[[VAL_99]] : Core, %[[VAL_827]] : Core)
// CHECK:           aie.wire(%[[VAL_99]] : DMA, %[[VAL_827]] : DMA)
// CHECK:           aie.wire(%[[VAL_826]] : North, %[[VAL_827]] : South)
// CHECK:           aie.wire(%[[VAL_823]] : East, %[[VAL_828:.*]] : West)
// CHECK:           aie.wire(%[[VAL_629]] : Core, %[[VAL_828]] : Core)
// CHECK:           aie.wire(%[[VAL_629]] : DMA, %[[VAL_828]] : DMA)
// CHECK:           aie.wire(%[[VAL_824]] : East, %[[VAL_829:.*]] : West)
// CHECK:           aie.wire(%[[VAL_830:.*]] : North, %[[VAL_829]] : South)
// CHECK:           aie.wire(%[[VAL_82]] : DMA, %[[VAL_830]] : DMA)
// CHECK:           aie.wire(%[[VAL_826]] : East, %[[VAL_831:.*]] : West)
// CHECK:           aie.wire(%[[VAL_81]] : Core, %[[VAL_831]] : Core)
// CHECK:           aie.wire(%[[VAL_81]] : DMA, %[[VAL_831]] : DMA)
// CHECK:           aie.wire(%[[VAL_829]] : North, %[[VAL_831]] : South)
// CHECK:           aie.wire(%[[VAL_827]] : East, %[[VAL_832:.*]] : West)
// CHECK:           aie.wire(%[[VAL_80]] : Core, %[[VAL_832]] : Core)
// CHECK:           aie.wire(%[[VAL_80]] : DMA, %[[VAL_832]] : DMA)
// CHECK:           aie.wire(%[[VAL_831]] : North, %[[VAL_832]] : South)
// CHECK:           aie.wire(%[[VAL_828]] : East, %[[VAL_833:.*]] : West)
// CHECK:           aie.wire(%[[VAL_631]] : Core, %[[VAL_833]] : Core)
// CHECK:           aie.wire(%[[VAL_631]] : DMA, %[[VAL_833]] : DMA)
// CHECK:           aie.wire(%[[VAL_829]] : East, %[[VAL_834:.*]] : West)
// CHECK:           aie.wire(%[[VAL_832]] : East, %[[VAL_835:.*]] : West)
// CHECK:           aie.wire(%[[VAL_576]] : Core, %[[VAL_835]] : Core)
// CHECK:           aie.wire(%[[VAL_576]] : DMA, %[[VAL_835]] : DMA)
// CHECK:           aie.wire(%[[VAL_833]] : East, %[[VAL_836:.*]] : West)
// CHECK:           aie.wire(%[[VAL_633]] : Core, %[[VAL_836]] : Core)
// CHECK:           aie.wire(%[[VAL_633]] : DMA, %[[VAL_836]] : DMA)
// CHECK:           aie.wire(%[[VAL_834]] : East, %[[VAL_837:.*]] : West)
// CHECK:           aie.wire(%[[VAL_835]] : East, %[[VAL_838:.*]] : West)
// CHECK:           aie.wire(%[[VAL_578]] : Core, %[[VAL_838]] : Core)
// CHECK:           aie.wire(%[[VAL_578]] : DMA, %[[VAL_838]] : DMA)
// CHECK:           aie.wire(%[[VAL_836]] : East, %[[VAL_839:.*]] : West)
// CHECK:           aie.wire(%[[VAL_635]] : Core, %[[VAL_839]] : Core)
// CHECK:           aie.wire(%[[VAL_635]] : DMA, %[[VAL_839]] : DMA)
// CHECK:           aie.wire(%[[VAL_837]] : East, %[[VAL_840:.*]] : West)
// CHECK:           aie.wire(%[[VAL_838]] : East, %[[VAL_841:.*]] : West)
// CHECK:           aie.wire(%[[VAL_580]] : Core, %[[VAL_841]] : Core)
// CHECK:           aie.wire(%[[VAL_580]] : DMA, %[[VAL_841]] : DMA)
// CHECK:           aie.wire(%[[VAL_839]] : East, %[[VAL_842:.*]] : West)
// CHECK:           aie.wire(%[[VAL_637]] : Core, %[[VAL_842]] : Core)
// CHECK:           aie.wire(%[[VAL_637]] : DMA, %[[VAL_842]] : DMA)
// CHECK:           aie.wire(%[[VAL_840]] : East, %[[VAL_843:.*]] : West)
// CHECK:           aie.wire(%[[VAL_841]] : East, %[[VAL_844:.*]] : West)
// CHECK:           aie.wire(%[[VAL_582]] : Core, %[[VAL_844]] : Core)
// CHECK:           aie.wire(%[[VAL_582]] : DMA, %[[VAL_844]] : DMA)
// CHECK:           aie.wire(%[[VAL_842]] : East, %[[VAL_845:.*]] : West)
// CHECK:           aie.wire(%[[VAL_639]] : Core, %[[VAL_845]] : Core)
// CHECK:           aie.wire(%[[VAL_639]] : DMA, %[[VAL_845]] : DMA)
// CHECK:           aie.wire(%[[VAL_843]] : East, %[[VAL_846:.*]] : West)
// CHECK:           aie.wire(%[[VAL_844]] : East, %[[VAL_847:.*]] : West)
// CHECK:           aie.wire(%[[VAL_584]] : Core, %[[VAL_847]] : Core)
// CHECK:           aie.wire(%[[VAL_584]] : DMA, %[[VAL_847]] : DMA)
// CHECK:           aie.wire(%[[VAL_845]] : East, %[[VAL_848:.*]] : West)
// CHECK:           aie.wire(%[[VAL_641]] : Core, %[[VAL_848]] : Core)
// CHECK:           aie.wire(%[[VAL_641]] : DMA, %[[VAL_848]] : DMA)
// CHECK:           aie.wire(%[[VAL_846]] : East, %[[VAL_849:.*]] : West)
// CHECK:           aie.wire(%[[VAL_586]] : Core, %[[VAL_850:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_586]] : DMA, %[[VAL_850]] : DMA)
// CHECK:           aie.wire(%[[VAL_849]] : North, %[[VAL_850]] : South)
// CHECK:           aie.wire(%[[VAL_847]] : East, %[[VAL_851:.*]] : West)
// CHECK:           aie.wire(%[[VAL_588]] : Core, %[[VAL_851]] : Core)
// CHECK:           aie.wire(%[[VAL_588]] : DMA, %[[VAL_851]] : DMA)
// CHECK:           aie.wire(%[[VAL_850]] : North, %[[VAL_851]] : South)
// CHECK:           aie.wire(%[[VAL_848]] : East, %[[VAL_852:.*]] : West)
// CHECK:           aie.wire(%[[VAL_643]] : Core, %[[VAL_852]] : Core)
// CHECK:           aie.wire(%[[VAL_643]] : DMA, %[[VAL_852]] : DMA)
// CHECK:           aie.wire(%[[VAL_849]] : East, %[[VAL_853:.*]] : West)
// CHECK:           aie.wire(%[[VAL_854:.*]] : North, %[[VAL_853]] : South)
// CHECK:           aie.wire(%[[VAL_62]] : DMA, %[[VAL_854]] : DMA)
// CHECK:           aie.wire(%[[VAL_850]] : East, %[[VAL_855:.*]] : West)
// CHECK:           aie.wire(%[[VAL_61]] : Core, %[[VAL_855]] : Core)
// CHECK:           aie.wire(%[[VAL_61]] : DMA, %[[VAL_855]] : DMA)
// CHECK:           aie.wire(%[[VAL_853]] : North, %[[VAL_855]] : South)
// CHECK:           aie.wire(%[[VAL_645]] : Core, %[[VAL_856:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_645]] : DMA, %[[VAL_856]] : DMA)
// CHECK:           aie.wire(%[[VAL_852]] : East, %[[VAL_857:.*]] : West)
// CHECK:           aie.wire(%[[VAL_647]] : Core, %[[VAL_857]] : Core)
// CHECK:           aie.wire(%[[VAL_647]] : DMA, %[[VAL_857]] : DMA)
// CHECK:           aie.wire(%[[VAL_856]] : North, %[[VAL_857]] : South)
// CHECK:           aie.wire(%[[VAL_853]] : East, %[[VAL_858:.*]] : West)
// CHECK:           aie.wire(%[[VAL_859:.*]] : North, %[[VAL_858]] : South)
// CHECK:           aie.wire(%[[VAL_42]] : DMA, %[[VAL_859]] : DMA)
// CHECK:           aie.wire(%[[VAL_856]] : East, %[[VAL_860:.*]] : West)
// CHECK:           aie.wire(%[[VAL_649]] : Core, %[[VAL_860]] : Core)
// CHECK:           aie.wire(%[[VAL_649]] : DMA, %[[VAL_860]] : DMA)
// CHECK:           aie.wire(%[[VAL_858]] : East, %[[VAL_861:.*]] : West)
// CHECK:           aie.wire(%[[VAL_860]] : East, %[[VAL_862:.*]] : West)
// CHECK:           aie.wire(%[[VAL_651]] : Core, %[[VAL_862]] : Core)
// CHECK:           aie.wire(%[[VAL_651]] : DMA, %[[VAL_862]] : DMA)
// CHECK:           aie.wire(%[[VAL_861]] : East, %[[VAL_863:.*]] : West)
// CHECK:           aie.wire(%[[VAL_653]] : Core, %[[VAL_864:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_653]] : DMA, %[[VAL_864]] : DMA)
// CHECK:           aie.wire(%[[VAL_862]] : East, %[[VAL_865:.*]] : West)
// CHECK:           aie.wire(%[[VAL_655]] : Core, %[[VAL_865]] : Core)
// CHECK:           aie.wire(%[[VAL_655]] : DMA, %[[VAL_865]] : DMA)
// CHECK:           aie.wire(%[[VAL_864]] : North, %[[VAL_865]] : South)
// CHECK:           aie.wire(%[[VAL_863]] : East, %[[VAL_866:.*]] : West)
// CHECK:           aie.wire(%[[VAL_867:.*]] : North, %[[VAL_866]] : South)
// CHECK:           aie.wire(%[[VAL_22]] : DMA, %[[VAL_867]] : DMA)
// CHECK:           aie.wire(%[[VAL_21]] : Core, %[[VAL_868:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_21]] : DMA, %[[VAL_868]] : DMA)
// CHECK:           aie.wire(%[[VAL_866]] : North, %[[VAL_868]] : South)
// CHECK:           aie.wire(%[[VAL_864]] : East, %[[VAL_869:.*]] : West)
// CHECK:           aie.wire(%[[VAL_20]] : Core, %[[VAL_869]] : Core)
// CHECK:           aie.wire(%[[VAL_20]] : DMA, %[[VAL_869]] : DMA)
// CHECK:           aie.wire(%[[VAL_868]] : North, %[[VAL_869]] : South)
// CHECK:           aie.wire(%[[VAL_866]] : East, %[[VAL_870:.*]] : West)
// CHECK:           aie.wire(%[[VAL_871:.*]] : North, %[[VAL_870]] : South)
// CHECK:           aie.wire(%[[VAL_2]] : DMA, %[[VAL_871]] : DMA)
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
      aie.dma_bd(%10 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%9, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%7, Acquire, 0)
      aie.dma_bd(%8 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%7, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%5, Acquire, 1)
      aie.dma_bd(%6 : memref<64xi32, 2>) { len = 64 : i32 }
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
      aie.dma_bd(%23 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%22, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%20, Acquire, 0)
      aie.dma_bd(%21 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%20, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%18, Acquire, 1)
      aie.dma_bd(%19 : memref<64xi32, 2>) { len = 64 : i32 }
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
      aie.dma_bd(%36 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%35, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%33, Acquire, 0)
      aie.dma_bd(%34 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%33, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%31, Acquire, 1)
      aie.dma_bd(%32 : memref<64xi32, 2>) { len = 64 : i32 }
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
      aie.dma_bd(%49 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%48, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%46, Acquire, 0)
      aie.dma_bd(%47 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%46, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%44, Acquire, 1)
      aie.dma_bd(%45 : memref<64xi32, 2>) { len = 64 : i32 }
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
      aie.dma_bd(%61 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%60, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%58, Acquire, 0)
      aie.dma_bd(%59 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%58, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%56, Acquire, 1)
      aie.dma_bd(%57 : memref<64xi32, 2>) { len = 64 : i32 }
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
      aie.dma_bd(%73 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%72, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%70, Acquire, 0)
      aie.dma_bd(%71 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%70, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%68, Acquire, 1)
      aie.dma_bd(%69 : memref<64xi32, 2>) { len = 64 : i32 }
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
      aie.dma_bd(%86 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%85, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%83, Acquire, 0)
      aie.dma_bd(%84 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%83, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%81, Acquire, 1)
      aie.dma_bd(%82 : memref<64xi32, 2>) { len = 64 : i32 }
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
      aie.dma_bd(%99 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%98, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%96, Acquire, 0)
      aie.dma_bd(%97 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%96, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%94, Acquire, 1)
      aie.dma_bd(%95 : memref<64xi32, 2>) { len = 64 : i32 }
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
      aie.dma_bd(%111 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%110, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%108, Acquire, 0)
      aie.dma_bd(%109 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%108, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%106, Acquire, 1)
      aie.dma_bd(%107 : memref<64xi32, 2>) { len = 64 : i32 }
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
      aie.dma_bd(%123 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%122, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%120, Acquire, 0)
      aie.dma_bd(%121 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%120, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%118, Acquire, 1)
      aie.dma_bd(%119 : memref<64xi32, 2>) { len = 64 : i32 }
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
      aie.dma_bd(%136 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%135, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%133, Acquire, 0)
      aie.dma_bd(%134 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%133, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%131, Acquire, 1)
      aie.dma_bd(%132 : memref<64xi32, 2>) { len = 64 : i32 }
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
      aie.dma_bd(%148 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%147, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%145, Acquire, 0)
      aie.dma_bd(%146 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%145, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%143, Acquire, 1)
      aie.dma_bd(%144 : memref<64xi32, 2>) { len = 64 : i32 }
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
      aie.dma_bd(%159 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%158, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%156, Acquire, 0)
      aie.dma_bd(%157 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%156, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%154, Acquire, 1)
      aie.dma_bd(%155 : memref<64xi32, 2>) { len = 64 : i32 }
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
      aie.dma_bd(%171 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%170, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%168, Acquire, 0)
      aie.dma_bd(%169 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%168, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%166, Acquire, 1)
      aie.dma_bd(%167 : memref<64xi32, 2>) { len = 64 : i32 }
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
      aie.dma_bd(%184 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%183, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%181, Acquire, 0)
      aie.dma_bd(%182 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%181, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%179, Acquire, 1)
      aie.dma_bd(%180 : memref<64xi32, 2>) { len = 64 : i32 }
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
      aie.dma_bd(%197 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%196, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%194, Acquire, 0)
      aie.dma_bd(%195 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%194, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%192, Acquire, 1)
      aie.dma_bd(%193 : memref<64xi32, 2>) { len = 64 : i32 }
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
