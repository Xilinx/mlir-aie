//===- mmult.mlir ----------------------------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_0:.*]] = AIE.tile(7, 1)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(7, 0)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(1, 1)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(8, 3)
// CHECK:           %[[VAL_4:.*]] = AIE.lock(%[[VAL_3]], 1)
// CHECK:           %[[VAL_5:.*]] = AIE.lock(%[[VAL_3]], 3)
// CHECK:           %[[VAL_6:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = "buf11"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_7:.*]] = AIE.lock(%[[VAL_3]], 2)
// CHECK:           %[[VAL_8:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = "buf10"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_9:.*]] = AIE.lock(%[[VAL_3]], 0)
// CHECK:           %[[VAL_10:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = "buf9"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_11:.*]] = AIE.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_12:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_9]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_10]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_9]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_4]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_6]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_4]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_13:.*]] = AIE.dmaStart(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             AIE.useLock(%[[VAL_7]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_8]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:             AIE.nextBd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_14:.*]] = AIE.dmaStart(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             AIE.useLock(%[[VAL_5]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_6]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_5]], Release, 0)
// CHECK:             AIE.nextBd ^bb6
// CHECK:           ^bb7:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = AIE.tile(6, 2)
// CHECK:           %[[VAL_16:.*]] = AIE.tile(6, 1)
// CHECK:           %[[VAL_17:.*]] = AIE.tile(6, 0)
// CHECK:           %[[VAL_18:.*]] = AIE.tile(0, 1)
// CHECK:           %[[VAL_19:.*]] = AIE.tile(7, 3)
// CHECK:           %[[VAL_20:.*]] = AIE.lock(%[[VAL_19]], 1)
// CHECK:           %[[VAL_21:.*]] = AIE.lock(%[[VAL_19]], 3)
// CHECK:           %[[VAL_22:.*]] = AIE.buffer(%[[VAL_19]]) {sym_name = "buf8"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_23:.*]] = AIE.lock(%[[VAL_19]], 2)
// CHECK:           %[[VAL_24:.*]] = AIE.buffer(%[[VAL_19]]) {sym_name = "buf7"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_25:.*]] = AIE.lock(%[[VAL_19]], 0)
// CHECK:           %[[VAL_26:.*]] = AIE.buffer(%[[VAL_19]]) {sym_name = "buf6"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_27:.*]] = AIE.mem(%[[VAL_19]]) {
// CHECK:             %[[VAL_28:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_25]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_26]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_25]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_20]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_22]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_20]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_29:.*]] = AIE.dmaStart(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             AIE.useLock(%[[VAL_23]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_24]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_23]], Release, 1)
// CHECK:             AIE.nextBd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_30:.*]] = AIE.dmaStart(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             AIE.useLock(%[[VAL_21]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_22]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_21]], Release, 0)
// CHECK:             AIE.nextBd ^bb6
// CHECK:           ^bb7:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = AIE.tile(3, 2)
// CHECK:           %[[VAL_32:.*]] = AIE.tile(3, 1)
// CHECK:           %[[VAL_33:.*]] = AIE.tile(3, 0)
// CHECK:           %[[VAL_34:.*]] = AIE.tile(1, 0)
// CHECK:           %[[VAL_35:.*]] = AIE.tile(8, 2)
// CHECK:           %[[VAL_36:.*]] = AIE.lock(%[[VAL_35]], 1)
// CHECK:           %[[VAL_37:.*]] = AIE.lock(%[[VAL_35]], 3)
// CHECK:           %[[VAL_38:.*]] = AIE.buffer(%[[VAL_35]]) {sym_name = "buf5"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_39:.*]] = AIE.lock(%[[VAL_35]], 2)
// CHECK:           %[[VAL_40:.*]] = AIE.buffer(%[[VAL_35]]) {sym_name = "buf4"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_41:.*]] = AIE.lock(%[[VAL_35]], 0)
// CHECK:           %[[VAL_42:.*]] = AIE.buffer(%[[VAL_35]]) {sym_name = "buf3"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_43:.*]] = AIE.mem(%[[VAL_35]]) {
// CHECK:             %[[VAL_44:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_41]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_42]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_41]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_36]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_38]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_36]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_45:.*]] = AIE.dmaStart(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             AIE.useLock(%[[VAL_39]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_40]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_39]], Release, 1)
// CHECK:             AIE.nextBd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_46:.*]] = AIE.dmaStart(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             AIE.useLock(%[[VAL_37]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_38]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_37]], Release, 0)
// CHECK:             AIE.nextBd ^bb6
// CHECK:           ^bb7:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_47:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_48:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_49:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_50:.*]] = AIE.tile(0, 0)
// CHECK:           %[[VAL_51:.*]] = AIE.tile(7, 2)
// CHECK:           %[[VAL_52:.*]] = AIE.lock(%[[VAL_51]], 1)
// CHECK:           %[[VAL_53:.*]] = AIE.lock(%[[VAL_51]], 3)
// CHECK:           %[[VAL_54:.*]] = AIE.buffer(%[[VAL_51]]) {sym_name = "buf2"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_55:.*]] = AIE.lock(%[[VAL_51]], 2)
// CHECK:           %[[VAL_56:.*]] = AIE.buffer(%[[VAL_51]]) {sym_name = "buf1"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_57:.*]] = AIE.lock(%[[VAL_51]], 0)
// CHECK:           %[[VAL_58:.*]] = AIE.buffer(%[[VAL_51]]) {sym_name = "buf0"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_59:.*]] = AIE.mem(%[[VAL_51]]) {
// CHECK:             %[[VAL_60:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             AIE.useLock(%[[VAL_57]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_58]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_57]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.useLock(%[[VAL_52]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_54]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_52]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_61:.*]] = AIE.dmaStart(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             AIE.useLock(%[[VAL_55]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_56]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_55]], Release, 1)
// CHECK:             AIE.nextBd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_62:.*]] = AIE.dmaStart(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             AIE.useLock(%[[VAL_53]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_54]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_53]], Release, 0)
// CHECK:             AIE.nextBd ^bb6
// CHECK:           ^bb7:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_63:.*]] = AIE.switchbox(%[[VAL_49]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:             AIE.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_64:.*]] = AIE.switchbox(%[[VAL_48]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, East : 1>
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_65:.*]] = AIE.switchbox(%[[VAL_32]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<South : 0, East : 2>
// CHECK:             AIE.connect<South : 1, East : 3>
// CHECK:             AIE.connect<North : 1, South : 0>
// CHECK:             AIE.connect<East : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_66:.*]] = AIE.tile(4, 1)
// CHECK:           %[[VAL_67:.*]] = AIE.switchbox(%[[VAL_66]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<West : 1, North : 1>
// CHECK:             AIE.connect<West : 2, East : 0>
// CHECK:             AIE.connect<West : 3, East : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_68:.*]] = AIE.tile(4, 2)
// CHECK:           %[[VAL_69:.*]] = AIE.switchbox(%[[VAL_68]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, East : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_70:.*]] = AIE.tile(5, 2)
// CHECK:           %[[VAL_71:.*]] = AIE.switchbox(%[[VAL_70]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<South : 0, East : 2>
// CHECK:             AIE.connect<South : 1, East : 3>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_72:.*]] = AIE.switchbox(%[[VAL_15]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<West : 2, East : 2>
// CHECK:             AIE.connect<West : 3, East : 3>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<East : 2, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_73:.*]] = AIE.switchbox(%[[VAL_51]]) {
// CHECK:             AIE.connect<West : 0, DMA : 0>
// CHECK:             AIE.connect<West : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, West : 0>
// CHECK:             AIE.connect<West : 2, East : 0>
// CHECK:             AIE.connect<West : 3, East : 1>
// CHECK:             AIE.connect<East : 0, West : 1>
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:             AIE.connect<South : 2, East : 2>
// CHECK:             AIE.connect<South : 3, East : 3>
// CHECK:             AIE.connect<East : 1, West : 2>
// CHECK:           }
// CHECK:           %[[VAL_74:.*]] = AIE.switchbox(%[[VAL_31]]) {
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:             AIE.connect<East : 1, West : 0>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_75:.*]] = AIE.switchbox(%[[VAL_33]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:             AIE.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_76:.*]] = AIE.tile(5, 1)
// CHECK:           %[[VAL_77:.*]] = AIE.switchbox(%[[VAL_76]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<West : 1, North : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_78:.*]] = AIE.switchbox(%[[VAL_35]]) {
// CHECK:             AIE.connect<West : 0, DMA : 0>
// CHECK:             AIE.connect<West : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, West : 0>
// CHECK:             AIE.connect<West : 2, North : 0>
// CHECK:             AIE.connect<West : 3, North : 1>
// CHECK:             AIE.connect<North : 0, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_79:.*]] = AIE.switchbox(%[[VAL_47]]) {
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_80:.*]] = AIE.switchbox(%[[VAL_17]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:             AIE.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_81:.*]] = AIE.switchbox(%[[VAL_16]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, East : 1>
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_82:.*]] = AIE.switchbox(%[[VAL_0]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<West : 1, North : 1>
// CHECK:             AIE.connect<South : 0, North : 2>
// CHECK:             AIE.connect<South : 1, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_83:.*]] = AIE.switchbox(%[[VAL_19]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<South : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_84:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_85:.*]] = AIE.switchbox(%[[VAL_84]]) {
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_86:.*]] = AIE.tile(4, 3)
// CHECK:           %[[VAL_87:.*]] = AIE.switchbox(%[[VAL_86]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_88:.*]] = AIE.tile(5, 3)
// CHECK:           %[[VAL_89:.*]] = AIE.switchbox(%[[VAL_88]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_90:.*]] = AIE.tile(6, 3)
// CHECK:           %[[VAL_91:.*]] = AIE.switchbox(%[[VAL_90]]) {
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_92:.*]] = AIE.switchbox(%[[VAL_1]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:             AIE.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_93:.*]] = AIE.switchbox(%[[VAL_3]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<South : 1, DMA : 1>
// CHECK:             AIE.connect<DMA : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_94:.*]] = AIE.shimmux(%[[VAL_49]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_95:.*]] = AIE.shimmux(%[[VAL_33]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_96:.*]] = AIE.shimmux(%[[VAL_17]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_97:.*]] = AIE.shimmux(%[[VAL_1]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_98:.*]] : North, %[[VAL_99:.*]] : South)
// CHECK:           AIE.wire(%[[VAL_49]] : DMA, %[[VAL_98]] : DMA)
// CHECK:           AIE.wire(%[[VAL_48]] : Core, %[[VAL_100:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_48]] : DMA, %[[VAL_100]] : DMA)
// CHECK:           AIE.wire(%[[VAL_99]] : North, %[[VAL_100]] : South)
// CHECK:           AIE.wire(%[[VAL_47]] : Core, %[[VAL_101:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_47]] : DMA, %[[VAL_101]] : DMA)
// CHECK:           AIE.wire(%[[VAL_100]] : North, %[[VAL_101]] : South)
// CHECK:           AIE.wire(%[[VAL_99]] : East, %[[VAL_102:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_103:.*]] : North, %[[VAL_102]] : South)
// CHECK:           AIE.wire(%[[VAL_33]] : DMA, %[[VAL_103]] : DMA)
// CHECK:           AIE.wire(%[[VAL_100]] : East, %[[VAL_104:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_32]] : Core, %[[VAL_104]] : Core)
// CHECK:           AIE.wire(%[[VAL_32]] : DMA, %[[VAL_104]] : DMA)
// CHECK:           AIE.wire(%[[VAL_102]] : North, %[[VAL_104]] : South)
// CHECK:           AIE.wire(%[[VAL_101]] : East, %[[VAL_105:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_31]] : Core, %[[VAL_105]] : Core)
// CHECK:           AIE.wire(%[[VAL_31]] : DMA, %[[VAL_105]] : DMA)
// CHECK:           AIE.wire(%[[VAL_104]] : North, %[[VAL_105]] : South)
// CHECK:           AIE.wire(%[[VAL_84]] : Core, %[[VAL_106:.*]] : Core)
// CHECK:           AIE.wire(%[[VAL_84]] : DMA, %[[VAL_106]] : DMA)
// CHECK:           AIE.wire(%[[VAL_105]] : North, %[[VAL_106]] : South)
// CHECK:           AIE.wire(%[[VAL_104]] : East, %[[VAL_107:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_66]] : Core, %[[VAL_107]] : Core)
// CHECK:           AIE.wire(%[[VAL_66]] : DMA, %[[VAL_107]] : DMA)
// CHECK:           AIE.wire(%[[VAL_105]] : East, %[[VAL_108:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_68]] : Core, %[[VAL_108]] : Core)
// CHECK:           AIE.wire(%[[VAL_68]] : DMA, %[[VAL_108]] : DMA)
// CHECK:           AIE.wire(%[[VAL_107]] : North, %[[VAL_108]] : South)
// CHECK:           AIE.wire(%[[VAL_106]] : East, %[[VAL_109:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_86]] : Core, %[[VAL_109]] : Core)
// CHECK:           AIE.wire(%[[VAL_86]] : DMA, %[[VAL_109]] : DMA)
// CHECK:           AIE.wire(%[[VAL_108]] : North, %[[VAL_109]] : South)
// CHECK:           AIE.wire(%[[VAL_107]] : East, %[[VAL_110:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_76]] : Core, %[[VAL_110]] : Core)
// CHECK:           AIE.wire(%[[VAL_76]] : DMA, %[[VAL_110]] : DMA)
// CHECK:           AIE.wire(%[[VAL_108]] : East, %[[VAL_111:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_70]] : Core, %[[VAL_111]] : Core)
// CHECK:           AIE.wire(%[[VAL_70]] : DMA, %[[VAL_111]] : DMA)
// CHECK:           AIE.wire(%[[VAL_110]] : North, %[[VAL_111]] : South)
// CHECK:           AIE.wire(%[[VAL_109]] : East, %[[VAL_112:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_88]] : Core, %[[VAL_112]] : Core)
// CHECK:           AIE.wire(%[[VAL_88]] : DMA, %[[VAL_112]] : DMA)
// CHECK:           AIE.wire(%[[VAL_111]] : North, %[[VAL_112]] : South)
// CHECK:           AIE.wire(%[[VAL_113:.*]] : North, %[[VAL_114:.*]] : South)
// CHECK:           AIE.wire(%[[VAL_17]] : DMA, %[[VAL_113]] : DMA)
// CHECK:           AIE.wire(%[[VAL_110]] : East, %[[VAL_115:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_16]] : Core, %[[VAL_115]] : Core)
// CHECK:           AIE.wire(%[[VAL_16]] : DMA, %[[VAL_115]] : DMA)
// CHECK:           AIE.wire(%[[VAL_114]] : North, %[[VAL_115]] : South)
// CHECK:           AIE.wire(%[[VAL_111]] : East, %[[VAL_116:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_15]] : Core, %[[VAL_116]] : Core)
// CHECK:           AIE.wire(%[[VAL_15]] : DMA, %[[VAL_116]] : DMA)
// CHECK:           AIE.wire(%[[VAL_115]] : North, %[[VAL_116]] : South)
// CHECK:           AIE.wire(%[[VAL_112]] : East, %[[VAL_117:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_90]] : Core, %[[VAL_117]] : Core)
// CHECK:           AIE.wire(%[[VAL_90]] : DMA, %[[VAL_117]] : DMA)
// CHECK:           AIE.wire(%[[VAL_116]] : North, %[[VAL_117]] : South)
// CHECK:           AIE.wire(%[[VAL_114]] : East, %[[VAL_118:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_119:.*]] : North, %[[VAL_118]] : South)
// CHECK:           AIE.wire(%[[VAL_1]] : DMA, %[[VAL_119]] : DMA)
// CHECK:           AIE.wire(%[[VAL_115]] : East, %[[VAL_120:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_0]] : Core, %[[VAL_120]] : Core)
// CHECK:           AIE.wire(%[[VAL_0]] : DMA, %[[VAL_120]] : DMA)
// CHECK:           AIE.wire(%[[VAL_118]] : North, %[[VAL_120]] : South)
// CHECK:           AIE.wire(%[[VAL_116]] : East, %[[VAL_121:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_51]] : Core, %[[VAL_121]] : Core)
// CHECK:           AIE.wire(%[[VAL_51]] : DMA, %[[VAL_121]] : DMA)
// CHECK:           AIE.wire(%[[VAL_120]] : North, %[[VAL_121]] : South)
// CHECK:           AIE.wire(%[[VAL_117]] : East, %[[VAL_122:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_19]] : Core, %[[VAL_122]] : Core)
// CHECK:           AIE.wire(%[[VAL_19]] : DMA, %[[VAL_122]] : DMA)
// CHECK:           AIE.wire(%[[VAL_121]] : North, %[[VAL_122]] : South)
// CHECK:           AIE.wire(%[[VAL_121]] : East, %[[VAL_123:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_35]] : Core, %[[VAL_123]] : Core)
// CHECK:           AIE.wire(%[[VAL_35]] : DMA, %[[VAL_123]] : DMA)
// CHECK:           AIE.wire(%[[VAL_122]] : East, %[[VAL_124:.*]] : West)
// CHECK:           AIE.wire(%[[VAL_3]] : Core, %[[VAL_124]] : Core)
// CHECK:           AIE.wire(%[[VAL_3]] : DMA, %[[VAL_124]] : DMA)
// CHECK:           AIE.wire(%[[VAL_123]] : North, %[[VAL_124]] : South)
// CHECK:         }

module @aie.herd_0  {
  AIE.device(xcvc1902) {
    %0 = AIE.tile(7, 1)
    %1 = AIE.tile(7, 0)
    %2 = AIE.tile(1, 1)
    %3 = AIE.tile(8, 3)
    %4 = AIE.lock(%3, 1)
    %5 = AIE.lock(%3, 3)
    %6 = AIE.buffer(%3) {sym_name = "buf11"} : memref<16x16xf32, 2>
    %7 = AIE.lock(%3, 2)
    %8 = AIE.buffer(%3) {sym_name = "buf10"} : memref<16x16xf32, 2>
    %9 = AIE.lock(%3, 0)
    %10 = AIE.buffer(%3) {sym_name = "buf9"} : memref<16x16xf32, 2>
    %11 = AIE.mem(%3)  {
      %63 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      AIE.useLock(%9, Acquire, 0)
      AIE.dmaBd(<%10 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%9, Release, 1)
      AIE.nextBd ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%4, Acquire, 0)
      AIE.dmaBd(<%6 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%4, Release, 1)
      AIE.nextBd ^bb1
    ^bb3:  // pred: ^bb5
      %64 = AIE.dmaStart(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      AIE.useLock(%7, Acquire, 0)
      AIE.dmaBd(<%8 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%7, Release, 1)
      AIE.nextBd ^bb4
    ^bb5:  // pred: ^bb0
      %65 = AIE.dmaStart(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      AIE.useLock(%5, Acquire, 1)
      AIE.dmaBd(<%6 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%5, Release, 0)
      AIE.nextBd ^bb6
    ^bb7:  // pred: ^bb3
      AIE.end
    }
    %13 = AIE.tile(6, 2)
    %14 = AIE.tile(6, 1)
    %15 = AIE.tile(6, 0)
    %16 = AIE.tile(0, 1)
    %17 = AIE.tile(7, 3)
    %18 = AIE.lock(%17, 1)
    %19 = AIE.lock(%17, 3)
    %20 = AIE.buffer(%17) {sym_name = "buf8"} : memref<16x16xf32, 2>
    %21 = AIE.lock(%17, 2)
    %22 = AIE.buffer(%17) {sym_name = "buf7"} : memref<16x16xf32, 2>
    %23 = AIE.lock(%17, 0)
    %24 = AIE.buffer(%17) {sym_name = "buf6"} : memref<16x16xf32, 2>
    %25 = AIE.mem(%17)  {
      %63 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      AIE.useLock(%23, Acquire, 0)
      AIE.dmaBd(<%24 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%23, Release, 1)
      AIE.nextBd ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%18, Acquire, 0)
      AIE.dmaBd(<%20 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%18, Release, 1)
      AIE.nextBd ^bb1
    ^bb3:  // pred: ^bb5
      %64 = AIE.dmaStart(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      AIE.useLock(%21, Acquire, 0)
      AIE.dmaBd(<%22 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%21, Release, 1)
      AIE.nextBd ^bb4
    ^bb5:  // pred: ^bb0
      %65 = AIE.dmaStart(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      AIE.useLock(%19, Acquire, 1)
      AIE.dmaBd(<%20 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%19, Release, 0)
      AIE.nextBd ^bb6
    ^bb7:  // pred: ^bb3
      AIE.end
    }
    %27 = AIE.tile(3, 2)
    %28 = AIE.tile(3, 1)
    %29 = AIE.tile(3, 0)
    %30 = AIE.tile(1, 0)
    %31 = AIE.tile(8, 2)
    %32 = AIE.lock(%31, 1)
    %33 = AIE.lock(%31, 3)
    %34 = AIE.buffer(%31) {sym_name = "buf5"} : memref<16x16xf32, 2>
    %35 = AIE.lock(%31, 2)
    %36 = AIE.buffer(%31) {sym_name = "buf4"} : memref<16x16xf32, 2>
    %37 = AIE.lock(%31, 0)
    %38 = AIE.buffer(%31) {sym_name = "buf3"} : memref<16x16xf32, 2>
    %39 = AIE.mem(%31)  {
      %63 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      AIE.useLock(%37, Acquire, 0)
      AIE.dmaBd(<%38 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%37, Release, 1)
      AIE.nextBd ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%32, Acquire, 0)
      AIE.dmaBd(<%34 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%32, Release, 1)
      AIE.nextBd ^bb1
    ^bb3:  // pred: ^bb5
      %64 = AIE.dmaStart(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      AIE.useLock(%35, Acquire, 0)
      AIE.dmaBd(<%36 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%35, Release, 1)
      AIE.nextBd ^bb4
    ^bb5:  // pred: ^bb0
      %65 = AIE.dmaStart(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      AIE.useLock(%33, Acquire, 1)
      AIE.dmaBd(<%34 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%33, Release, 0)
      AIE.nextBd ^bb6
    ^bb7:  // pred: ^bb3
      AIE.end
    }
    %41 = AIE.tile(2, 2)
    %42 = AIE.tile(2, 1)
    %43 = AIE.tile(2, 0)
    %44 = AIE.tile(0, 0)
    %45 = AIE.tile(7, 2)
    %46 = AIE.lock(%45, 1)
    %47 = AIE.lock(%45, 3)
    %48 = AIE.buffer(%45) {sym_name = "buf2"} : memref<16x16xf32, 2>
    %49 = AIE.lock(%45, 2)
    %50 = AIE.buffer(%45) {sym_name = "buf1"} : memref<16x16xf32, 2>
    %51 = AIE.lock(%45, 0)
    %52 = AIE.buffer(%45) {sym_name = "buf0"} : memref<16x16xf32, 2>
    %53 = AIE.mem(%45)  {
      %63 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      AIE.useLock(%51, Acquire, 0)
      AIE.dmaBd(<%52 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%51, Release, 1)
      AIE.nextBd ^bb2
    ^bb2:  // pred: ^bb1
      AIE.useLock(%46, Acquire, 0)
      AIE.dmaBd(<%48 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%46, Release, 1)
      AIE.nextBd ^bb1
    ^bb3:  // pred: ^bb5
      %64 = AIE.dmaStart(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      AIE.useLock(%49, Acquire, 0)
      AIE.dmaBd(<%50 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%49, Release, 1)
      AIE.nextBd ^bb4
    ^bb5:  // pred: ^bb0
      %65 = AIE.dmaStart(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      AIE.useLock(%47, Acquire, 1)
      AIE.dmaBd(<%48 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.useLock(%47, Release, 0)
      AIE.nextBd ^bb6
    ^bb7:  // pred: ^bb3
      AIE.end
    }
    %55 = AIE.switchbox(%43)  {
      AIE.connect<South : 3, North : 0>
      AIE.connect<South : 7, North : 1>
      AIE.connect<North : 0, South : 2>
      AIE.connect<North : 1, South : 3>
    }
    AIE.flow(%42, South : 0, %45, DMA : 0)
    AIE.flow(%42, South : 1, %45, DMA : 1)
    AIE.flow(%45, DMA : 0, %42, South : 0)
    %56 = AIE.switchbox(%29)  {
      AIE.connect<South : 3, North : 0>
      AIE.connect<South : 7, North : 1>
      AIE.connect<North : 0, South : 2>
      AIE.connect<North : 1, South : 3>
    }
    AIE.flow(%28, South : 0, %31, DMA : 0)
    AIE.flow(%28, South : 1, %31, DMA : 1)
    AIE.flow(%31, DMA : 0, %42, South : 1)
    %57 = AIE.switchbox(%15)  {
      AIE.connect<South : 3, North : 0>
      AIE.connect<South : 7, North : 1>
      AIE.connect<North : 0, South : 2>
      AIE.connect<North : 1, South : 3>
    }
    AIE.flow(%14, South : 0, %17, DMA : 0)
    AIE.flow(%14, South : 1, %17, DMA : 1)
    AIE.flow(%17, DMA : 0, %28, South : 0)
    %58 = AIE.switchbox(%1)  {
      AIE.connect<South : 3, North : 0>
      AIE.connect<South : 7, North : 1>
      AIE.connect<North : 0, South : 2>
      AIE.connect<North : 1, South : 3>
    }
    AIE.flow(%0, South : 0, %3, DMA : 0)
    AIE.flow(%0, South : 1, %3, DMA : 1)
    AIE.flow(%3, DMA : 0, %28, South : 1)
    %59 = AIE.shimmux(%43)  {
      AIE.connect<DMA : 0, North : 3>
      AIE.connect<DMA : 1, North : 7>
      AIE.connect<North : 2, DMA : 0>
      AIE.connect<North : 3, DMA : 1>
    }
    %60 = AIE.shimmux(%29)  {
      AIE.connect<DMA : 0, North : 3>
      AIE.connect<DMA : 1, North : 7>
      AIE.connect<North : 2, DMA : 0>
      AIE.connect<North : 3, DMA : 1>
    }
    %61 = AIE.shimmux(%15)  {
      AIE.connect<DMA : 0, North : 3>
      AIE.connect<DMA : 1, North : 7>
      AIE.connect<North : 2, DMA : 0>
      AIE.connect<North : 3, DMA : 1>
    }
    %62 = AIE.shimmux(%1)  {
      AIE.connect<DMA : 0, North : 3>
      AIE.connect<DMA : 1, North : 7>
      AIE.connect<North : 2, DMA : 0>
      AIE.connect<North : 3, DMA : 1>
    }
  }
}
