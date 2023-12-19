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
// CHECK:             %[[VAL_12:.*]] = AIE.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             AIE.use_lock(%[[VAL_9]], Acquire, 0)
// CHECK:             AIE.dma_bd(<%[[VAL_10]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_9]], Release, 1)
// CHECK:             AIE.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.use_lock(%[[VAL_4]], Acquire, 0)
// CHECK:             AIE.dma_bd(<%[[VAL_6]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_4]], Release, 1)
// CHECK:             AIE.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_13:.*]] = AIE.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             AIE.use_lock(%[[VAL_7]], Acquire, 0)
// CHECK:             AIE.dma_bd(<%[[VAL_8]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             AIE.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_14:.*]] = AIE.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             AIE.use_lock(%[[VAL_5]], Acquire, 1)
// CHECK:             AIE.dma_bd(<%[[VAL_6]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_5]], Release, 0)
// CHECK:             AIE.next_bd ^bb6
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
// CHECK:             %[[VAL_28:.*]] = AIE.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             AIE.use_lock(%[[VAL_25]], Acquire, 0)
// CHECK:             AIE.dma_bd(<%[[VAL_26]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_25]], Release, 1)
// CHECK:             AIE.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.use_lock(%[[VAL_20]], Acquire, 0)
// CHECK:             AIE.dma_bd(<%[[VAL_22]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_20]], Release, 1)
// CHECK:             AIE.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_29:.*]] = AIE.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             AIE.use_lock(%[[VAL_23]], Acquire, 0)
// CHECK:             AIE.dma_bd(<%[[VAL_24]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_23]], Release, 1)
// CHECK:             AIE.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_30:.*]] = AIE.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             AIE.use_lock(%[[VAL_21]], Acquire, 1)
// CHECK:             AIE.dma_bd(<%[[VAL_22]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_21]], Release, 0)
// CHECK:             AIE.next_bd ^bb6
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
// CHECK:             %[[VAL_44:.*]] = AIE.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             AIE.use_lock(%[[VAL_41]], Acquire, 0)
// CHECK:             AIE.dma_bd(<%[[VAL_42]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_41]], Release, 1)
// CHECK:             AIE.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.use_lock(%[[VAL_36]], Acquire, 0)
// CHECK:             AIE.dma_bd(<%[[VAL_38]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_36]], Release, 1)
// CHECK:             AIE.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_45:.*]] = AIE.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             AIE.use_lock(%[[VAL_39]], Acquire, 0)
// CHECK:             AIE.dma_bd(<%[[VAL_40]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_39]], Release, 1)
// CHECK:             AIE.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_46:.*]] = AIE.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             AIE.use_lock(%[[VAL_37]], Acquire, 1)
// CHECK:             AIE.dma_bd(<%[[VAL_38]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_37]], Release, 0)
// CHECK:             AIE.next_bd ^bb6
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
// CHECK:             %[[VAL_60:.*]] = AIE.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             AIE.use_lock(%[[VAL_57]], Acquire, 0)
// CHECK:             AIE.dma_bd(<%[[VAL_58]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_57]], Release, 1)
// CHECK:             AIE.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.use_lock(%[[VAL_52]], Acquire, 0)
// CHECK:             AIE.dma_bd(<%[[VAL_54]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_52]], Release, 1)
// CHECK:             AIE.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_61:.*]] = AIE.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             AIE.use_lock(%[[VAL_55]], Acquire, 0)
// CHECK:             AIE.dma_bd(<%[[VAL_56]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_55]], Release, 1)
// CHECK:             AIE.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_62:.*]] = AIE.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             AIE.use_lock(%[[VAL_53]], Acquire, 1)
// CHECK:             AIE.dma_bd(<%[[VAL_54]] : memref<16x16xf32, 2>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_53]], Release, 0)
// CHECK:             AIE.next_bd ^bb6
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
// CHECK:           %[[VAL_94:.*]] = AIE.shim_mux(%[[VAL_49]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_95:.*]] = AIE.shim_mux(%[[VAL_33]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_96:.*]] = AIE.shim_mux(%[[VAL_17]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_97:.*]] = AIE.shim_mux(%[[VAL_1]]) {
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

module @aie.herd_0 {
  AIE.device(xcvc1902) {
    %tile_7_1 = AIE.tile(7, 1)
    %tile_7_0 = AIE.tile(7, 0)
    %tile_1_1 = AIE.tile(1, 1)
    %tile_8_3 = AIE.tile(8, 3)
    %lock_8_3 = AIE.lock(%tile_8_3, 1)
    %lock_8_3_0 = AIE.lock(%tile_8_3, 3)
    %buffer_8_3 = AIE.buffer(%tile_8_3) {sym_name = "buf11"} : memref<16x16xf32, 2>
    %lock_8_3_1 = AIE.lock(%tile_8_3, 2)
    %buffer_8_3_2 = AIE.buffer(%tile_8_3) {sym_name = "buf10"} : memref<16x16xf32, 2>
    %lock_8_3_3 = AIE.lock(%tile_8_3, 0)
    %buffer_8_3_4 = AIE.buffer(%tile_8_3) {sym_name = "buf9"} : memref<16x16xf32, 2>
    %mem_8_3 = AIE.mem(%tile_8_3) {
      %0 = AIE.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      AIE.use_lock(%lock_8_3_3, Acquire, 0)
      AIE.dma_bd(<%buffer_8_3_4 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_8_3_3, Release, 1)
      AIE.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      AIE.use_lock(%lock_8_3, Acquire, 0)
      AIE.dma_bd(<%buffer_8_3 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_8_3, Release, 1)
      AIE.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = AIE.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      AIE.use_lock(%lock_8_3_1, Acquire, 0)
      AIE.dma_bd(<%buffer_8_3_2 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_8_3_1, Release, 1)
      AIE.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = AIE.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      AIE.use_lock(%lock_8_3_0, Acquire, 1)
      AIE.dma_bd(<%buffer_8_3 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_8_3_0, Release, 0)
      AIE.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      AIE.end
    }
    %tile_6_2 = AIE.tile(6, 2)
    %tile_6_1 = AIE.tile(6, 1)
    %tile_6_0 = AIE.tile(6, 0)
    %tile_0_1 = AIE.tile(0, 1)
    %tile_7_3 = AIE.tile(7, 3)
    %lock_7_3 = AIE.lock(%tile_7_3, 1)
    %lock_7_3_5 = AIE.lock(%tile_7_3, 3)
    %buffer_7_3 = AIE.buffer(%tile_7_3) {sym_name = "buf8"} : memref<16x16xf32, 2>
    %lock_7_3_6 = AIE.lock(%tile_7_3, 2)
    %buffer_7_3_7 = AIE.buffer(%tile_7_3) {sym_name = "buf7"} : memref<16x16xf32, 2>
    %lock_7_3_8 = AIE.lock(%tile_7_3, 0)
    %buffer_7_3_9 = AIE.buffer(%tile_7_3) {sym_name = "buf6"} : memref<16x16xf32, 2>
    %mem_7_3 = AIE.mem(%tile_7_3) {
      %0 = AIE.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      AIE.use_lock(%lock_7_3_8, Acquire, 0)
      AIE.dma_bd(<%buffer_7_3_9 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_7_3_8, Release, 1)
      AIE.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      AIE.use_lock(%lock_7_3, Acquire, 0)
      AIE.dma_bd(<%buffer_7_3 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_7_3, Release, 1)
      AIE.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = AIE.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      AIE.use_lock(%lock_7_3_6, Acquire, 0)
      AIE.dma_bd(<%buffer_7_3_7 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_7_3_6, Release, 1)
      AIE.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = AIE.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      AIE.use_lock(%lock_7_3_5, Acquire, 1)
      AIE.dma_bd(<%buffer_7_3 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_7_3_5, Release, 0)
      AIE.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      AIE.end
    }
    %tile_3_2 = AIE.tile(3, 2)
    %tile_3_1 = AIE.tile(3, 1)
    %tile_3_0 = AIE.tile(3, 0)
    %tile_1_0 = AIE.tile(1, 0)
    %tile_8_2 = AIE.tile(8, 2)
    %lock_8_2 = AIE.lock(%tile_8_2, 1)
    %lock_8_2_10 = AIE.lock(%tile_8_2, 3)
    %buffer_8_2 = AIE.buffer(%tile_8_2) {sym_name = "buf5"} : memref<16x16xf32, 2>
    %lock_8_2_11 = AIE.lock(%tile_8_2, 2)
    %buffer_8_2_12 = AIE.buffer(%tile_8_2) {sym_name = "buf4"} : memref<16x16xf32, 2>
    %lock_8_2_13 = AIE.lock(%tile_8_2, 0)
    %buffer_8_2_14 = AIE.buffer(%tile_8_2) {sym_name = "buf3"} : memref<16x16xf32, 2>
    %mem_8_2 = AIE.mem(%tile_8_2) {
      %0 = AIE.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      AIE.use_lock(%lock_8_2_13, Acquire, 0)
      AIE.dma_bd(<%buffer_8_2_14 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_8_2_13, Release, 1)
      AIE.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      AIE.use_lock(%lock_8_2, Acquire, 0)
      AIE.dma_bd(<%buffer_8_2 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_8_2, Release, 1)
      AIE.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = AIE.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      AIE.use_lock(%lock_8_2_11, Acquire, 0)
      AIE.dma_bd(<%buffer_8_2_12 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_8_2_11, Release, 1)
      AIE.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = AIE.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      AIE.use_lock(%lock_8_2_10, Acquire, 1)
      AIE.dma_bd(<%buffer_8_2 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_8_2_10, Release, 0)
      AIE.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      AIE.end
    }
    %tile_2_2 = AIE.tile(2, 2)
    %tile_2_1 = AIE.tile(2, 1)
    %tile_2_0 = AIE.tile(2, 0)
    %tile_0_0 = AIE.tile(0, 0)
    %tile_7_2 = AIE.tile(7, 2)
    %lock_7_2 = AIE.lock(%tile_7_2, 1)
    %lock_7_2_15 = AIE.lock(%tile_7_2, 3)
    %buffer_7_2 = AIE.buffer(%tile_7_2) {sym_name = "buf2"} : memref<16x16xf32, 2>
    %lock_7_2_16 = AIE.lock(%tile_7_2, 2)
    %buffer_7_2_17 = AIE.buffer(%tile_7_2) {sym_name = "buf1"} : memref<16x16xf32, 2>
    %lock_7_2_18 = AIE.lock(%tile_7_2, 0)
    %buffer_7_2_19 = AIE.buffer(%tile_7_2) {sym_name = "buf0"} : memref<16x16xf32, 2>
    %mem_7_2 = AIE.mem(%tile_7_2) {
      %0 = AIE.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      AIE.use_lock(%lock_7_2_18, Acquire, 0)
      AIE.dma_bd(<%buffer_7_2_19 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_7_2_18, Release, 1)
      AIE.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      AIE.use_lock(%lock_7_2, Acquire, 0)
      AIE.dma_bd(<%buffer_7_2 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_7_2, Release, 1)
      AIE.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = AIE.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      AIE.use_lock(%lock_7_2_16, Acquire, 0)
      AIE.dma_bd(<%buffer_7_2_17 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_7_2_16, Release, 1)
      AIE.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = AIE.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      AIE.use_lock(%lock_7_2_15, Acquire, 1)
      AIE.dma_bd(<%buffer_7_2 : memref<16x16xf32, 2>, 0, 256>, 0)
      AIE.use_lock(%lock_7_2_15, Release, 0)
      AIE.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      AIE.end
    }
    %switchbox_2_0 = AIE.switchbox(%tile_2_0) {
      AIE.connect<South : 3, North : 0>
      AIE.connect<South : 7, North : 1>
      AIE.connect<North : 0, South : 2>
      AIE.connect<North : 1, South : 3>
    }
    AIE.flow(%tile_2_1, South : 0, %tile_7_2, DMA : 0)
    AIE.flow(%tile_2_1, South : 1, %tile_7_2, DMA : 1)
    AIE.flow(%tile_7_2, DMA : 0, %tile_2_1, South : 0)
    %switchbox_3_0 = AIE.switchbox(%tile_3_0) {
      AIE.connect<South : 3, North : 0>
      AIE.connect<South : 7, North : 1>
      AIE.connect<North : 0, South : 2>
      AIE.connect<North : 1, South : 3>
    }
    AIE.flow(%tile_3_1, South : 0, %tile_8_2, DMA : 0)
    AIE.flow(%tile_3_1, South : 1, %tile_8_2, DMA : 1)
    AIE.flow(%tile_8_2, DMA : 0, %tile_2_1, South : 1)
    %switchbox_6_0 = AIE.switchbox(%tile_6_0) {
      AIE.connect<South : 3, North : 0>
      AIE.connect<South : 7, North : 1>
      AIE.connect<North : 0, South : 2>
      AIE.connect<North : 1, South : 3>
    }
    AIE.flow(%tile_6_1, South : 0, %tile_7_3, DMA : 0)
    AIE.flow(%tile_6_1, South : 1, %tile_7_3, DMA : 1)
    AIE.flow(%tile_7_3, DMA : 0, %tile_3_1, South : 0)
    %switchbox_7_0 = AIE.switchbox(%tile_7_0) {
      AIE.connect<South : 3, North : 0>
      AIE.connect<South : 7, North : 1>
      AIE.connect<North : 0, South : 2>
      AIE.connect<North : 1, South : 3>
    }
    AIE.flow(%tile_7_1, South : 0, %tile_8_3, DMA : 0)
    AIE.flow(%tile_7_1, South : 1, %tile_8_3, DMA : 1)
    AIE.flow(%tile_8_3, DMA : 0, %tile_3_1, South : 1)
    %shimmux_2_0 = AIE.shim_mux(%tile_2_0) {
      AIE.connect<DMA : 0, North : 3>
      AIE.connect<DMA : 1, North : 7>
      AIE.connect<North : 2, DMA : 0>
      AIE.connect<North : 3, DMA : 1>
    }
    %shimmux_3_0 = AIE.shim_mux(%tile_3_0) {
      AIE.connect<DMA : 0, North : 3>
      AIE.connect<DMA : 1, North : 7>
      AIE.connect<North : 2, DMA : 0>
      AIE.connect<North : 3, DMA : 1>
    }
    %shimmux_6_0 = AIE.shim_mux(%tile_6_0) {
      AIE.connect<DMA : 0, North : 3>
      AIE.connect<DMA : 1, North : 7>
      AIE.connect<North : 2, DMA : 0>
      AIE.connect<North : 3, DMA : 1>
    }
    %shimmux_7_0 = AIE.shim_mux(%tile_7_0) {
      AIE.connect<DMA : 0, North : 3>
      AIE.connect<DMA : 1, North : 7>
      AIE.connect<North : 2, DMA : 0>
      AIE.connect<North : 3, DMA : 1>
    }
  }
}
