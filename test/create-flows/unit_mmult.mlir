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

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(7, 1)
// CHECK:           %[[VAL_1:.*]] = aie.tile(7, 0)
// CHECK:           %[[VAL_2:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_3:.*]] = aie.tile(8, 3)
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_3]], 1)
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_3]], 3)
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "buf11"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_3]], 2)
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "buf10"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_3]], 0)
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "buf9"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_11:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_12:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_9]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_9]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_4]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_13:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_7]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_14:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_5]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_5]], Release, 0)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = aie.tile(6, 2)
// CHECK:           %[[VAL_16:.*]] = aie.tile(6, 1)
// CHECK:           %[[VAL_17:.*]] = aie.tile(6, 0)
// CHECK:           %[[VAL_18:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_19:.*]] = aie.tile(7, 3)
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_19]], 1)
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_19]], 3)
// CHECK:           %[[VAL_22:.*]] = aie.buffer(%[[VAL_19]]) {sym_name = "buf8"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_23:.*]] = aie.lock(%[[VAL_19]], 2)
// CHECK:           %[[VAL_24:.*]] = aie.buffer(%[[VAL_19]]) {sym_name = "buf7"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_25:.*]] = aie.lock(%[[VAL_19]], 0)
// CHECK:           %[[VAL_26:.*]] = aie.buffer(%[[VAL_19]]) {sym_name = "buf6"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_27:.*]] = aie.mem(%[[VAL_19]]) {
// CHECK:             %[[VAL_28:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_25]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_26]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_25]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_20]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_22]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_20]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_29:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_23]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_24]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_23]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_30:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_21]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_22]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_21]], Release, 0)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = aie.tile(3, 2)
// CHECK:           %[[VAL_32:.*]] = aie.tile(3, 1)
// CHECK:           %[[VAL_33:.*]] = aie.tile(3, 0)
// CHECK:           %[[VAL_34:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_35:.*]] = aie.tile(8, 2)
// CHECK:           %[[VAL_36:.*]] = aie.lock(%[[VAL_35]], 1)
// CHECK:           %[[VAL_37:.*]] = aie.lock(%[[VAL_35]], 3)
// CHECK:           %[[VAL_38:.*]] = aie.buffer(%[[VAL_35]]) {sym_name = "buf5"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_39:.*]] = aie.lock(%[[VAL_35]], 2)
// CHECK:           %[[VAL_40:.*]] = aie.buffer(%[[VAL_35]]) {sym_name = "buf4"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_41:.*]] = aie.lock(%[[VAL_35]], 0)
// CHECK:           %[[VAL_42:.*]] = aie.buffer(%[[VAL_35]]) {sym_name = "buf3"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_43:.*]] = aie.mem(%[[VAL_35]]) {
// CHECK:             %[[VAL_44:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_41]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_42]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_41]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_36]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_38]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_36]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_45:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_39]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_40]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_39]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_46:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_37]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_38]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_37]], Release, 0)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_47:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_48:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_49:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_50:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_51:.*]] = aie.tile(7, 2)
// CHECK:           %[[VAL_52:.*]] = aie.lock(%[[VAL_51]], 1)
// CHECK:           %[[VAL_53:.*]] = aie.lock(%[[VAL_51]], 3)
// CHECK:           %[[VAL_54:.*]] = aie.buffer(%[[VAL_51]]) {sym_name = "buf2"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_55:.*]] = aie.lock(%[[VAL_51]], 2)
// CHECK:           %[[VAL_56:.*]] = aie.buffer(%[[VAL_51]]) {sym_name = "buf1"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_57:.*]] = aie.lock(%[[VAL_51]], 0)
// CHECK:           %[[VAL_58:.*]] = aie.buffer(%[[VAL_51]]) {sym_name = "buf0"} : memref<16x16xf32, 2>
// CHECK:           %[[VAL_59:.*]] = aie.mem(%[[VAL_51]]) {
// CHECK:             %[[VAL_60:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_57]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_58]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_57]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_52]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_54]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_52]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_61:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_55]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_56]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_55]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_62:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_53]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_54]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[VAL_53]], Release, 0)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_63:.*]] = aie.switchbox(%[[VAL_49]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_64:.*]] = aie.switchbox(%[[VAL_48]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_65:.*]] = aie.switchbox(%[[VAL_32]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<South : 0, East : 2>
// CHECK:             aie.connect<South : 1, East : 3>
// CHECK:             aie.connect<North : 1, South : 0>
// CHECK:             aie.connect<East : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_66:.*]] = aie.tile(4, 1)
// CHECK:           %[[VAL_67:.*]] = aie.switchbox(%[[VAL_66]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, North : 1>
// CHECK:             aie.connect<West : 2, East : 0>
// CHECK:             aie.connect<West : 3, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_68:.*]] = aie.tile(4, 2)
// CHECK:           %[[VAL_69:.*]] = aie.switchbox(%[[VAL_68]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_70:.*]] = aie.tile(5, 2)
// CHECK:           %[[VAL_71:.*]] = aie.switchbox(%[[VAL_70]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<South : 0, East : 2>
// CHECK:             aie.connect<South : 1, East : 3>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_72:.*]] = aie.switchbox(%[[VAL_15]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 2, East : 2>
// CHECK:             aie.connect<West : 3, East : 3>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_73:.*]] = aie.switchbox(%[[VAL_51]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<West : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:             aie.connect<West : 2, East : 0>
// CHECK:             aie.connect<West : 3, East : 1>
// CHECK:             aie.connect<East : 0, West : 1>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<South : 2, East : 2>
// CHECK:             aie.connect<South : 3, East : 3>
// CHECK:             aie.connect<East : 1, West : 2>
// CHECK:           }
// CHECK:           %[[VAL_74:.*]] = aie.switchbox(%[[VAL_31]]) {
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:             aie.connect<East : 1, West : 0>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_75:.*]] = aie.switchbox(%[[VAL_33]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_76:.*]] = aie.tile(5, 1)
// CHECK:           %[[VAL_77:.*]] = aie.switchbox(%[[VAL_76]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, North : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_78:.*]] = aie.switchbox(%[[VAL_35]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<West : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:             aie.connect<West : 2, North : 0>
// CHECK:             aie.connect<West : 3, North : 1>
// CHECK:             aie.connect<North : 0, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_79:.*]] = aie.switchbox(%[[VAL_47]]) {
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_80:.*]] = aie.switchbox(%[[VAL_17]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_81:.*]] = aie.switchbox(%[[VAL_16]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_82:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, North : 1>
// CHECK:             aie.connect<South : 0, North : 2>
// CHECK:             aie.connect<South : 1, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_83:.*]] = aie.switchbox(%[[VAL_19]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_84:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_85:.*]] = aie.switchbox(%[[VAL_84]]) {
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_86:.*]] = aie.tile(4, 3)
// CHECK:           %[[VAL_87:.*]] = aie.switchbox(%[[VAL_86]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_88:.*]] = aie.tile(5, 3)
// CHECK:           %[[VAL_89:.*]] = aie.switchbox(%[[VAL_88]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_90:.*]] = aie.tile(6, 3)
// CHECK:           %[[VAL_91:.*]] = aie.switchbox(%[[VAL_90]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_92:.*]] = aie.switchbox(%[[VAL_1]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_93:.*]] = aie.switchbox(%[[VAL_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_94:.*]] = aie.shim_mux(%[[VAL_49]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_95:.*]] = aie.shim_mux(%[[VAL_33]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_96:.*]] = aie.shim_mux(%[[VAL_17]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_97:.*]] = aie.shim_mux(%[[VAL_1]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           aie.wire(%[[VAL_98:.*]] : North, %[[VAL_99:.*]] : South)
// CHECK:           aie.wire(%[[VAL_49]] : DMA, %[[VAL_98]] : DMA)
// CHECK:           aie.wire(%[[VAL_48]] : Core, %[[VAL_100:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_48]] : DMA, %[[VAL_100]] : DMA)
// CHECK:           aie.wire(%[[VAL_99]] : North, %[[VAL_100]] : South)
// CHECK:           aie.wire(%[[VAL_47]] : Core, %[[VAL_101:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_47]] : DMA, %[[VAL_101]] : DMA)
// CHECK:           aie.wire(%[[VAL_100]] : North, %[[VAL_101]] : South)
// CHECK:           aie.wire(%[[VAL_99]] : East, %[[VAL_102:.*]] : West)
// CHECK:           aie.wire(%[[VAL_103:.*]] : North, %[[VAL_102]] : South)
// CHECK:           aie.wire(%[[VAL_33]] : DMA, %[[VAL_103]] : DMA)
// CHECK:           aie.wire(%[[VAL_100]] : East, %[[VAL_104:.*]] : West)
// CHECK:           aie.wire(%[[VAL_32]] : Core, %[[VAL_104]] : Core)
// CHECK:           aie.wire(%[[VAL_32]] : DMA, %[[VAL_104]] : DMA)
// CHECK:           aie.wire(%[[VAL_102]] : North, %[[VAL_104]] : South)
// CHECK:           aie.wire(%[[VAL_101]] : East, %[[VAL_105:.*]] : West)
// CHECK:           aie.wire(%[[VAL_31]] : Core, %[[VAL_105]] : Core)
// CHECK:           aie.wire(%[[VAL_31]] : DMA, %[[VAL_105]] : DMA)
// CHECK:           aie.wire(%[[VAL_104]] : North, %[[VAL_105]] : South)
// CHECK:           aie.wire(%[[VAL_84]] : Core, %[[VAL_106:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_84]] : DMA, %[[VAL_106]] : DMA)
// CHECK:           aie.wire(%[[VAL_105]] : North, %[[VAL_106]] : South)
// CHECK:           aie.wire(%[[VAL_104]] : East, %[[VAL_107:.*]] : West)
// CHECK:           aie.wire(%[[VAL_66]] : Core, %[[VAL_107]] : Core)
// CHECK:           aie.wire(%[[VAL_66]] : DMA, %[[VAL_107]] : DMA)
// CHECK:           aie.wire(%[[VAL_105]] : East, %[[VAL_108:.*]] : West)
// CHECK:           aie.wire(%[[VAL_68]] : Core, %[[VAL_108]] : Core)
// CHECK:           aie.wire(%[[VAL_68]] : DMA, %[[VAL_108]] : DMA)
// CHECK:           aie.wire(%[[VAL_107]] : North, %[[VAL_108]] : South)
// CHECK:           aie.wire(%[[VAL_106]] : East, %[[VAL_109:.*]] : West)
// CHECK:           aie.wire(%[[VAL_86]] : Core, %[[VAL_109]] : Core)
// CHECK:           aie.wire(%[[VAL_86]] : DMA, %[[VAL_109]] : DMA)
// CHECK:           aie.wire(%[[VAL_108]] : North, %[[VAL_109]] : South)
// CHECK:           aie.wire(%[[VAL_107]] : East, %[[VAL_110:.*]] : West)
// CHECK:           aie.wire(%[[VAL_76]] : Core, %[[VAL_110]] : Core)
// CHECK:           aie.wire(%[[VAL_76]] : DMA, %[[VAL_110]] : DMA)
// CHECK:           aie.wire(%[[VAL_108]] : East, %[[VAL_111:.*]] : West)
// CHECK:           aie.wire(%[[VAL_70]] : Core, %[[VAL_111]] : Core)
// CHECK:           aie.wire(%[[VAL_70]] : DMA, %[[VAL_111]] : DMA)
// CHECK:           aie.wire(%[[VAL_110]] : North, %[[VAL_111]] : South)
// CHECK:           aie.wire(%[[VAL_109]] : East, %[[VAL_112:.*]] : West)
// CHECK:           aie.wire(%[[VAL_88]] : Core, %[[VAL_112]] : Core)
// CHECK:           aie.wire(%[[VAL_88]] : DMA, %[[VAL_112]] : DMA)
// CHECK:           aie.wire(%[[VAL_111]] : North, %[[VAL_112]] : South)
// CHECK:           aie.wire(%[[VAL_113:.*]] : North, %[[VAL_114:.*]] : South)
// CHECK:           aie.wire(%[[VAL_17]] : DMA, %[[VAL_113]] : DMA)
// CHECK:           aie.wire(%[[VAL_110]] : East, %[[VAL_115:.*]] : West)
// CHECK:           aie.wire(%[[VAL_16]] : Core, %[[VAL_115]] : Core)
// CHECK:           aie.wire(%[[VAL_16]] : DMA, %[[VAL_115]] : DMA)
// CHECK:           aie.wire(%[[VAL_114]] : North, %[[VAL_115]] : South)
// CHECK:           aie.wire(%[[VAL_111]] : East, %[[VAL_116:.*]] : West)
// CHECK:           aie.wire(%[[VAL_15]] : Core, %[[VAL_116]] : Core)
// CHECK:           aie.wire(%[[VAL_15]] : DMA, %[[VAL_116]] : DMA)
// CHECK:           aie.wire(%[[VAL_115]] : North, %[[VAL_116]] : South)
// CHECK:           aie.wire(%[[VAL_112]] : East, %[[VAL_117:.*]] : West)
// CHECK:           aie.wire(%[[VAL_90]] : Core, %[[VAL_117]] : Core)
// CHECK:           aie.wire(%[[VAL_90]] : DMA, %[[VAL_117]] : DMA)
// CHECK:           aie.wire(%[[VAL_116]] : North, %[[VAL_117]] : South)
// CHECK:           aie.wire(%[[VAL_114]] : East, %[[VAL_118:.*]] : West)
// CHECK:           aie.wire(%[[VAL_119:.*]] : North, %[[VAL_118]] : South)
// CHECK:           aie.wire(%[[VAL_1]] : DMA, %[[VAL_119]] : DMA)
// CHECK:           aie.wire(%[[VAL_115]] : East, %[[VAL_120:.*]] : West)
// CHECK:           aie.wire(%[[VAL_0]] : Core, %[[VAL_120]] : Core)
// CHECK:           aie.wire(%[[VAL_0]] : DMA, %[[VAL_120]] : DMA)
// CHECK:           aie.wire(%[[VAL_118]] : North, %[[VAL_120]] : South)
// CHECK:           aie.wire(%[[VAL_116]] : East, %[[VAL_121:.*]] : West)
// CHECK:           aie.wire(%[[VAL_51]] : Core, %[[VAL_121]] : Core)
// CHECK:           aie.wire(%[[VAL_51]] : DMA, %[[VAL_121]] : DMA)
// CHECK:           aie.wire(%[[VAL_120]] : North, %[[VAL_121]] : South)
// CHECK:           aie.wire(%[[VAL_117]] : East, %[[VAL_122:.*]] : West)
// CHECK:           aie.wire(%[[VAL_19]] : Core, %[[VAL_122]] : Core)
// CHECK:           aie.wire(%[[VAL_19]] : DMA, %[[VAL_122]] : DMA)
// CHECK:           aie.wire(%[[VAL_121]] : North, %[[VAL_122]] : South)
// CHECK:           aie.wire(%[[VAL_121]] : East, %[[VAL_123:.*]] : West)
// CHECK:           aie.wire(%[[VAL_35]] : Core, %[[VAL_123]] : Core)
// CHECK:           aie.wire(%[[VAL_35]] : DMA, %[[VAL_123]] : DMA)
// CHECK:           aie.wire(%[[VAL_122]] : East, %[[VAL_124:.*]] : West)
// CHECK:           aie.wire(%[[VAL_3]] : Core, %[[VAL_124]] : Core)
// CHECK:           aie.wire(%[[VAL_3]] : DMA, %[[VAL_124]] : DMA)
// CHECK:           aie.wire(%[[VAL_123]] : North, %[[VAL_124]] : South)
// CHECK:         }

module @aie.herd_0 {
  aie.device(xcvc1902) {
    %tile_7_1 = aie.tile(7, 1)
    %tile_7_0 = aie.tile(7, 0)
    %tile_1_1 = aie.tile(1, 1)
    %tile_8_3 = aie.tile(8, 3)
    %lock_8_3 = aie.lock(%tile_8_3, 1)
    %lock_8_3_0 = aie.lock(%tile_8_3, 3)
    %buffer_8_3 = aie.buffer(%tile_8_3) {sym_name = "buf11"} : memref<16x16xf32, 2>
    %lock_8_3_1 = aie.lock(%tile_8_3, 2)
    %buffer_8_3_2 = aie.buffer(%tile_8_3) {sym_name = "buf10"} : memref<16x16xf32, 2>
    %lock_8_3_3 = aie.lock(%tile_8_3, 0)
    %buffer_8_3_4 = aie.buffer(%tile_8_3) {sym_name = "buf9"} : memref<16x16xf32, 2>
    %mem_8_3 = aie.mem(%tile_8_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_8_3_3, Acquire, 0)
      aie.dma_bd(%buffer_8_3_4 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_8_3_3, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_8_3, Acquire, 0)
      aie.dma_bd(%buffer_8_3 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_8_3, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_8_3_1, Acquire, 0)
      aie.dma_bd(%buffer_8_3_2 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_8_3_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_8_3_0, Acquire, 1)
      aie.dma_bd(%buffer_8_3 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_8_3_0, Release, 0)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      aie.end
    }
    %tile_6_2 = aie.tile(6, 2)
    %tile_6_1 = aie.tile(6, 1)
    %tile_6_0 = aie.tile(6, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_7_3 = aie.tile(7, 3)
    %lock_7_3 = aie.lock(%tile_7_3, 1)
    %lock_7_3_5 = aie.lock(%tile_7_3, 3)
    %buffer_7_3 = aie.buffer(%tile_7_3) {sym_name = "buf8"} : memref<16x16xf32, 2>
    %lock_7_3_6 = aie.lock(%tile_7_3, 2)
    %buffer_7_3_7 = aie.buffer(%tile_7_3) {sym_name = "buf7"} : memref<16x16xf32, 2>
    %lock_7_3_8 = aie.lock(%tile_7_3, 0)
    %buffer_7_3_9 = aie.buffer(%tile_7_3) {sym_name = "buf6"} : memref<16x16xf32, 2>
    %mem_7_3 = aie.mem(%tile_7_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_7_3_8, Acquire, 0)
      aie.dma_bd(%buffer_7_3_9 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_7_3_8, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_7_3, Acquire, 0)
      aie.dma_bd(%buffer_7_3 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_7_3, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_3_6, Acquire, 0)
      aie.dma_bd(%buffer_7_3_7 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_7_3_6, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_3_5, Acquire, 1)
      aie.dma_bd(%buffer_7_3 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_7_3_5, Release, 0)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      aie.end
    }
    %tile_3_2 = aie.tile(3, 2)
    %tile_3_1 = aie.tile(3, 1)
    %tile_3_0 = aie.tile(3, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_8_2 = aie.tile(8, 2)
    %lock_8_2 = aie.lock(%tile_8_2, 1)
    %lock_8_2_10 = aie.lock(%tile_8_2, 3)
    %buffer_8_2 = aie.buffer(%tile_8_2) {sym_name = "buf5"} : memref<16x16xf32, 2>
    %lock_8_2_11 = aie.lock(%tile_8_2, 2)
    %buffer_8_2_12 = aie.buffer(%tile_8_2) {sym_name = "buf4"} : memref<16x16xf32, 2>
    %lock_8_2_13 = aie.lock(%tile_8_2, 0)
    %buffer_8_2_14 = aie.buffer(%tile_8_2) {sym_name = "buf3"} : memref<16x16xf32, 2>
    %mem_8_2 = aie.mem(%tile_8_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_8_2_13, Acquire, 0)
      aie.dma_bd(%buffer_8_2_14 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_8_2_13, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_8_2, Acquire, 0)
      aie.dma_bd(%buffer_8_2 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_8_2, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_8_2_11, Acquire, 0)
      aie.dma_bd(%buffer_8_2_12 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_8_2_11, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_8_2_10, Acquire, 1)
      aie.dma_bd(%buffer_8_2 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_8_2_10, Release, 0)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      aie.end
    }
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_1 = aie.tile(2, 1)
    %tile_2_0 = aie.tile(2, 0)
    %tile_0_0 = aie.tile(0, 0)
    %tile_7_2 = aie.tile(7, 2)
    %lock_7_2 = aie.lock(%tile_7_2, 1)
    %lock_7_2_15 = aie.lock(%tile_7_2, 3)
    %buffer_7_2 = aie.buffer(%tile_7_2) {sym_name = "buf2"} : memref<16x16xf32, 2>
    %lock_7_2_16 = aie.lock(%tile_7_2, 2)
    %buffer_7_2_17 = aie.buffer(%tile_7_2) {sym_name = "buf1"} : memref<16x16xf32, 2>
    %lock_7_2_18 = aie.lock(%tile_7_2, 0)
    %buffer_7_2_19 = aie.buffer(%tile_7_2) {sym_name = "buf0"} : memref<16x16xf32, 2>
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_7_2_18, Acquire, 0)
      aie.dma_bd(%buffer_7_2_19 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_7_2_18, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_7_2, Acquire, 0)
      aie.dma_bd(%buffer_7_2 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_7_2, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_2_16, Acquire, 0)
      aie.dma_bd(%buffer_7_2_17 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_7_2_16, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_2_15, Acquire, 1)
      aie.dma_bd(%buffer_7_2 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%lock_7_2_15, Release, 0)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      aie.end
    }
    %switchbox_2_0 = aie.switchbox(%tile_2_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, North : 1>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    aie.flow(%tile_2_1, South : 0, %tile_7_2, DMA : 0)
    aie.flow(%tile_2_1, South : 1, %tile_7_2, DMA : 1)
    aie.flow(%tile_7_2, DMA : 0, %tile_2_1, South : 0)
    %switchbox_3_0 = aie.switchbox(%tile_3_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, North : 1>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    aie.flow(%tile_3_1, South : 0, %tile_8_2, DMA : 0)
    aie.flow(%tile_3_1, South : 1, %tile_8_2, DMA : 1)
    aie.flow(%tile_8_2, DMA : 0, %tile_2_1, South : 1)
    %switchbox_6_0 = aie.switchbox(%tile_6_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, North : 1>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    aie.flow(%tile_6_1, South : 0, %tile_7_3, DMA : 0)
    aie.flow(%tile_6_1, South : 1, %tile_7_3, DMA : 1)
    aie.flow(%tile_7_3, DMA : 0, %tile_3_1, South : 0)
    %switchbox_7_0 = aie.switchbox(%tile_7_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, North : 1>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    aie.flow(%tile_7_1, South : 0, %tile_8_3, DMA : 0)
    aie.flow(%tile_7_1, South : 1, %tile_8_3, DMA : 1)
    aie.flow(%tile_8_3, DMA : 0, %tile_3_1, South : 1)
    %shimmux_2_0 = aie.shim_mux(%tile_2_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %shimmux_3_0 = aie.shim_mux(%tile_3_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %shimmux_6_0 = aie.shim_mux(%tile_6_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %shimmux_7_0 = aie.shim_mux(%tile_7_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
  }
}
