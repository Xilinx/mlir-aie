//===- flow_test_1.mlir ----------------------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(3, 0)
// CHECK:           %[[VAL_2:.*]] = aie.tile(3, 4)
// CHECK:           %[[VAL_3:.*]] = aie.tile(4, 3)
// CHECK:           %[[VAL_4:.*]] = aie.tile(4, 4)
// CHECK:           %[[VAL_5:.*]] = aie.tile(5, 4)
// CHECK:           %[[VAL_6:.*]] = aie.tile(6, 0)
// CHECK:           %[[VAL_7:.*]] = aie.tile(6, 3)
// CHECK:           %[[VAL_8:.*]] = aie.tile(7, 0)
// CHECK:           %[[VAL_9:.*]] = aie.tile(7, 2)
// CHECK:           %[[VAL_10:.*]] = aie.tile(8, 3)
// CHECK:           %[[VAL_11:.*]] = aie.tile(8, 4)
// CHECK:           %[[VAL_12:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<East : 0, South : 3>
// CHECK:             aie.connect<East : 1, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = aie.shim_mux(%[[VAL_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_15:.*]] = aie.switchbox(%[[VAL_14]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_17:.*]] = aie.switchbox(%[[VAL_16]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = aie.tile(3, 2)
// CHECK:           %[[VAL_19:.*]] = aie.switchbox(%[[VAL_18]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<South : 0, East : 2>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = aie.tile(4, 2)
// CHECK:           %[[VAL_21:.*]] = aie.switchbox(%[[VAL_20]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<West : 2, East : 2>
// CHECK:             aie.connect<North : 0, East : 3>
// CHECK:             aie.connect<North : 1, South : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = aie.tile(5, 2)
// CHECK:           %[[VAL_23:.*]] = aie.switchbox(%[[VAL_22]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, North : 1>
// CHECK:             aie.connect<West : 2, North : 2>
// CHECK:             aie.connect<West : 3, South : 0>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:             aie.connect<North : 1, South : 2>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = aie.tile(5, 3)
// CHECK:           %[[VAL_25:.*]] = aie.switchbox(%[[VAL_24]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:             aie.connect<South : 2, North : 0>
// CHECK:             aie.connect<West : 0, East : 2>
// CHECK:             aie.connect<West : 1, South : 0>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:             aie.connect<East : 0, West : 1>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = aie.switchbox(%[[VAL_7]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<West : 1, East : 0>
// CHECK:             aie.connect<North : 0, Core : 1>
// CHECK:             aie.connect<West : 2, North : 0>
// CHECK:             aie.connect<South : 0, North : 1>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:             aie.connect<DMA : 1, South : 0>
// CHECK:             aie.connect<East : 0, West : 1>
// CHECK:             aie.connect<East : 1, West : 2>
// CHECK:             aie.connect<East : 2, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = aie.tile(7, 3)
// CHECK:           %[[VAL_28:.*]] = aie.switchbox(%[[VAL_27]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:             aie.connect<South : 2, East : 2>
// CHECK:             aie.connect<East : 0, West : 1>
// CHECK:             aie.connect<East : 1, West : 2>
// CHECK:             aie.connect<East : 2, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = aie.switchbox(%[[VAL_10]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<West : 1, North : 0>
// CHECK:             aie.connect<West : 2, Core : 1>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:             aie.connect<DMA : 1, West : 1>
// CHECK:             aie.connect<North : 0, West : 2>
// CHECK:             aie.connect<North : 1, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = aie.switchbox(%[[VAL_1]]) {
// CHECK:             aie.connect<South : 3, East : 0>
// CHECK:             aie.connect<South : 7, North : 0>
// CHECK:             aie.connect<East : 0, South : 3>
// CHECK:             aie.connect<East : 1, West : 0>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = aie.shim_mux(%[[VAL_1]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = aie.tile(4, 0)
// CHECK:           %[[VAL_33:.*]] = aie.switchbox(%[[VAL_32]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = aie.tile(5, 0)
// CHECK:           %[[VAL_35:.*]] = aie.switchbox(%[[VAL_34]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<North : 0, East : 1>
// CHECK:             aie.connect<North : 1, West : 0>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<North : 2, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = aie.switchbox(%[[VAL_6]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:             aie.connect<West : 1, South : 2>
// CHECK:             aie.connect<South : 3, North : 1>
// CHECK:             aie.connect<South : 7, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = aie.tile(6, 1)
// CHECK:           %[[VAL_38:.*]] = aie.switchbox(%[[VAL_37]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<West : 1, South : 1>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_39:.*]] = aie.tile(6, 2)
// CHECK:           %[[VAL_40:.*]] = aie.switchbox(%[[VAL_39]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = aie.switchbox(%[[VAL_9]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<Core : 0, North : 2>
// CHECK:             aie.connect<DMA : 1, West : 0>
// CHECK:             aie.connect<North : 0, Core : 1>
// CHECK:           }
// CHECK:           %[[VAL_42:.*]] = aie.tile(3, 1)
// CHECK:           %[[VAL_43:.*]] = aie.switchbox(%[[VAL_42]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_44:.*]] = aie.switchbox(%[[VAL_5]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, Core : 1>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:             aie.connect<DMA : 1, South : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_45:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK:             aie.connect<Core : 0, East : 0>
// CHECK:             aie.connect<DMA : 1, South : 0>
// CHECK:             aie.connect<East : 0, Core : 1>
// CHECK:             aie.connect<East : 1, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_46:.*]] = aie.switchbox(%[[VAL_4]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<Core : 0, East : 1>
// CHECK:             aie.connect<DMA : 1, South : 0>
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<South : 1, Core : 1>
// CHECK:           }
// CHECK:           %[[VAL_47:.*]] = aie.tile(6, 4)
// CHECK:           %[[VAL_48:.*]] = aie.switchbox(%[[VAL_47]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_49:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_50:.*]] = aie.switchbox(%[[VAL_49]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_51:.*]] = aie.switchbox(%[[VAL_3]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<Core : 0, East : 0>
// CHECK:             aie.connect<DMA : 1, East : 1>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:             aie.connect<East : 0, Core : 1>
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<East : 1, North : 0>
// CHECK:             aie.connect<East : 2, North : 1>
// CHECK:             aie.connect<East : 3, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_52:.*]] = aie.tile(5, 1)
// CHECK:           %[[VAL_53:.*]] = aie.switchbox(%[[VAL_52]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:             aie.connect<North : 1, East : 1>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<North : 2, South : 1>
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<East : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = aie.switchbox(%[[VAL_8]]) {
// CHECK:             aie.connect<West : 0, South : 2>
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<East : 0, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_55:.*]] = aie.shim_mux(%[[VAL_8]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_56:.*]] = aie.tile(7, 4)
// CHECK:           %[[VAL_57:.*]] = aie.switchbox(%[[VAL_56]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_58:.*]] = aie.switchbox(%[[VAL_11]]) {
// CHECK:             aie.connect<West : 0, Core : 1>
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:             aie.connect<DMA : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_59:.*]] = aie.shim_mux(%[[VAL_6]]) {
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_60:.*]] = aie.tile(4, 1)
// CHECK:           %[[VAL_61:.*]] = aie.switchbox(%[[VAL_60]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_62:.*]] = aie.tile(7, 1)
// CHECK:           %[[VAL_63:.*]] = aie.switchbox(%[[VAL_62]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_64:.*]] = aie.tile(8, 0)
// CHECK:           %[[VAL_65:.*]] = aie.switchbox(%[[VAL_64]]) {
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_66:.*]] = aie.tile(8, 1)
// CHECK:           %[[VAL_67:.*]] = aie.switchbox(%[[VAL_66]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_68:.*]] = aie.tile(8, 2)
// CHECK:           %[[VAL_69:.*]] = aie.switchbox(%[[VAL_68]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[VAL_70:.*]] : North, %[[VAL_71:.*]] : South)
// CHECK:           aie.wire(%[[VAL_0]] : DMA, %[[VAL_70]] : DMA)
// CHECK:           aie.wire(%[[VAL_14]] : Core, %[[VAL_72:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_14]] : DMA, %[[VAL_72]] : DMA)
// CHECK:           aie.wire(%[[VAL_71]] : North, %[[VAL_72]] : South)
// CHECK:           aie.wire(%[[VAL_16]] : Core, %[[VAL_73:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_16]] : DMA, %[[VAL_73]] : DMA)
// CHECK:           aie.wire(%[[VAL_72]] : North, %[[VAL_73]] : South)
// CHECK:           aie.wire(%[[VAL_71]] : East, %[[VAL_74:.*]] : West)
// CHECK:           aie.wire(%[[VAL_75:.*]] : North, %[[VAL_74]] : South)
// CHECK:           aie.wire(%[[VAL_1]] : DMA, %[[VAL_75]] : DMA)
// CHECK:           aie.wire(%[[VAL_72]] : East, %[[VAL_76:.*]] : West)
// CHECK:           aie.wire(%[[VAL_42]] : Core, %[[VAL_76]] : Core)
// CHECK:           aie.wire(%[[VAL_42]] : DMA, %[[VAL_76]] : DMA)
// CHECK:           aie.wire(%[[VAL_74]] : North, %[[VAL_76]] : South)
// CHECK:           aie.wire(%[[VAL_73]] : East, %[[VAL_77:.*]] : West)
// CHECK:           aie.wire(%[[VAL_18]] : Core, %[[VAL_77]] : Core)
// CHECK:           aie.wire(%[[VAL_18]] : DMA, %[[VAL_77]] : DMA)
// CHECK:           aie.wire(%[[VAL_76]] : North, %[[VAL_77]] : South)
// CHECK:           aie.wire(%[[VAL_49]] : Core, %[[VAL_78:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_49]] : DMA, %[[VAL_78]] : DMA)
// CHECK:           aie.wire(%[[VAL_77]] : North, %[[VAL_78]] : South)
// CHECK:           aie.wire(%[[VAL_2]] : Core, %[[VAL_79:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_2]] : DMA, %[[VAL_79]] : DMA)
// CHECK:           aie.wire(%[[VAL_78]] : North, %[[VAL_79]] : South)
// CHECK:           aie.wire(%[[VAL_74]] : East, %[[VAL_80:.*]] : West)
// CHECK:           aie.wire(%[[VAL_76]] : East, %[[VAL_81:.*]] : West)
// CHECK:           aie.wire(%[[VAL_60]] : Core, %[[VAL_81]] : Core)
// CHECK:           aie.wire(%[[VAL_60]] : DMA, %[[VAL_81]] : DMA)
// CHECK:           aie.wire(%[[VAL_80]] : North, %[[VAL_81]] : South)
// CHECK:           aie.wire(%[[VAL_77]] : East, %[[VAL_82:.*]] : West)
// CHECK:           aie.wire(%[[VAL_20]] : Core, %[[VAL_82]] : Core)
// CHECK:           aie.wire(%[[VAL_20]] : DMA, %[[VAL_82]] : DMA)
// CHECK:           aie.wire(%[[VAL_81]] : North, %[[VAL_82]] : South)
// CHECK:           aie.wire(%[[VAL_78]] : East, %[[VAL_83:.*]] : West)
// CHECK:           aie.wire(%[[VAL_3]] : Core, %[[VAL_83]] : Core)
// CHECK:           aie.wire(%[[VAL_3]] : DMA, %[[VAL_83]] : DMA)
// CHECK:           aie.wire(%[[VAL_82]] : North, %[[VAL_83]] : South)
// CHECK:           aie.wire(%[[VAL_79]] : East, %[[VAL_84:.*]] : West)
// CHECK:           aie.wire(%[[VAL_4]] : Core, %[[VAL_84]] : Core)
// CHECK:           aie.wire(%[[VAL_4]] : DMA, %[[VAL_84]] : DMA)
// CHECK:           aie.wire(%[[VAL_83]] : North, %[[VAL_84]] : South)
// CHECK:           aie.wire(%[[VAL_80]] : East, %[[VAL_85:.*]] : West)
// CHECK:           aie.wire(%[[VAL_81]] : East, %[[VAL_86:.*]] : West)
// CHECK:           aie.wire(%[[VAL_52]] : Core, %[[VAL_86]] : Core)
// CHECK:           aie.wire(%[[VAL_52]] : DMA, %[[VAL_86]] : DMA)
// CHECK:           aie.wire(%[[VAL_85]] : North, %[[VAL_86]] : South)
// CHECK:           aie.wire(%[[VAL_82]] : East, %[[VAL_87:.*]] : West)
// CHECK:           aie.wire(%[[VAL_22]] : Core, %[[VAL_87]] : Core)
// CHECK:           aie.wire(%[[VAL_22]] : DMA, %[[VAL_87]] : DMA)
// CHECK:           aie.wire(%[[VAL_86]] : North, %[[VAL_87]] : South)
// CHECK:           aie.wire(%[[VAL_83]] : East, %[[VAL_88:.*]] : West)
// CHECK:           aie.wire(%[[VAL_24]] : Core, %[[VAL_88]] : Core)
// CHECK:           aie.wire(%[[VAL_24]] : DMA, %[[VAL_88]] : DMA)
// CHECK:           aie.wire(%[[VAL_87]] : North, %[[VAL_88]] : South)
// CHECK:           aie.wire(%[[VAL_84]] : East, %[[VAL_89:.*]] : West)
// CHECK:           aie.wire(%[[VAL_5]] : Core, %[[VAL_89]] : Core)
// CHECK:           aie.wire(%[[VAL_5]] : DMA, %[[VAL_89]] : DMA)
// CHECK:           aie.wire(%[[VAL_88]] : North, %[[VAL_89]] : South)
// CHECK:           aie.wire(%[[VAL_85]] : East, %[[VAL_90:.*]] : West)
// CHECK:           aie.wire(%[[VAL_91:.*]] : North, %[[VAL_90]] : South)
// CHECK:           aie.wire(%[[VAL_6]] : DMA, %[[VAL_91]] : DMA)
// CHECK:           aie.wire(%[[VAL_86]] : East, %[[VAL_92:.*]] : West)
// CHECK:           aie.wire(%[[VAL_37]] : Core, %[[VAL_92]] : Core)
// CHECK:           aie.wire(%[[VAL_37]] : DMA, %[[VAL_92]] : DMA)
// CHECK:           aie.wire(%[[VAL_90]] : North, %[[VAL_92]] : South)
// CHECK:           aie.wire(%[[VAL_87]] : East, %[[VAL_93:.*]] : West)
// CHECK:           aie.wire(%[[VAL_39]] : Core, %[[VAL_93]] : Core)
// CHECK:           aie.wire(%[[VAL_39]] : DMA, %[[VAL_93]] : DMA)
// CHECK:           aie.wire(%[[VAL_92]] : North, %[[VAL_93]] : South)
// CHECK:           aie.wire(%[[VAL_88]] : East, %[[VAL_94:.*]] : West)
// CHECK:           aie.wire(%[[VAL_7]] : Core, %[[VAL_94]] : Core)
// CHECK:           aie.wire(%[[VAL_7]] : DMA, %[[VAL_94]] : DMA)
// CHECK:           aie.wire(%[[VAL_93]] : North, %[[VAL_94]] : South)
// CHECK:           aie.wire(%[[VAL_89]] : East, %[[VAL_95:.*]] : West)
// CHECK:           aie.wire(%[[VAL_47]] : Core, %[[VAL_95]] : Core)
// CHECK:           aie.wire(%[[VAL_47]] : DMA, %[[VAL_95]] : DMA)
// CHECK:           aie.wire(%[[VAL_94]] : North, %[[VAL_95]] : South)
// CHECK:           aie.wire(%[[VAL_90]] : East, %[[VAL_96:.*]] : West)
// CHECK:           aie.wire(%[[VAL_97:.*]] : North, %[[VAL_96]] : South)
// CHECK:           aie.wire(%[[VAL_8]] : DMA, %[[VAL_97]] : DMA)
// CHECK:           aie.wire(%[[VAL_92]] : East, %[[VAL_98:.*]] : West)
// CHECK:           aie.wire(%[[VAL_62]] : Core, %[[VAL_98]] : Core)
// CHECK:           aie.wire(%[[VAL_62]] : DMA, %[[VAL_98]] : DMA)
// CHECK:           aie.wire(%[[VAL_96]] : North, %[[VAL_98]] : South)
// CHECK:           aie.wire(%[[VAL_93]] : East, %[[VAL_99:.*]] : West)
// CHECK:           aie.wire(%[[VAL_9]] : Core, %[[VAL_99]] : Core)
// CHECK:           aie.wire(%[[VAL_9]] : DMA, %[[VAL_99]] : DMA)
// CHECK:           aie.wire(%[[VAL_98]] : North, %[[VAL_99]] : South)
// CHECK:           aie.wire(%[[VAL_94]] : East, %[[VAL_100:.*]] : West)
// CHECK:           aie.wire(%[[VAL_27]] : Core, %[[VAL_100]] : Core)
// CHECK:           aie.wire(%[[VAL_27]] : DMA, %[[VAL_100]] : DMA)
// CHECK:           aie.wire(%[[VAL_99]] : North, %[[VAL_100]] : South)
// CHECK:           aie.wire(%[[VAL_95]] : East, %[[VAL_101:.*]] : West)
// CHECK:           aie.wire(%[[VAL_56]] : Core, %[[VAL_101]] : Core)
// CHECK:           aie.wire(%[[VAL_56]] : DMA, %[[VAL_101]] : DMA)
// CHECK:           aie.wire(%[[VAL_100]] : North, %[[VAL_101]] : South)
// CHECK:           aie.wire(%[[VAL_96]] : East, %[[VAL_102:.*]] : West)
// CHECK:           aie.wire(%[[VAL_98]] : East, %[[VAL_103:.*]] : West)
// CHECK:           aie.wire(%[[VAL_66]] : Core, %[[VAL_103]] : Core)
// CHECK:           aie.wire(%[[VAL_66]] : DMA, %[[VAL_103]] : DMA)
// CHECK:           aie.wire(%[[VAL_102]] : North, %[[VAL_103]] : South)
// CHECK:           aie.wire(%[[VAL_99]] : East, %[[VAL_104:.*]] : West)
// CHECK:           aie.wire(%[[VAL_68]] : Core, %[[VAL_104]] : Core)
// CHECK:           aie.wire(%[[VAL_68]] : DMA, %[[VAL_104]] : DMA)
// CHECK:           aie.wire(%[[VAL_103]] : North, %[[VAL_104]] : South)
// CHECK:           aie.wire(%[[VAL_100]] : East, %[[VAL_105:.*]] : West)
// CHECK:           aie.wire(%[[VAL_10]] : Core, %[[VAL_105]] : Core)
// CHECK:           aie.wire(%[[VAL_10]] : DMA, %[[VAL_105]] : DMA)
// CHECK:           aie.wire(%[[VAL_104]] : North, %[[VAL_105]] : South)
// CHECK:           aie.wire(%[[VAL_101]] : East, %[[VAL_106:.*]] : West)
// CHECK:           aie.wire(%[[VAL_11]] : Core, %[[VAL_106]] : Core)
// CHECK:           aie.wire(%[[VAL_11]] : DMA, %[[VAL_106]] : DMA)
// CHECK:           aie.wire(%[[VAL_105]] : North, %[[VAL_106]] : South)
// CHECK:         }

module {
    aie.device(xcvc1902) {
        %t20 = aie.tile(2, 0)
        %t30 = aie.tile(3, 0)
        %t34 = aie.tile(3, 4)
        %t43 = aie.tile(4, 3)
        %t44 = aie.tile(4, 4)
        %t54 = aie.tile(5, 4)
        %t60 = aie.tile(6, 0)
        %t63 = aie.tile(6, 3)
        %t70 = aie.tile(7, 0)
        %t72 = aie.tile(7, 2)
        %t83 = aie.tile(8, 3)
        %t84 = aie.tile(8, 4)

        aie.flow(%t20, DMA : 0, %t63, DMA : 0)
        aie.flow(%t20, DMA : 1, %t83, DMA : 0)
        aie.flow(%t30, DMA : 0, %t72, DMA : 0)
        aie.flow(%t30, DMA : 1, %t54, DMA : 0)

        aie.flow(%t34, Core : 0, %t63, Core : 1)
        aie.flow(%t34, DMA : 1, %t70, DMA : 0)
        aie.flow(%t43, Core : 0, %t84, Core : 1)
        aie.flow(%t43, DMA : 1, %t60, DMA : 1)

        aie.flow(%t44, Core : 0, %t54, Core : 1)
        aie.flow(%t44, DMA : 1, %t60, DMA : 0)
        aie.flow(%t54, Core : 0, %t43, Core : 1)
        aie.flow(%t54, DMA : 1, %t30, DMA : 1)

        aie.flow(%t60, DMA : 0, %t44, DMA : 0)
        aie.flow(%t60, DMA : 1, %t43, DMA : 0)
        aie.flow(%t63, Core : 0, %t34, Core : 1)
        aie.flow(%t63, DMA : 1, %t20, DMA : 1)

        aie.flow(%t70, DMA : 0, %t34, DMA : 0)
        aie.flow(%t70, DMA : 1, %t84, DMA : 0)
        aie.flow(%t72, Core : 0, %t83, Core : 1)
        aie.flow(%t72, DMA : 1, %t30, DMA : 0)

        aie.flow(%t83, Core : 0, %t44, Core : 1)
        aie.flow(%t83, DMA : 1, %t20, DMA : 0)
        aie.flow(%t84, Core : 0, %t72, Core : 1)
        aie.flow(%t84, DMA : 1, %t70, DMA : 1)
    }
}
