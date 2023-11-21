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

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(3, 0)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(3, 4)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(4, 3)
// CHECK:           %[[VAL_4:.*]] = AIE.tile(4, 4)
// CHECK:           %[[VAL_5:.*]] = AIE.tile(5, 4)
// CHECK:           %[[VAL_6:.*]] = AIE.tile(6, 0)
// CHECK:           %[[VAL_7:.*]] = AIE.tile(6, 3)
// CHECK:           %[[VAL_8:.*]] = AIE.tile(7, 0)
// CHECK:           %[[VAL_9:.*]] = AIE.tile(7, 2)
// CHECK:           %[[VAL_10:.*]] = AIE.tile(8, 3)
// CHECK:           %[[VAL_11:.*]] = AIE.tile(8, 4)
// CHECK:           %[[VAL_12:.*]] = AIE.switchbox(%[[VAL_0]]) {
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:             AIE.connect<East : 0, South : 3>
// CHECK:             AIE.connect<East : 1, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = AIE.shimmux(%[[VAL_0]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_15:.*]] = AIE.switchbox(%[[VAL_14]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_17:.*]] = AIE.switchbox(%[[VAL_16]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = AIE.tile(3, 2)
// CHECK:           %[[VAL_19:.*]] = AIE.switchbox(%[[VAL_18]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<South : 0, East : 2>
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = AIE.tile(4, 2)
// CHECK:           %[[VAL_21:.*]] = AIE.switchbox(%[[VAL_20]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, East : 1>
// CHECK:             AIE.connect<West : 2, East : 2>
// CHECK:             AIE.connect<North : 0, East : 3>
// CHECK:             AIE.connect<North : 1, South : 0>
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = AIE.tile(5, 2)
// CHECK:           %[[VAL_23:.*]] = AIE.switchbox(%[[VAL_22]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<West : 1, North : 1>
// CHECK:             AIE.connect<West : 2, North : 2>
// CHECK:             AIE.connect<West : 3, South : 0>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:             AIE.connect<North : 1, South : 2>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = AIE.tile(5, 3)
// CHECK:           %[[VAL_25:.*]] = AIE.switchbox(%[[VAL_24]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, East : 1>
// CHECK:             AIE.connect<South : 2, North : 0>
// CHECK:             AIE.connect<West : 0, East : 2>
// CHECK:             AIE.connect<West : 1, South : 0>
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:             AIE.connect<North : 1, South : 1>
// CHECK:             AIE.connect<East : 0, West : 1>
// CHECK:             AIE.connect<East : 1, North : 1>
// CHECK:             AIE.connect<East : 2, West : 2>
// CHECK:             AIE.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = AIE.switchbox(%[[VAL_7]]) {
// CHECK:             AIE.connect<West : 0, DMA : 0>
// CHECK:             AIE.connect<West : 1, East : 0>
// CHECK:             AIE.connect<North : 0, Core : 1>
// CHECK:             AIE.connect<West : 2, North : 0>
// CHECK:             AIE.connect<South : 0, North : 1>
// CHECK:             AIE.connect<Core : 0, West : 0>
// CHECK:             AIE.connect<DMA : 1, South : 0>
// CHECK:             AIE.connect<East : 0, West : 1>
// CHECK:             AIE.connect<East : 1, West : 2>
// CHECK:             AIE.connect<East : 2, West : 3>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = AIE.tile(7, 3)
// CHECK:           %[[VAL_28:.*]] = AIE.switchbox(%[[VAL_27]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<South : 1, East : 1>
// CHECK:             AIE.connect<South : 2, East : 2>
// CHECK:             AIE.connect<East : 0, West : 1>
// CHECK:             AIE.connect<East : 1, West : 2>
// CHECK:             AIE.connect<East : 2, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = AIE.switchbox(%[[VAL_10]]) {
// CHECK:             AIE.connect<West : 0, DMA : 0>
// CHECK:             AIE.connect<West : 1, North : 0>
// CHECK:             AIE.connect<West : 2, Core : 1>
// CHECK:             AIE.connect<Core : 0, West : 0>
// CHECK:             AIE.connect<DMA : 1, West : 1>
// CHECK:             AIE.connect<North : 0, West : 2>
// CHECK:             AIE.connect<North : 1, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = AIE.switchbox(%[[VAL_1]]) {
// CHECK:             AIE.connect<South : 3, East : 0>
// CHECK:             AIE.connect<South : 7, North : 0>
// CHECK:             AIE.connect<East : 0, South : 3>
// CHECK:             AIE.connect<East : 1, West : 0>
// CHECK:             AIE.connect<North : 0, South : 2>
// CHECK:             AIE.connect<North : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = AIE.shimmux(%[[VAL_1]]) {
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = AIE.tile(4, 0)
// CHECK:           %[[VAL_33:.*]] = AIE.switchbox(%[[VAL_32]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = AIE.tile(5, 0)
// CHECK:           %[[VAL_35:.*]] = AIE.switchbox(%[[VAL_34]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<North : 0, East : 1>
// CHECK:             AIE.connect<North : 1, West : 0>
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:             AIE.connect<North : 2, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = AIE.switchbox(%[[VAL_6]]) {
// CHECK:             AIE.connect<West : 0, North : 0>
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<North : 1, South : 3>
// CHECK:             AIE.connect<West : 1, South : 2>
// CHECK:             AIE.connect<South : 3, North : 1>
// CHECK:             AIE.connect<South : 7, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = AIE.tile(6, 1)
// CHECK:           %[[VAL_38:.*]] = AIE.switchbox(%[[VAL_37]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<West : 1, South : 1>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_39:.*]] = AIE.tile(6, 2)
// CHECK:           %[[VAL_40:.*]] = AIE.switchbox(%[[VAL_39]]) {
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, North : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = AIE.switchbox(%[[VAL_9]]) {
// CHECK:             AIE.connect<West : 0, DMA : 0>
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:             AIE.connect<Core : 0, North : 2>
// CHECK:             AIE.connect<DMA : 1, West : 0>
// CHECK:             AIE.connect<North : 0, Core : 1>
// CHECK:           }
// CHECK:           %[[VAL_42:.*]] = AIE.tile(3, 1)
// CHECK:           %[[VAL_43:.*]] = AIE.switchbox(%[[VAL_42]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:             AIE.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_44:.*]] = AIE.switchbox(%[[VAL_5]]) {
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<West : 1, Core : 1>
// CHECK:             AIE.connect<Core : 0, South : 0>
// CHECK:             AIE.connect<DMA : 1, South : 1>
// CHECK:             AIE.connect<East : 0, West : 0>
// CHECK:             AIE.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[VAL_45:.*]] = AIE.switchbox(%[[VAL_2]]) {
// CHECK:             AIE.connect<Core : 0, East : 0>
// CHECK:             AIE.connect<DMA : 1, South : 0>
// CHECK:             AIE.connect<East : 0, Core : 1>
// CHECK:             AIE.connect<East : 1, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_46:.*]] = AIE.switchbox(%[[VAL_4]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:             AIE.connect<Core : 0, East : 1>
// CHECK:             AIE.connect<DMA : 1, South : 0>
// CHECK:             AIE.connect<East : 0, DMA : 0>
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<East : 1, West : 1>
// CHECK:             AIE.connect<South : 1, Core : 1>
// CHECK:           }
// CHECK:           %[[VAL_47:.*]] = AIE.tile(6, 4)
// CHECK:           %[[VAL_48:.*]] = AIE.switchbox(%[[VAL_47]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<South : 0, East : 0>
// CHECK:             AIE.connect<South : 1, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_49:.*]] = AIE.switchbox(%[[VAL_8]]) {
// CHECK:             AIE.connect<West : 0, South : 2>
// CHECK:             AIE.connect<South : 3, North : 0>
// CHECK:             AIE.connect<South : 7, North : 1>
// CHECK:             AIE.connect<East : 0, South : 3>
// CHECK:           }
// CHECK:           %[[VAL_50:.*]] = AIE.shimmux(%[[VAL_8]]) {
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[VAL_51:.*]] = AIE.tile(5, 1)
// CHECK:           %[[VAL_52:.*]] = AIE.switchbox(%[[VAL_51]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<North : 1, East : 1>
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<North : 2, South : 1>
// CHECK:             AIE.connect<South : 0, West : 0>
// CHECK:             AIE.connect<East : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_53:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_54:.*]] = AIE.switchbox(%[[VAL_53]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_55:.*]] = AIE.switchbox(%[[VAL_3]]) {
// CHECK:             AIE.connect<West : 0, South : 0>
// CHECK:             AIE.connect<Core : 0, East : 0>
// CHECK:             AIE.connect<DMA : 1, East : 1>
// CHECK:             AIE.connect<North : 0, South : 1>
// CHECK:             AIE.connect<East : 0, Core : 1>
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<East : 1, North : 0>
// CHECK:             AIE.connect<East : 2, North : 1>
// CHECK:             AIE.connect<East : 3, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_56:.*]] = AIE.tile(7, 4)
// CHECK:           %[[VAL_57:.*]] = AIE.switchbox(%[[VAL_56]]) {
// CHECK:             AIE.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_58:.*]] = AIE.switchbox(%[[VAL_11]]) {
// CHECK:             AIE.connect<West : 0, Core : 1>
// CHECK:             AIE.connect<South : 0, DMA : 0>
// CHECK:             AIE.connect<Core : 0, South : 0>
// CHECK:             AIE.connect<DMA : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_59:.*]] = AIE.shimmux(%[[VAL_6]]) {
// CHECK:             AIE.connect<North : 3, DMA : 1>
// CHECK:             AIE.connect<North : 2, DMA : 0>
// CHECK:             AIE.connect<DMA : 0, North : 3>
// CHECK:             AIE.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_60:.*]] = AIE.tile(4, 1)
// CHECK:           %[[VAL_61:.*]] = AIE.switchbox(%[[VAL_60]]) {
// CHECK:             AIE.connect<North : 0, East : 0>
// CHECK:             AIE.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_62:.*]] = AIE.tile(7, 1)
// CHECK:           %[[VAL_63:.*]] = AIE.switchbox(%[[VAL_62]]) {
// CHECK:             AIE.connect<South : 0, North : 0>
// CHECK:             AIE.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_64:.*]] = AIE.tile(8, 0)
// CHECK:           %[[VAL_65:.*]] = AIE.switchbox(%[[VAL_64]]) {
// CHECK:             AIE.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_66:.*]] = AIE.tile(8, 1)
// CHECK:           %[[VAL_67:.*]] = AIE.switchbox(%[[VAL_66]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_68:.*]] = AIE.tile(8, 2)
// CHECK:           %[[VAL_69:.*]] = AIE.switchbox(%[[VAL_68]]) {
// CHECK:             AIE.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           AIE.wire(%[[VAL_13]] : North, %[[VAL_12]] : South)
// CHECK:           AIE.wire(%[[VAL_0]] : DMA, %[[VAL_13]] : DMA)
// CHECK:           AIE.wire(%[[VAL_14]] : Core, %[[VAL_15]] : Core)
// CHECK:           AIE.wire(%[[VAL_14]] : DMA, %[[VAL_15]] : DMA)
// CHECK:           AIE.wire(%[[VAL_12]] : North, %[[VAL_15]] : South)
// CHECK:           AIE.wire(%[[VAL_16]] : Core, %[[VAL_17]] : Core)
// CHECK:           AIE.wire(%[[VAL_16]] : DMA, %[[VAL_17]] : DMA)
// CHECK:           AIE.wire(%[[VAL_15]] : North, %[[VAL_17]] : South)
// CHECK:           AIE.wire(%[[VAL_12]] : East, %[[VAL_30]] : West)
// CHECK:           AIE.wire(%[[VAL_31]] : North, %[[VAL_30]] : South)
// CHECK:           AIE.wire(%[[VAL_1]] : DMA, %[[VAL_31]] : DMA)
// CHECK:           AIE.wire(%[[VAL_15]] : East, %[[VAL_43]] : West)
// CHECK:           AIE.wire(%[[VAL_42]] : Core, %[[VAL_43]] : Core)
// CHECK:           AIE.wire(%[[VAL_42]] : DMA, %[[VAL_43]] : DMA)
// CHECK:           AIE.wire(%[[VAL_30]] : North, %[[VAL_43]] : South)
// CHECK:           AIE.wire(%[[VAL_17]] : East, %[[VAL_19]] : West)
// CHECK:           AIE.wire(%[[VAL_18]] : Core, %[[VAL_19]] : Core)
// CHECK:           AIE.wire(%[[VAL_18]] : DMA, %[[VAL_19]] : DMA)
// CHECK:           AIE.wire(%[[VAL_43]] : North, %[[VAL_19]] : South)
// CHECK:           AIE.wire(%[[VAL_53]] : Core, %[[VAL_54]] : Core)
// CHECK:           AIE.wire(%[[VAL_53]] : DMA, %[[VAL_54]] : DMA)
// CHECK:           AIE.wire(%[[VAL_19]] : North, %[[VAL_54]] : South)
// CHECK:           AIE.wire(%[[VAL_2]] : Core, %[[VAL_45]] : Core)
// CHECK:           AIE.wire(%[[VAL_2]] : DMA, %[[VAL_45]] : DMA)
// CHECK:           AIE.wire(%[[VAL_54]] : North, %[[VAL_45]] : South)
// CHECK:           AIE.wire(%[[VAL_30]] : East, %[[VAL_33]] : West)
// CHECK:           AIE.wire(%[[VAL_43]] : East, %[[VAL_61]] : West)
// CHECK:           AIE.wire(%[[VAL_60]] : Core, %[[VAL_61]] : Core)
// CHECK:           AIE.wire(%[[VAL_60]] : DMA, %[[VAL_61]] : DMA)
// CHECK:           AIE.wire(%[[VAL_33]] : North, %[[VAL_61]] : South)
// CHECK:           AIE.wire(%[[VAL_19]] : East, %[[VAL_21]] : West)
// CHECK:           AIE.wire(%[[VAL_20]] : Core, %[[VAL_21]] : Core)
// CHECK:           AIE.wire(%[[VAL_20]] : DMA, %[[VAL_21]] : DMA)
// CHECK:           AIE.wire(%[[VAL_61]] : North, %[[VAL_21]] : South)
// CHECK:           AIE.wire(%[[VAL_54]] : East, %[[VAL_55]] : West)
// CHECK:           AIE.wire(%[[VAL_3]] : Core, %[[VAL_55]] : Core)
// CHECK:           AIE.wire(%[[VAL_3]] : DMA, %[[VAL_55]] : DMA)
// CHECK:           AIE.wire(%[[VAL_21]] : North, %[[VAL_55]] : South)
// CHECK:           AIE.wire(%[[VAL_45]] : East, %[[VAL_46]] : West)
// CHECK:           AIE.wire(%[[VAL_4]] : Core, %[[VAL_46]] : Core)
// CHECK:           AIE.wire(%[[VAL_4]] : DMA, %[[VAL_46]] : DMA)
// CHECK:           AIE.wire(%[[VAL_55]] : North, %[[VAL_46]] : South)
// CHECK:           AIE.wire(%[[VAL_33]] : East, %[[VAL_35]] : West)
// CHECK:           AIE.wire(%[[VAL_61]] : East, %[[VAL_52]] : West)
// CHECK:           AIE.wire(%[[VAL_51]] : Core, %[[VAL_52]] : Core)
// CHECK:           AIE.wire(%[[VAL_51]] : DMA, %[[VAL_52]] : DMA)
// CHECK:           AIE.wire(%[[VAL_35]] : North, %[[VAL_52]] : South)
// CHECK:           AIE.wire(%[[VAL_21]] : East, %[[VAL_23]] : West)
// CHECK:           AIE.wire(%[[VAL_22]] : Core, %[[VAL_23]] : Core)
// CHECK:           AIE.wire(%[[VAL_22]] : DMA, %[[VAL_23]] : DMA)
// CHECK:           AIE.wire(%[[VAL_52]] : North, %[[VAL_23]] : South)
// CHECK:           AIE.wire(%[[VAL_55]] : East, %[[VAL_25]] : West)
// CHECK:           AIE.wire(%[[VAL_24]] : Core, %[[VAL_25]] : Core)
// CHECK:           AIE.wire(%[[VAL_24]] : DMA, %[[VAL_25]] : DMA)
// CHECK:           AIE.wire(%[[VAL_23]] : North, %[[VAL_25]] : South)
// CHECK:           AIE.wire(%[[VAL_46]] : East, %[[VAL_44]] : West)
// CHECK:           AIE.wire(%[[VAL_5]] : Core, %[[VAL_44]] : Core)
// CHECK:           AIE.wire(%[[VAL_5]] : DMA, %[[VAL_44]] : DMA)
// CHECK:           AIE.wire(%[[VAL_25]] : North, %[[VAL_44]] : South)
// CHECK:           AIE.wire(%[[VAL_35]] : East, %[[VAL_36]] : West)
// CHECK:           AIE.wire(%[[VAL_59]] : North, %[[VAL_36]] : South)
// CHECK:           AIE.wire(%[[VAL_6]] : DMA, %[[VAL_59]] : DMA)
// CHECK:           AIE.wire(%[[VAL_52]] : East, %[[VAL_38]] : West)
// CHECK:           AIE.wire(%[[VAL_37]] : Core, %[[VAL_38]] : Core)
// CHECK:           AIE.wire(%[[VAL_37]] : DMA, %[[VAL_38]] : DMA)
// CHECK:           AIE.wire(%[[VAL_36]] : North, %[[VAL_38]] : South)
// CHECK:           AIE.wire(%[[VAL_23]] : East, %[[VAL_40]] : West)
// CHECK:           AIE.wire(%[[VAL_39]] : Core, %[[VAL_40]] : Core)
// CHECK:           AIE.wire(%[[VAL_39]] : DMA, %[[VAL_40]] : DMA)
// CHECK:           AIE.wire(%[[VAL_38]] : North, %[[VAL_40]] : South)
// CHECK:           AIE.wire(%[[VAL_25]] : East, %[[VAL_26]] : West)
// CHECK:           AIE.wire(%[[VAL_7]] : Core, %[[VAL_26]] : Core)
// CHECK:           AIE.wire(%[[VAL_7]] : DMA, %[[VAL_26]] : DMA)
// CHECK:           AIE.wire(%[[VAL_40]] : North, %[[VAL_26]] : South)
// CHECK:           AIE.wire(%[[VAL_44]] : East, %[[VAL_48]] : West)
// CHECK:           AIE.wire(%[[VAL_47]] : Core, %[[VAL_48]] : Core)
// CHECK:           AIE.wire(%[[VAL_47]] : DMA, %[[VAL_48]] : DMA)
// CHECK:           AIE.wire(%[[VAL_26]] : North, %[[VAL_48]] : South)
// CHECK:           AIE.wire(%[[VAL_36]] : East, %[[VAL_49]] : West)
// CHECK:           AIE.wire(%[[VAL_50]] : North, %[[VAL_49]] : South)
// CHECK:           AIE.wire(%[[VAL_8]] : DMA, %[[VAL_50]] : DMA)
// CHECK:           AIE.wire(%[[VAL_38]] : East, %[[VAL_63]] : West)
// CHECK:           AIE.wire(%[[VAL_62]] : Core, %[[VAL_63]] : Core)
// CHECK:           AIE.wire(%[[VAL_62]] : DMA, %[[VAL_63]] : DMA)
// CHECK:           AIE.wire(%[[VAL_49]] : North, %[[VAL_63]] : South)
// CHECK:           AIE.wire(%[[VAL_40]] : East, %[[VAL_41]] : West)
// CHECK:           AIE.wire(%[[VAL_9]] : Core, %[[VAL_41]] : Core)
// CHECK:           AIE.wire(%[[VAL_9]] : DMA, %[[VAL_41]] : DMA)
// CHECK:           AIE.wire(%[[VAL_63]] : North, %[[VAL_41]] : South)
// CHECK:           AIE.wire(%[[VAL_26]] : East, %[[VAL_28]] : West)
// CHECK:           AIE.wire(%[[VAL_27]] : Core, %[[VAL_28]] : Core)
// CHECK:           AIE.wire(%[[VAL_27]] : DMA, %[[VAL_28]] : DMA)
// CHECK:           AIE.wire(%[[VAL_41]] : North, %[[VAL_28]] : South)
// CHECK:           AIE.wire(%[[VAL_48]] : East, %[[VAL_57]] : West)
// CHECK:           AIE.wire(%[[VAL_56]] : Core, %[[VAL_57]] : Core)
// CHECK:           AIE.wire(%[[VAL_56]] : DMA, %[[VAL_57]] : DMA)
// CHECK:           AIE.wire(%[[VAL_28]] : North, %[[VAL_57]] : South)
// CHECK:           AIE.wire(%[[VAL_49]] : East, %[[VAL_65]] : West)
// CHECK:           AIE.wire(%[[VAL_63]] : East, %[[VAL_67]] : West)
// CHECK:           AIE.wire(%[[VAL_66]] : Core, %[[VAL_67]] : Core)
// CHECK:           AIE.wire(%[[VAL_66]] : DMA, %[[VAL_67]] : DMA)
// CHECK:           AIE.wire(%[[VAL_65]] : North, %[[VAL_67]] : South)
// CHECK:           AIE.wire(%[[VAL_41]] : East, %[[VAL_69]] : West)
// CHECK:           AIE.wire(%[[VAL_68]] : Core, %[[VAL_69]] : Core)
// CHECK:           AIE.wire(%[[VAL_68]] : DMA, %[[VAL_69]] : DMA)
// CHECK:           AIE.wire(%[[VAL_67]] : North, %[[VAL_69]] : South)
// CHECK:           AIE.wire(%[[VAL_28]] : East, %[[VAL_29]] : West)
// CHECK:           AIE.wire(%[[VAL_10]] : Core, %[[VAL_29]] : Core)
// CHECK:           AIE.wire(%[[VAL_10]] : DMA, %[[VAL_29]] : DMA)
// CHECK:           AIE.wire(%[[VAL_69]] : North, %[[VAL_29]] : South)
// CHECK:           AIE.wire(%[[VAL_57]] : East, %[[VAL_58]] : West)
// CHECK:           AIE.wire(%[[VAL_11]] : Core, %[[VAL_58]] : Core)
// CHECK:           AIE.wire(%[[VAL_11]] : DMA, %[[VAL_58]] : DMA)
// CHECK:           AIE.wire(%[[VAL_29]] : North, %[[VAL_58]] : South)
// CHECK:         }









module {
    AIE.device(xcvc1902) {
        %t20 = AIE.tile(2, 0)
        %t30 = AIE.tile(3, 0)
        %t34 = AIE.tile(3, 4)
        %t43 = AIE.tile(4, 3)
        %t44 = AIE.tile(4, 4)
        %t54 = AIE.tile(5, 4)
        %t60 = AIE.tile(6, 0)
        %t63 = AIE.tile(6, 3)
        %t70 = AIE.tile(7, 0)
        %t72 = AIE.tile(7, 2)
        %t83 = AIE.tile(8, 3)
        %t84 = AIE.tile(8, 4)

        AIE.flow(%t20, DMA : 0, %t63, DMA : 0)
        AIE.flow(%t20, DMA : 1, %t83, DMA : 0)
        AIE.flow(%t30, DMA : 0, %t72, DMA : 0)
        AIE.flow(%t30, DMA : 1, %t54, DMA : 0)

        AIE.flow(%t34, Core : 0, %t63, Core : 1)
        AIE.flow(%t34, DMA : 1, %t70, DMA : 0)
        AIE.flow(%t43, Core : 0, %t84, Core : 1)
        AIE.flow(%t43, DMA : 1, %t60, DMA : 1)

        AIE.flow(%t44, Core : 0, %t54, Core : 1)
        AIE.flow(%t44, DMA : 1, %t60, DMA : 0)
        AIE.flow(%t54, Core : 0, %t43, Core : 1)
        AIE.flow(%t54, DMA : 1, %t30, DMA : 1)

        AIE.flow(%t60, DMA : 0, %t44, DMA : 0)
        AIE.flow(%t60, DMA : 1, %t43, DMA : 0)
        AIE.flow(%t63, Core : 0, %t34, Core : 1)
        AIE.flow(%t63, DMA : 1, %t20, DMA : 1)

        AIE.flow(%t70, DMA : 0, %t34, DMA : 0)
        AIE.flow(%t70, DMA : 1, %t84, DMA : 0)
        AIE.flow(%t72, Core : 0, %t83, Core : 1)
        AIE.flow(%t72, DMA : 1, %t30, DMA : 0)

        AIE.flow(%t83, Core : 0, %t44, Core : 1)
        AIE.flow(%t83, DMA : 1, %t20, DMA : 0)
        AIE.flow(%t84, Core : 0, %t72, Core : 1)
        AIE.flow(%t84, DMA : 1, %t70, DMA : 1)
    }
}
