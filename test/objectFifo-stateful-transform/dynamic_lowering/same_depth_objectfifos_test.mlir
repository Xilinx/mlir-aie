//===- same_depth_objectfifos_test.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform=dynamic-objFifos %s | FileCheck %s

// CHECK-LABEL: module {
// CHECK:         aie.device(npu1_1col) {
// CHECK:           func.func @add_10_i32(%[[VAL_0:.*]]: memref<10xi32>, %[[VAL_1:.*]]: memref<10xi32>, %[[VAL_2:.*]]: memref<10xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_3:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_4:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_3]], 2) {init = 0 : i32, sym_name = "output_fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_3]], 3) {init = 0 : i32, sym_name = "output_fifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "output_fifo_buff_0"} : memref<10xi32>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "output_fifo_buff_1"} : memref<10xi32>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_4]], 2) {init = 2 : i32, sym_name = "output_fifo_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_4]], 3) {init = 0 : i32, sym_name = "output_fifo_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32>
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "input_fifo_cons_buff_1"} : memref<10xi32>
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_4]], 0) {init = 2 : i32, sym_name = "input_fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_4]], 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_3]], 0) {init = 0 : i32, sym_name = "input_fifo_prod_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_4]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_4]]) : memref<2xi32>
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_4]]) : memref<2xi32>
// CHECK:           %[[VAL_19:.*]] = aie.core(%[[VAL_4]]) {
// CHECK:             %[[VAL_20:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_21:.*]] = arith.constant 0 : index
// CHECK:             memref.store %[[VAL_20]], %[[VAL_18]]{{\[}}%[[VAL_21]]] : memref<2xi32>
// CHECK:             %[[VAL_22:.*]] = arith.constant 1 : index
// CHECK:             memref.store %[[VAL_20]], %[[VAL_18]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_23:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_24:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_25:.*]] = arith.constant 2 : i32
// CHECK:             memref.store %[[VAL_23]], %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:             %[[VAL_26:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_27:.*]] = arith.constant 2 : i32
// CHECK:             memref.store %[[VAL_23]], %[[VAL_17]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:             %[[VAL_28:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_29:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_30:.*]] = arith.constant 9 : index
// CHECK:             %[[VAL_31:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_21]]] : memref<2xi32>
// CHECK:             %[[VAL_32:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_33:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_34:.*]] = arith.subi %[[VAL_32]], %[[VAL_31]] : i32
// CHECK:             %[[VAL_35:.*]] = arith.maxsi %[[VAL_34]], %[[VAL_33]] : i32
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             %[[VAL_36:.*]] = arith.addi %[[VAL_31]], %[[VAL_35]] : i32
// CHECK:             memref.store %[[VAL_36]], %[[VAL_18]]{{\[}}%[[VAL_21]]] : memref<2xi32>
// CHECK:             %[[VAL_37:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:             %[[VAL_38:.*]] = arith.index_cast %[[VAL_37]] : i32 to index
// CHECK:             %[[VAL_39:.*]] = scf.index_switch %[[VAL_38]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_8]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[VAL_40:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_41:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_42:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_43:.*]] = arith.subi %[[VAL_41]], %[[VAL_40]] : i32
// CHECK:             %[[VAL_44:.*]] = arith.maxsi %[[VAL_43]], %[[VAL_42]] : i32
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_44]])
// CHECK:             %[[VAL_45:.*]] = arith.addi %[[VAL_40]], %[[VAL_44]] : i32
// CHECK:             memref.store %[[VAL_45]], %[[VAL_18]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_46:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:             %[[VAL_47:.*]] = arith.index_cast %[[VAL_46]] : i32 to index
// CHECK:             %[[VAL_48:.*]] = scf.index_switch %[[VAL_47]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:             }
// CHECK:             func.call @add_10_i32(%[[VAL_48]], %[[VAL_48]], %[[VAL_39]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:             %[[VAL_49:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_49]])
// CHECK:             %[[VAL_50:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_21]]] : memref<2xi32>
// CHECK:             %[[VAL_51:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_52:.*]] = arith.subi %[[VAL_50]], %[[VAL_51]] : i32
// CHECK:             memref.store %[[VAL_52]], %[[VAL_18]]{{\[}}%[[VAL_21]]] : memref<2xi32>
// CHECK:             %[[VAL_53:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:             %[[VAL_54:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_55:.*]] = arith.addi %[[VAL_53]], %[[VAL_54]] : i32
// CHECK:             %[[VAL_56:.*]] = arith.cmpi sge, %[[VAL_55]], %[[VAL_25]] : i32
// CHECK:             %[[VAL_57:.*]] = arith.subi %[[VAL_55]], %[[VAL_25]] : i32
// CHECK:             %[[VAL_58:.*]] = arith.select %[[VAL_56]], %[[VAL_57]], %[[VAL_55]] : i32
// CHECK:             memref.store %[[VAL_58]], %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:             scf.for %[[VAL_59:.*]] = %[[VAL_28]] to %[[VAL_30]] step %[[VAL_29]] {
// CHECK:               %[[VAL_60:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_21]]] : memref<2xi32>
// CHECK:               %[[VAL_61:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_62:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_63:.*]] = arith.subi %[[VAL_61]], %[[VAL_60]] : i32
// CHECK:               %[[VAL_64:.*]] = arith.maxsi %[[VAL_63]], %[[VAL_62]] : i32
// CHECK:               aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_64]])
// CHECK:               %[[VAL_65:.*]] = arith.addi %[[VAL_60]], %[[VAL_64]] : i32
// CHECK:               memref.store %[[VAL_65]], %[[VAL_18]]{{\[}}%[[VAL_21]]] : memref<2xi32>
// CHECK:               %[[VAL_66:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:               %[[VAL_67:.*]] = arith.index_cast %[[VAL_66]] : i32 to index
// CHECK:               %[[VAL_68:.*]] = scf.index_switch %[[VAL_67]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_8]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:               }
// CHECK:               %[[VAL_69:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:               %[[VAL_70:.*]] = arith.constant 2 : i32
// CHECK:               %[[VAL_71:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_72:.*]] = arith.subi %[[VAL_70]], %[[VAL_69]] : i32
// CHECK:               %[[VAL_73:.*]] = arith.maxsi %[[VAL_72]], %[[VAL_71]] : i32
// CHECK:               aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_73]])
// CHECK:               %[[VAL_74:.*]] = arith.addi %[[VAL_69]], %[[VAL_73]] : i32
// CHECK:               memref.store %[[VAL_74]], %[[VAL_18]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:               %[[VAL_75:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:               %[[VAL_76:.*]] = arith.index_cast %[[VAL_75]] : i32 to index
// CHECK:               %[[VAL_77:.*]] = scf.index_switch %[[VAL_76]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:               }
// CHECK:               %[[VAL_78:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:               %[[VAL_79:.*]] = arith.index_cast %[[VAL_78]] : i32 to index
// CHECK:               %[[VAL_80:.*]] = scf.index_switch %[[VAL_79]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:               }
// CHECK:               func.call @add_10_i32(%[[VAL_77]], %[[VAL_80]], %[[VAL_68]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               %[[VAL_81:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_13]], Release, %[[VAL_81]])
// CHECK:               %[[VAL_82:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:               %[[VAL_83:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_84:.*]] = arith.subi %[[VAL_82]], %[[VAL_83]] : i32
// CHECK:               memref.store %[[VAL_84]], %[[VAL_18]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:               %[[VAL_85:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:               %[[VAL_86:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_87:.*]] = arith.addi %[[VAL_85]], %[[VAL_86]] : i32
// CHECK:               %[[VAL_88:.*]] = arith.cmpi sge, %[[VAL_87]], %[[VAL_27]] : i32
// CHECK:               %[[VAL_89:.*]] = arith.subi %[[VAL_87]], %[[VAL_27]] : i32
// CHECK:               %[[VAL_90:.*]] = arith.select %[[VAL_88]], %[[VAL_89]], %[[VAL_87]] : i32
// CHECK:               memref.store %[[VAL_90]], %[[VAL_17]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:               %[[VAL_91:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_10]], Release, %[[VAL_91]])
// CHECK:               %[[VAL_92:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_21]]] : memref<2xi32>
// CHECK:               %[[VAL_93:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_94:.*]] = arith.subi %[[VAL_92]], %[[VAL_93]] : i32
// CHECK:               memref.store %[[VAL_94]], %[[VAL_18]]{{\[}}%[[VAL_21]]] : memref<2xi32>
// CHECK:               %[[VAL_95:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:               %[[VAL_96:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_97:.*]] = arith.addi %[[VAL_95]], %[[VAL_96]] : i32
// CHECK:               %[[VAL_98:.*]] = arith.cmpi sge, %[[VAL_97]], %[[VAL_25]] : i32
// CHECK:               %[[VAL_99:.*]] = arith.subi %[[VAL_97]], %[[VAL_25]] : i32
// CHECK:               %[[VAL_100:.*]] = arith.select %[[VAL_98]], %[[VAL_99]], %[[VAL_97]] : i32
// CHECK:               memref.store %[[VAL_100]], %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:             }
// CHECK:             %[[VAL_101:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_21]]] : memref<2xi32>
// CHECK:             %[[VAL_102:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_103:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_104:.*]] = arith.subi %[[VAL_102]], %[[VAL_101]] : i32
// CHECK:             %[[VAL_105:.*]] = arith.maxsi %[[VAL_104]], %[[VAL_103]] : i32
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_105]])
// CHECK:             %[[VAL_106:.*]] = arith.addi %[[VAL_101]], %[[VAL_105]] : i32
// CHECK:             memref.store %[[VAL_106]], %[[VAL_18]]{{\[}}%[[VAL_21]]] : memref<2xi32>
// CHECK:             %[[VAL_107:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:             %[[VAL_108:.*]] = arith.index_cast %[[VAL_107]] : i32 to index
// CHECK:             %[[VAL_109:.*]] = scf.index_switch %[[VAL_108]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_8]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[VAL_110:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_111:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_112:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_113:.*]] = arith.subi %[[VAL_111]], %[[VAL_110]] : i32
// CHECK:             %[[VAL_114:.*]] = arith.maxsi %[[VAL_113]], %[[VAL_112]] : i32
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_114]])
// CHECK:             %[[VAL_115:.*]] = arith.addi %[[VAL_110]], %[[VAL_114]] : i32
// CHECK:             memref.store %[[VAL_115]], %[[VAL_18]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_116:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:             %[[VAL_117:.*]] = arith.index_cast %[[VAL_116]] : i32 to index
// CHECK:             %[[VAL_118:.*]] = scf.index_switch %[[VAL_117]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[VAL_119:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:             %[[VAL_120:.*]] = arith.index_cast %[[VAL_119]] : i32 to index
// CHECK:             %[[VAL_121:.*]] = scf.index_switch %[[VAL_120]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:             }
// CHECK:             func.call @add_10_i32(%[[VAL_118]], %[[VAL_121]], %[[VAL_109]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:             %[[VAL_122:.*]] = arith.constant 2 : i32
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[VAL_122]])
// CHECK:             %[[VAL_123:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_124:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_125:.*]] = arith.subi %[[VAL_123]], %[[VAL_124]] : i32
// CHECK:             memref.store %[[VAL_125]], %[[VAL_18]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_126:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:             %[[VAL_127:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_128:.*]] = arith.addi %[[VAL_126]], %[[VAL_127]] : i32
// CHECK:             %[[VAL_129:.*]] = arith.cmpi sge, %[[VAL_128]], %[[VAL_27]] : i32
// CHECK:             %[[VAL_130:.*]] = arith.subi %[[VAL_128]], %[[VAL_27]] : i32
// CHECK:             %[[VAL_131:.*]] = arith.select %[[VAL_129]], %[[VAL_130]], %[[VAL_128]] : i32
// CHECK:             memref.store %[[VAL_131]], %[[VAL_17]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:             %[[VAL_132:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_132]])
// CHECK:             %[[VAL_133:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_21]]] : memref<2xi32>
// CHECK:             %[[VAL_134:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_135:.*]] = arith.subi %[[VAL_133]], %[[VAL_134]] : i32
// CHECK:             memref.store %[[VAL_135]], %[[VAL_18]]{{\[}}%[[VAL_21]]] : memref<2xi32>
// CHECK:             %[[VAL_136:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:             %[[VAL_137:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_138:.*]] = arith.addi %[[VAL_136]], %[[VAL_137]] : i32
// CHECK:             %[[VAL_139:.*]] = arith.cmpi sge, %[[VAL_138]], %[[VAL_25]] : i32
// CHECK:             %[[VAL_140:.*]] = arith.subi %[[VAL_138]], %[[VAL_25]] : i32
// CHECK:             %[[VAL_141:.*]] = arith.select %[[VAL_139]], %[[VAL_140]], %[[VAL_138]] : i32
// CHECK:             memref.store %[[VAL_141]], %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @input_fifo_shim_alloc(%[[VAL_3]], MM2S, 0)
// CHECK:           %[[VAL_142:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_143:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_144:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_144]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_145:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_145]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_146:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_146]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_147:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_147]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_148:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             %[[VAL_149:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_149]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_150:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_150]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             %[[VAL_151:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_151]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_152:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_152]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @output_fifo_shim_alloc(%[[VAL_3]], S2MM, 0)
// CHECK:         }
// CHECK:       }

module {
  aie.device(npu1_1col) {
    func.func @add_10_i32(%line_in1: memref<10xi32>, %line_in2: memref<10xi32>, %line_out: memref<10xi32>) -> () {
        return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, [2, 2]) : !aie.objectfifo<memref<10xi32>>
    aie.objectfifo @output_fifo(%tile_0_2, {%tile_0_0}, [2, 2]) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 9 : index

      %0 = aie.objectfifo.acquire @output_fifo(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
      %2 = aie.objectfifo.acquire @input_fifo(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
      %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
      func.call @add_10_i32(%3, %3, %1) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
      aie.objectfifo.release @output_fifo(Produce, 1)

      scf.for %arg0 = %c0 to %c8 step %c1 {
        %4 = aie.objectfifo.acquire @output_fifo(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        %6 = aie.objectfifo.acquire @input_fifo(Consume, 2) : !aie.objectfifosubview<memref<10xi32>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        %8 = aie.objectfifo.subview.access %6[1] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        func.call @add_10_i32(%7, %8, %5) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo(Consume, 1)
        aie.objectfifo.release @output_fifo(Produce, 1)
      }

      %9 = aie.objectfifo.acquire @output_fifo(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
      %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
      %11 = aie.objectfifo.acquire @input_fifo(Consume, 2) : !aie.objectfifosubview<memref<10xi32>>
      %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
      %13 = aie.objectfifo.subview.access %11[1] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
      func.call @add_10_i32(%12, %13, %10) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
      aie.objectfifo.release @input_fifo(Consume, 2)
      aie.objectfifo.release @output_fifo(Produce, 1)

      aie.end
    }
  }
}
