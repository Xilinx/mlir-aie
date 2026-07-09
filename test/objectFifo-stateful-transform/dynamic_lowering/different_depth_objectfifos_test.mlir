//===- different_depth_objectfifos_test.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
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
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "input_fifo_cons_buff_2"} : memref<10xi32>
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_4]], 0) {init = 3 : i32, sym_name = "input_fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_4]], 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_3]], 0) {init = 0 : i32, sym_name = "input_fifo_prod_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_4]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_4]]) : memref<2xi32>
// CHECK:           %[[VAL_19:.*]] = aie.buffer(%[[VAL_4]]) : memref<2xi32>
// CHECK:           %[[VAL_20:.*]] = aie.core(%[[VAL_4]]) {
// CHECK:             %[[VAL_21:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_22:.*]] = arith.constant 0 : index
// CHECK:             memref.store %[[VAL_21]], %[[VAL_19]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_23:.*]] = arith.constant 1 : index
// CHECK:             memref.store %[[VAL_21]], %[[VAL_19]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:             %[[VAL_24:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_25:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_26:.*]] = arith.constant 2 : i32
// CHECK:             memref.store %[[VAL_24]], %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:             %[[VAL_27:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_28:.*]] = arith.constant 3 : i32
// CHECK:             memref.store %[[VAL_24]], %[[VAL_18]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:             %[[VAL_29:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_30:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_31:.*]] = arith.constant 9 : index
// CHECK:             %[[VAL_32:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_33:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_34:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_35:.*]] = arith.subi %[[VAL_33]], %[[VAL_32]] : i32
// CHECK:             %[[VAL_36:.*]] = arith.maxsi %[[VAL_35]], %[[VAL_34]] : i32
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_36]])
// CHECK:             %[[VAL_37:.*]] = arith.addi %[[VAL_32]], %[[VAL_36]] : i32
// CHECK:             memref.store %[[VAL_37]], %[[VAL_19]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_38:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:             %[[VAL_39:.*]] = arith.index_cast %[[VAL_38]] : i32 to index
// CHECK:             %[[VAL_40:.*]] = scf.index_switch %[[VAL_39]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_8]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[VAL_41:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:             %[[VAL_42:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_43:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_44:.*]] = arith.subi %[[VAL_42]], %[[VAL_41]] : i32
// CHECK:             %[[VAL_45:.*]] = arith.maxsi %[[VAL_44]], %[[VAL_43]] : i32
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_45]])
// CHECK:             %[[VAL_46:.*]] = arith.addi %[[VAL_41]], %[[VAL_45]] : i32
// CHECK:             memref.store %[[VAL_46]], %[[VAL_19]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:             %[[VAL_47:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:             %[[VAL_48:.*]] = arith.index_cast %[[VAL_47]] : i32 to index
// CHECK:             %[[VAL_49:.*]] = scf.index_switch %[[VAL_48]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 2 {
// CHECK:               scf.yield %[[VAL_13]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:             }
// CHECK:             func.call @add_10_i32(%[[VAL_49]], %[[VAL_49]], %[[VAL_40]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:             %[[VAL_50:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_50]])
// CHECK:             %[[VAL_51:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_52:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_53:.*]] = arith.subi %[[VAL_51]], %[[VAL_52]] : i32
// CHECK:             memref.store %[[VAL_53]], %[[VAL_19]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_54:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:             %[[VAL_55:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_56:.*]] = arith.addi %[[VAL_54]], %[[VAL_55]] : i32
// CHECK:             %[[VAL_57:.*]] = arith.cmpi sge, %[[VAL_56]], %[[VAL_26]] : i32
// CHECK:             %[[VAL_58:.*]] = arith.subi %[[VAL_56]], %[[VAL_26]] : i32
// CHECK:             %[[VAL_59:.*]] = arith.select %[[VAL_57]], %[[VAL_58]], %[[VAL_56]] : i32
// CHECK:             memref.store %[[VAL_59]], %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:             scf.for %[[VAL_60:.*]] = %[[VAL_29]] to %[[VAL_31]] step %[[VAL_30]] {
// CHECK:               %[[VAL_61:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:               %[[VAL_62:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_63:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_64:.*]] = arith.subi %[[VAL_62]], %[[VAL_61]] : i32
// CHECK:               %[[VAL_65:.*]] = arith.maxsi %[[VAL_64]], %[[VAL_63]] : i32
// CHECK:               aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_65]])
// CHECK:               %[[VAL_66:.*]] = arith.addi %[[VAL_61]], %[[VAL_65]] : i32
// CHECK:               memref.store %[[VAL_66]], %[[VAL_19]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:               %[[VAL_67:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:               %[[VAL_68:.*]] = arith.index_cast %[[VAL_67]] : i32 to index
// CHECK:               %[[VAL_69:.*]] = scf.index_switch %[[VAL_68]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_8]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:               }
// CHECK:               %[[VAL_70:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:               %[[VAL_71:.*]] = arith.constant 2 : i32
// CHECK:               %[[VAL_72:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_73:.*]] = arith.subi %[[VAL_71]], %[[VAL_70]] : i32
// CHECK:               %[[VAL_74:.*]] = arith.maxsi %[[VAL_73]], %[[VAL_72]] : i32
// CHECK:               aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_74]])
// CHECK:               %[[VAL_75:.*]] = arith.addi %[[VAL_70]], %[[VAL_74]] : i32
// CHECK:               memref.store %[[VAL_75]], %[[VAL_19]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:               %[[VAL_76:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:               %[[VAL_77:.*]] = arith.index_cast %[[VAL_76]] : i32 to index
// CHECK:               %[[VAL_78:.*]] = scf.index_switch %[[VAL_77]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 2 {
// CHECK:                 scf.yield %[[VAL_13]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:               }
// CHECK:               %[[VAL_79:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:               %[[VAL_80:.*]] = arith.index_cast %[[VAL_79]] : i32 to index
// CHECK:               %[[VAL_81:.*]] = scf.index_switch %[[VAL_80]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_13]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 2 {
// CHECK:                 scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:               }
// CHECK:               func.call @add_10_i32(%[[VAL_78]], %[[VAL_81]], %[[VAL_69]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               %[[VAL_82:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_14]], Release, %[[VAL_82]])
// CHECK:               %[[VAL_83:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:               %[[VAL_84:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_85:.*]] = arith.subi %[[VAL_83]], %[[VAL_84]] : i32
// CHECK:               memref.store %[[VAL_85]], %[[VAL_19]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:               %[[VAL_86:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:               %[[VAL_87:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_88:.*]] = arith.addi %[[VAL_86]], %[[VAL_87]] : i32
// CHECK:               %[[VAL_89:.*]] = arith.cmpi sge, %[[VAL_88]], %[[VAL_28]] : i32
// CHECK:               %[[VAL_90:.*]] = arith.subi %[[VAL_88]], %[[VAL_28]] : i32
// CHECK:               %[[VAL_91:.*]] = arith.select %[[VAL_89]], %[[VAL_90]], %[[VAL_88]] : i32
// CHECK:               memref.store %[[VAL_91]], %[[VAL_18]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:               %[[VAL_92:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_10]], Release, %[[VAL_92]])
// CHECK:               %[[VAL_93:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:               %[[VAL_94:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_95:.*]] = arith.subi %[[VAL_93]], %[[VAL_94]] : i32
// CHECK:               memref.store %[[VAL_95]], %[[VAL_19]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:               %[[VAL_96:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:               %[[VAL_97:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_98:.*]] = arith.addi %[[VAL_96]], %[[VAL_97]] : i32
// CHECK:               %[[VAL_99:.*]] = arith.cmpi sge, %[[VAL_98]], %[[VAL_26]] : i32
// CHECK:               %[[VAL_100:.*]] = arith.subi %[[VAL_98]], %[[VAL_26]] : i32
// CHECK:               %[[VAL_101:.*]] = arith.select %[[VAL_99]], %[[VAL_100]], %[[VAL_98]] : i32
// CHECK:               memref.store %[[VAL_101]], %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:             }
// CHECK:             %[[VAL_102:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_103:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_104:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_105:.*]] = arith.subi %[[VAL_103]], %[[VAL_102]] : i32
// CHECK:             %[[VAL_106:.*]] = arith.maxsi %[[VAL_105]], %[[VAL_104]] : i32
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_106]])
// CHECK:             %[[VAL_107:.*]] = arith.addi %[[VAL_102]], %[[VAL_106]] : i32
// CHECK:             memref.store %[[VAL_107]], %[[VAL_19]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_108:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:             %[[VAL_109:.*]] = arith.index_cast %[[VAL_108]] : i32 to index
// CHECK:             %[[VAL_110:.*]] = scf.index_switch %[[VAL_109]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_8]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[VAL_111:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:             %[[VAL_112:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_113:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_114:.*]] = arith.subi %[[VAL_112]], %[[VAL_111]] : i32
// CHECK:             %[[VAL_115:.*]] = arith.maxsi %[[VAL_114]], %[[VAL_113]] : i32
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_115]])
// CHECK:             %[[VAL_116:.*]] = arith.addi %[[VAL_111]], %[[VAL_115]] : i32
// CHECK:             memref.store %[[VAL_116]], %[[VAL_19]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:             %[[VAL_117:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:             %[[VAL_118:.*]] = arith.index_cast %[[VAL_117]] : i32 to index
// CHECK:             %[[VAL_119:.*]] = scf.index_switch %[[VAL_118]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 2 {
// CHECK:               scf.yield %[[VAL_13]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[VAL_120:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:             %[[VAL_121:.*]] = arith.index_cast %[[VAL_120]] : i32 to index
// CHECK:             %[[VAL_122:.*]] = scf.index_switch %[[VAL_121]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_13]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 2 {
// CHECK:               scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:             }
// CHECK:             func.call @add_10_i32(%[[VAL_119]], %[[VAL_122]], %[[VAL_110]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:             %[[VAL_123:.*]] = arith.constant 2 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_123]])
// CHECK:             %[[VAL_124:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:             %[[VAL_125:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_126:.*]] = arith.subi %[[VAL_124]], %[[VAL_125]] : i32
// CHECK:             memref.store %[[VAL_126]], %[[VAL_19]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:             %[[VAL_127:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:             %[[VAL_128:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_129:.*]] = arith.addi %[[VAL_127]], %[[VAL_128]] : i32
// CHECK:             %[[VAL_130:.*]] = arith.cmpi sge, %[[VAL_129]], %[[VAL_28]] : i32
// CHECK:             %[[VAL_131:.*]] = arith.subi %[[VAL_129]], %[[VAL_28]] : i32
// CHECK:             %[[VAL_132:.*]] = arith.select %[[VAL_130]], %[[VAL_131]], %[[VAL_129]] : i32
// CHECK:             memref.store %[[VAL_132]], %[[VAL_18]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:             %[[VAL_133:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_133]])
// CHECK:             %[[VAL_134:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_135:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_136:.*]] = arith.subi %[[VAL_134]], %[[VAL_135]] : i32
// CHECK:             memref.store %[[VAL_136]], %[[VAL_19]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_137:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:             %[[VAL_138:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_139:.*]] = arith.addi %[[VAL_137]], %[[VAL_138]] : i32
// CHECK:             %[[VAL_140:.*]] = arith.cmpi sge, %[[VAL_139]], %[[VAL_26]] : i32
// CHECK:             %[[VAL_141:.*]] = arith.subi %[[VAL_139]], %[[VAL_26]] : i32
// CHECK:             %[[VAL_142:.*]] = arith.select %[[VAL_140]], %[[VAL_141]], %[[VAL_139]] : i32
// CHECK:             memref.store %[[VAL_142]], %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @input_fifo_shim_alloc(%[[VAL_3]], MM2S, 0)
// CHECK:           %[[VAL_143:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_144:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_145:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_145]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_146:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_146]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_147:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_147]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_148:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_148]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_149:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_149]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_150:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_150]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %[[VAL_151:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb7)
// CHECK:           ^bb5:
// CHECK:             %[[VAL_152:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_152]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_153:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_153]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             %[[VAL_154:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_154]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_155:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_155]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb7:
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
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, [2, 3]) : !aie.objectfifo<memref<10xi32>>
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
