//===- pass_flag_test.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform=dynamic-objFifos %s | FileCheck %s

// CHECK-LABEL: module {
// CHECK:         aie.device(npu1_1col) {
// CHECK:           func.func @passthrough_10_i32(%[[VAL_0:.*]]: memref<10xi32>, %[[VAL_1:.*]]: memref<10xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_2:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_3:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_4:.*]] = aie.tile(0, 4)
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_2]], 6) {init = 0 : i32, sym_name = "output_fifo2_cons_prod_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_2]], 7) {init = 0 : i32, sym_name = "output_fifo2_cons_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "output_fifo2_buff_0"} : memref<10xi32>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "output_fifo2_buff_1"} : memref<10xi32>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_4]], 2) {init = 2 : i32, sym_name = "output_fifo2_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_4]], 3) {init = 0 : i32, sym_name = "output_fifo2_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "input_fifo2_cons_buff_0"} : memref<10xi32>
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "input_fifo2_cons_buff_1"} : memref<10xi32>
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_4]], 0) {init = 2 : i32, sym_name = "input_fifo2_cons_prod_lock_0"}
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_4]], 1) {init = 0 : i32, sym_name = "input_fifo2_cons_cons_lock_0"}
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_2]], 4) {init = 0 : i32, sym_name = "input_fifo2_prod_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_2]], 5) {init = 0 : i32, sym_name = "input_fifo2_cons_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.lock(%[[VAL_2]], 2) {init = 0 : i32, sym_name = "output_fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_18:.*]] = aie.lock(%[[VAL_2]], 3) {init = 0 : i32, sym_name = "output_fifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_19:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "output_fifo_buff_0"} : memref<10xi32>
// CHECK:           %[[VAL_20:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "output_fifo_buff_1"} : memref<10xi32>
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_3]], 2) {init = 2 : i32, sym_name = "output_fifo_prod_lock_0"}
// CHECK:           %[[VAL_22:.*]] = aie.lock(%[[VAL_3]], 3) {init = 0 : i32, sym_name = "output_fifo_cons_lock_0"}
// CHECK:           %[[VAL_23:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32>
// CHECK:           %[[VAL_24:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "input_fifo_cons_buff_1"} : memref<10xi32>
// CHECK:           %[[VAL_25:.*]] = aie.lock(%[[VAL_3]], 0) {init = 2 : i32, sym_name = "input_fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_26:.*]] = aie.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_27:.*]] = aie.lock(%[[VAL_2]], 0) {init = 0 : i32, sym_name = "input_fifo_prod_lock_0"}
// CHECK:           %[[VAL_28:.*]] = aie.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_2]], DMA : 1, %[[VAL_4]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_4]], DMA : 0, %[[VAL_2]], DMA : 1)
// CHECK:           %[[VAL_29:.*]] = aie.buffer(%[[VAL_3]]) : memref<2xi32>
// CHECK:           %[[VAL_30:.*]] = aie.buffer(%[[VAL_3]]) : memref<2xi32>
// CHECK:           %[[VAL_31:.*]] = aie.core(%[[VAL_3]]) {
// CHECK:             %[[VAL_32:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_33:.*]] = arith.constant 0 : index
// CHECK:             memref.store %[[VAL_32]], %[[VAL_30]]{{\[}}%[[VAL_33]]] : memref<2xi32>
// CHECK:             %[[VAL_34:.*]] = arith.constant 1 : index
// CHECK:             memref.store %[[VAL_32]], %[[VAL_30]]{{\[}}%[[VAL_34]]] : memref<2xi32>
// CHECK:             %[[VAL_35:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_36:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_37:.*]] = arith.constant 2 : i32
// CHECK:             memref.store %[[VAL_35]], %[[VAL_29]]{{\[}}%[[VAL_36]]] : memref<2xi32>
// CHECK:             %[[VAL_38:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_39:.*]] = arith.constant 2 : i32
// CHECK:             memref.store %[[VAL_35]], %[[VAL_29]]{{\[}}%[[VAL_38]]] : memref<2xi32>
// CHECK:             %[[VAL_40:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_41:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_42:.*]] = arith.constant 10 : index
// CHECK:             scf.for %[[VAL_43:.*]] = %[[VAL_40]] to %[[VAL_42]] step %[[VAL_41]] {
// CHECK:               %[[VAL_44:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_33]]] : memref<2xi32>
// CHECK:               %[[VAL_45:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_46:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_47:.*]] = arith.subi %[[VAL_45]], %[[VAL_44]] : i32
// CHECK:               %[[VAL_48:.*]] = arith.maxsi %[[VAL_47]], %[[VAL_46]] : i32
// CHECK:               aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_48]])
// CHECK:               %[[VAL_49:.*]] = arith.addi %[[VAL_44]], %[[VAL_48]] : i32
// CHECK:               memref.store %[[VAL_49]], %[[VAL_30]]{{\[}}%[[VAL_33]]] : memref<2xi32>
// CHECK:               %[[VAL_50:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_36]]] : memref<2xi32>
// CHECK:               %[[VAL_51:.*]] = arith.index_cast %[[VAL_50]] : i32 to index
// CHECK:               %[[VAL_52:.*]] = scf.index_switch %[[VAL_51]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_19]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_20]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_19]] : memref<10xi32>
// CHECK:               }
// CHECK:               %[[VAL_53:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_34]]] : memref<2xi32>
// CHECK:               %[[VAL_54:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_55:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_56:.*]] = arith.subi %[[VAL_54]], %[[VAL_53]] : i32
// CHECK:               %[[VAL_57:.*]] = arith.maxsi %[[VAL_56]], %[[VAL_55]] : i32
// CHECK:               aie.use_lock(%[[VAL_26]], AcquireGreaterEqual, %[[VAL_57]])
// CHECK:               %[[VAL_58:.*]] = arith.addi %[[VAL_53]], %[[VAL_57]] : i32
// CHECK:               memref.store %[[VAL_58]], %[[VAL_30]]{{\[}}%[[VAL_34]]] : memref<2xi32>
// CHECK:               %[[VAL_59:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_38]]] : memref<2xi32>
// CHECK:               %[[VAL_60:.*]] = arith.index_cast %[[VAL_59]] : i32 to index
// CHECK:               %[[VAL_61:.*]] = scf.index_switch %[[VAL_60]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_23]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_24]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_23]] : memref<10xi32>
// CHECK:               }
// CHECK:               func.call @passthrough_10_i32(%[[VAL_61]], %[[VAL_52]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               %[[VAL_62:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_25]], Release, %[[VAL_62]])
// CHECK:               %[[VAL_63:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_34]]] : memref<2xi32>
// CHECK:               %[[VAL_64:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_65:.*]] = arith.subi %[[VAL_63]], %[[VAL_64]] : i32
// CHECK:               memref.store %[[VAL_65]], %[[VAL_30]]{{\[}}%[[VAL_34]]] : memref<2xi32>
// CHECK:               %[[VAL_66:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_38]]] : memref<2xi32>
// CHECK:               %[[VAL_67:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_68:.*]] = arith.addi %[[VAL_66]], %[[VAL_67]] : i32
// CHECK:               %[[VAL_69:.*]] = arith.cmpi sge, %[[VAL_68]], %[[VAL_39]] : i32
// CHECK:               %[[VAL_70:.*]] = arith.subi %[[VAL_68]], %[[VAL_39]] : i32
// CHECK:               %[[VAL_71:.*]] = arith.select %[[VAL_69]], %[[VAL_70]], %[[VAL_68]] : i32
// CHECK:               memref.store %[[VAL_71]], %[[VAL_29]]{{\[}}%[[VAL_38]]] : memref<2xi32>
// CHECK:               %[[VAL_72:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_22]], Release, %[[VAL_72]])
// CHECK:               %[[VAL_73:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_33]]] : memref<2xi32>
// CHECK:               %[[VAL_74:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_75:.*]] = arith.subi %[[VAL_73]], %[[VAL_74]] : i32
// CHECK:               memref.store %[[VAL_75]], %[[VAL_30]]{{\[}}%[[VAL_33]]] : memref<2xi32>
// CHECK:               %[[VAL_76:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_36]]] : memref<2xi32>
// CHECK:               %[[VAL_77:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_78:.*]] = arith.addi %[[VAL_76]], %[[VAL_77]] : i32
// CHECK:               %[[VAL_79:.*]] = arith.cmpi sge, %[[VAL_78]], %[[VAL_37]] : i32
// CHECK:               %[[VAL_80:.*]] = arith.subi %[[VAL_78]], %[[VAL_37]] : i32
// CHECK:               %[[VAL_81:.*]] = arith.select %[[VAL_79]], %[[VAL_80]], %[[VAL_78]] : i32
// CHECK:               memref.store %[[VAL_81]], %[[VAL_29]]{{\[}}%[[VAL_36]]] : memref<2xi32>
// CHECK:             }
// CHECK:             aie.end
// CHECK:           } {dynamic_objfifo_lowering = false}
// CHECK:           %[[VAL_82:.*]] = aie.buffer(%[[VAL_4]]) : memref<2xi32>
// CHECK:           %[[VAL_83:.*]] = aie.buffer(%[[VAL_4]]) : memref<2xi32>
// CHECK:           %[[VAL_84:.*]] = aie.core(%[[VAL_4]]) {
// CHECK:             %[[VAL_85:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_86:.*]] = arith.constant 0 : index
// CHECK:             memref.store %[[VAL_85]], %[[VAL_83]]{{\[}}%[[VAL_86]]] : memref<2xi32>
// CHECK:             %[[VAL_87:.*]] = arith.constant 1 : index
// CHECK:             memref.store %[[VAL_85]], %[[VAL_83]]{{\[}}%[[VAL_87]]] : memref<2xi32>
// CHECK:             %[[VAL_88:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_89:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_90:.*]] = arith.constant 2 : i32
// CHECK:             memref.store %[[VAL_88]], %[[VAL_82]]{{\[}}%[[VAL_89]]] : memref<2xi32>
// CHECK:             %[[VAL_91:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_92:.*]] = arith.constant 2 : i32
// CHECK:             memref.store %[[VAL_88]], %[[VAL_82]]{{\[}}%[[VAL_91]]] : memref<2xi32>
// CHECK:             %[[VAL_93:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_94:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_95:.*]] = arith.constant 10 : index
// CHECK:             scf.for %[[VAL_96:.*]] = %[[VAL_93]] to %[[VAL_95]] step %[[VAL_94]] {
// CHECK:               %[[VAL_97:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_86]]] : memref<2xi32>
// CHECK:               %[[VAL_98:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_99:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_100:.*]] = arith.subi %[[VAL_98]], %[[VAL_97]] : i32
// CHECK:               %[[VAL_101:.*]] = arith.maxsi %[[VAL_100]], %[[VAL_99]] : i32
// CHECK:               aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_101]])
// CHECK:               %[[VAL_102:.*]] = arith.addi %[[VAL_97]], %[[VAL_101]] : i32
// CHECK:               memref.store %[[VAL_102]], %[[VAL_83]]{{\[}}%[[VAL_86]]] : memref<2xi32>
// CHECK:               %[[VAL_103:.*]] = memref.load %[[VAL_82]]{{\[}}%[[VAL_89]]] : memref<2xi32>
// CHECK:               %[[VAL_104:.*]] = arith.index_cast %[[VAL_103]] : i32 to index
// CHECK:               %[[VAL_105:.*]] = scf.index_switch %[[VAL_104]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_8]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:               }
// CHECK:               %[[VAL_106:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_87]]] : memref<2xi32>
// CHECK:               %[[VAL_107:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_108:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_109:.*]] = arith.subi %[[VAL_107]], %[[VAL_106]] : i32
// CHECK:               %[[VAL_110:.*]] = arith.maxsi %[[VAL_109]], %[[VAL_108]] : i32
// CHECK:               aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_110]])
// CHECK:               %[[VAL_111:.*]] = arith.addi %[[VAL_106]], %[[VAL_110]] : i32
// CHECK:               memref.store %[[VAL_111]], %[[VAL_83]]{{\[}}%[[VAL_87]]] : memref<2xi32>
// CHECK:               %[[VAL_112:.*]] = memref.load %[[VAL_82]]{{\[}}%[[VAL_91]]] : memref<2xi32>
// CHECK:               %[[VAL_113:.*]] = arith.index_cast %[[VAL_112]] : i32 to index
// CHECK:               %[[VAL_114:.*]] = scf.index_switch %[[VAL_113]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:               }
// CHECK:               func.call @passthrough_10_i32(%[[VAL_114]], %[[VAL_105]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               %[[VAL_115:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_13]], Release, %[[VAL_115]])
// CHECK:               %[[VAL_116:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_87]]] : memref<2xi32>
// CHECK:               %[[VAL_117:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_118:.*]] = arith.subi %[[VAL_116]], %[[VAL_117]] : i32
// CHECK:               memref.store %[[VAL_118]], %[[VAL_83]]{{\[}}%[[VAL_87]]] : memref<2xi32>
// CHECK:               %[[VAL_119:.*]] = memref.load %[[VAL_82]]{{\[}}%[[VAL_91]]] : memref<2xi32>
// CHECK:               %[[VAL_120:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_121:.*]] = arith.addi %[[VAL_119]], %[[VAL_120]] : i32
// CHECK:               %[[VAL_122:.*]] = arith.cmpi sge, %[[VAL_121]], %[[VAL_92]] : i32
// CHECK:               %[[VAL_123:.*]] = arith.subi %[[VAL_121]], %[[VAL_92]] : i32
// CHECK:               %[[VAL_124:.*]] = arith.select %[[VAL_122]], %[[VAL_123]], %[[VAL_121]] : i32
// CHECK:               memref.store %[[VAL_124]], %[[VAL_82]]{{\[}}%[[VAL_91]]] : memref<2xi32>
// CHECK:               %[[VAL_125:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_10]], Release, %[[VAL_125]])
// CHECK:               %[[VAL_126:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_86]]] : memref<2xi32>
// CHECK:               %[[VAL_127:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_128:.*]] = arith.subi %[[VAL_126]], %[[VAL_127]] : i32
// CHECK:               memref.store %[[VAL_128]], %[[VAL_83]]{{\[}}%[[VAL_86]]] : memref<2xi32>
// CHECK:               %[[VAL_129:.*]] = memref.load %[[VAL_82]]{{\[}}%[[VAL_89]]] : memref<2xi32>
// CHECK:               %[[VAL_130:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_131:.*]] = arith.addi %[[VAL_129]], %[[VAL_130]] : i32
// CHECK:               %[[VAL_132:.*]] = arith.cmpi sge, %[[VAL_131]], %[[VAL_90]] : i32
// CHECK:               %[[VAL_133:.*]] = arith.subi %[[VAL_131]], %[[VAL_90]] : i32
// CHECK:               %[[VAL_134:.*]] = arith.select %[[VAL_132]], %[[VAL_133]], %[[VAL_131]] : i32
// CHECK:               memref.store %[[VAL_134]], %[[VAL_82]]{{\[}}%[[VAL_89]]] : memref<2xi32>
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @input_fifo_shim_alloc(%[[VAL_2]], MM2S, 0)
// CHECK:           %[[VAL_135:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_136:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_137:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_25]], AcquireGreaterEqual, %[[VAL_137]])
// CHECK:             aie.dma_bd(%[[VAL_23]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_138:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_26]], Release, %[[VAL_138]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_139:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_25]], AcquireGreaterEqual, %[[VAL_139]])
// CHECK:             aie.dma_bd(%[[VAL_24]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_140:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_26]], Release, %[[VAL_140]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_141:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             %[[VAL_142:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, %[[VAL_142]])
// CHECK:             aie.dma_bd(%[[VAL_19]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_143:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], Release, %[[VAL_143]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             %[[VAL_144:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, %[[VAL_144]])
// CHECK:             aie.dma_bd(%[[VAL_20]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_145:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], Release, %[[VAL_145]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @output_fifo_shim_alloc(%[[VAL_2]], S2MM, 0)
// CHECK:           aie.shim_dma_allocation @input_fifo2_shim_alloc(%[[VAL_2]], MM2S, 1)
// CHECK:           %[[VAL_146:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_147:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_148:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_148]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_149:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_149]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_150:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_150]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_151:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_151]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_152:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             %[[VAL_153:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_153]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_154:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_154]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             %[[VAL_155:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_155]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_156:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_156]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @output_fifo2_shim_alloc(%[[VAL_2]], S2MM, 1)
// CHECK:         }
// CHECK:       }

module {
  aie.device(npu1_1col) {
    func.func @passthrough_10_i32(%line_in: memref<10xi32>, %line_out: memref<10xi32>) -> () {
        return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<10xi32>>
    aie.objectfifo @output_fifo(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<10xi32>>

    aie.objectfifo @input_fifo2(%tile_0_0, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<10xi32>>
    aie.objectfifo @output_fifo2(%tile_0_4, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %arg0 = %c0 to %c10 step %c1 {
        %0 = aie.objectfifo.acquire @output_fifo(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        %2 = aie.objectfifo.acquire @input_fifo(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        func.call @passthrough_10_i32(%3, %1) : (memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo(Consume, 1)
        aie.objectfifo.release @output_fifo(Produce, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = false}

    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %arg0 = %c0 to %c10 step %c1 {
        %0 = aie.objectfifo.acquire @output_fifo2(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        %2 = aie.objectfifo.acquire @input_fifo2(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        func.call @passthrough_10_i32(%3, %1) : (memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo2(Consume, 1)
        aie.objectfifo.release @output_fifo2(Produce, 1)
      }

      aie.end
    }
  }
}
