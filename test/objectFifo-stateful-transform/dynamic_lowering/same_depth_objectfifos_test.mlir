//===- same_depth_objectfifos_test.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform=dynamic-objFifos %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_1col) {
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
// CHECK:           %[[VAL_18:.*]] = aie.core(%[[VAL_4]]) {
// CHECK:             %[[VAL_19:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_20:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_21:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_22:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_23:.*]] = arith.constant 2 : i32
// CHECK:             memref.store %[[VAL_21]], %[[VAL_17]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_25:.*]] = arith.constant 2 : i32
// CHECK:             memref.store %[[VAL_21]], %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:             %[[VAL_26:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_27:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_28:.*]] = arith.constant 9 : index
// CHECK:             %[[VAL_29:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_30:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_31:.*]] = arith.subi %[[VAL_29]], %[[VAL_20]] : i32
// CHECK:             %[[VAL_32:.*]] = arith.maxsi %[[VAL_31]], %[[VAL_30]] : i32
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_32]])
// CHECK:             %[[VAL_33:.*]] = arith.addi %[[VAL_20]], %[[VAL_32]] : i32
// CHECK:             %[[VAL_34:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_35:.*]] = arith.index_cast %[[VAL_34]] : i32 to index
// CHECK:             %[[VAL_36:.*]] = scf.index_switch %[[VAL_35]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_8]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[VAL_37:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_38:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_39:.*]] = arith.subi %[[VAL_37]], %[[VAL_19]] : i32
// CHECK:             %[[VAL_40:.*]] = arith.maxsi %[[VAL_39]], %[[VAL_38]] : i32
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_40]])
// CHECK:             %[[VAL_41:.*]] = arith.addi %[[VAL_19]], %[[VAL_40]] : i32
// CHECK:             %[[VAL_42:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:             %[[VAL_43:.*]] = arith.index_cast %[[VAL_42]] : i32 to index
// CHECK:             %[[VAL_44:.*]] = scf.index_switch %[[VAL_43]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:             }
// CHECK:             func.call @add_10_i32(%[[VAL_44]], %[[VAL_44]], %[[VAL_36]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:             %[[VAL_45:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_45]])
// CHECK:             %[[VAL_46:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_47:.*]] = arith.subi %[[VAL_33]], %[[VAL_46]] : i32
// CHECK:             %[[VAL_48:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_49:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_50:.*]] = arith.addi %[[VAL_48]], %[[VAL_49]] : i32
// CHECK:             %[[VAL_51:.*]] = arith.cmpi sge, %[[VAL_50]], %[[VAL_23]] : i32
// CHECK:             %[[VAL_52:.*]] = arith.subi %[[VAL_50]], %[[VAL_23]] : i32
// CHECK:             %[[VAL_53:.*]] = arith.select %[[VAL_51]], %[[VAL_52]], %[[VAL_50]] : i32
// CHECK:             memref.store %[[VAL_53]], %[[VAL_17]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_54:.*]]:2 = scf.for %[[VAL_55:.*]] = %[[VAL_26]] to %[[VAL_28]] step %[[VAL_27]] iter_args(%[[VAL_56:.*]] = %[[VAL_47]], %[[VAL_57:.*]] = %[[VAL_41]]) -> (i32, i32) {
// CHECK:               %[[VAL_58:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_59:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_60:.*]] = arith.subi %[[VAL_58]], %[[VAL_56]] : i32
// CHECK:               %[[VAL_61:.*]] = arith.maxsi %[[VAL_60]], %[[VAL_59]] : i32
// CHECK:               aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_61]])
// CHECK:               %[[VAL_62:.*]] = arith.addi %[[VAL_56]], %[[VAL_61]] : i32
// CHECK:               %[[VAL_63:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:               %[[VAL_64:.*]] = arith.index_cast %[[VAL_63]] : i32 to index
// CHECK:               %[[VAL_65:.*]] = scf.index_switch %[[VAL_64]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_8]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:               }
// CHECK:               %[[VAL_66:.*]] = arith.constant 2 : i32
// CHECK:               %[[VAL_67:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_68:.*]] = arith.subi %[[VAL_66]], %[[VAL_57]] : i32
// CHECK:               %[[VAL_69:.*]] = arith.maxsi %[[VAL_68]], %[[VAL_67]] : i32
// CHECK:               aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_69]])
// CHECK:               %[[VAL_70:.*]] = arith.addi %[[VAL_57]], %[[VAL_69]] : i32
// CHECK:               %[[VAL_71:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:               %[[VAL_72:.*]] = arith.index_cast %[[VAL_71]] : i32 to index
// CHECK:               %[[VAL_73:.*]] = scf.index_switch %[[VAL_72]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:               }
// CHECK:               %[[VAL_74:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:               %[[VAL_75:.*]] = arith.index_cast %[[VAL_74]] : i32 to index
// CHECK:               %[[VAL_76:.*]] = scf.index_switch %[[VAL_75]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:               }
// CHECK:               func.call @add_10_i32(%[[VAL_73]], %[[VAL_76]], %[[VAL_65]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               %[[VAL_77:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_13]], Release, %[[VAL_77]])
// CHECK:               %[[VAL_78:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_79:.*]] = arith.subi %[[VAL_70]], %[[VAL_78]] : i32
// CHECK:               %[[VAL_80:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:               %[[VAL_81:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_82:.*]] = arith.addi %[[VAL_80]], %[[VAL_81]] : i32
// CHECK:               %[[VAL_83:.*]] = arith.cmpi sge, %[[VAL_82]], %[[VAL_25]] : i32
// CHECK:               %[[VAL_84:.*]] = arith.subi %[[VAL_82]], %[[VAL_25]] : i32
// CHECK:               %[[VAL_85:.*]] = arith.select %[[VAL_83]], %[[VAL_84]], %[[VAL_82]] : i32
// CHECK:               memref.store %[[VAL_85]], %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:               %[[VAL_86:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_10]], Release, %[[VAL_86]])
// CHECK:               %[[VAL_87:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_88:.*]] = arith.subi %[[VAL_62]], %[[VAL_87]] : i32
// CHECK:               %[[VAL_89:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:               %[[VAL_90:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_91:.*]] = arith.addi %[[VAL_89]], %[[VAL_90]] : i32
// CHECK:               %[[VAL_92:.*]] = arith.cmpi sge, %[[VAL_91]], %[[VAL_23]] : i32
// CHECK:               %[[VAL_93:.*]] = arith.subi %[[VAL_91]], %[[VAL_23]] : i32
// CHECK:               %[[VAL_94:.*]] = arith.select %[[VAL_92]], %[[VAL_93]], %[[VAL_91]] : i32
// CHECK:               memref.store %[[VAL_94]], %[[VAL_17]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:               scf.yield %[[VAL_88]], %[[VAL_79]] : i32, i32
// CHECK:             }
// CHECK:             %[[VAL_95:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_96:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_97:.*]] = arith.subi %[[VAL_95]], %[[VAL_98:.*]]#0 : i32
// CHECK:             %[[VAL_99:.*]] = arith.maxsi %[[VAL_97]], %[[VAL_96]] : i32
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_99]])
// CHECK:             %[[VAL_100:.*]] = arith.addi %[[VAL_98]]#0, %[[VAL_99]] : i32
// CHECK:             %[[VAL_101:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_102:.*]] = arith.index_cast %[[VAL_101]] : i32 to index
// CHECK:             %[[VAL_103:.*]] = scf.index_switch %[[VAL_102]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_8]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[VAL_104:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_105:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_106:.*]] = arith.subi %[[VAL_104]], %[[VAL_98]]#1 : i32
// CHECK:             %[[VAL_107:.*]] = arith.maxsi %[[VAL_106]], %[[VAL_105]] : i32
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_107]])
// CHECK:             %[[VAL_108:.*]] = arith.addi %[[VAL_98]]#1, %[[VAL_107]] : i32
// CHECK:             %[[VAL_109:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:             %[[VAL_110:.*]] = arith.index_cast %[[VAL_109]] : i32 to index
// CHECK:             %[[VAL_111:.*]] = scf.index_switch %[[VAL_110]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[VAL_112:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:             %[[VAL_113:.*]] = arith.index_cast %[[VAL_112]] : i32 to index
// CHECK:             %[[VAL_114:.*]] = scf.index_switch %[[VAL_113]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_11]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_12]] : memref<10xi32>
// CHECK:             }
// CHECK:             func.call @add_10_i32(%[[VAL_111]], %[[VAL_114]], %[[VAL_103]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:             %[[VAL_115:.*]] = arith.constant 2 : i32
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[VAL_115]])
// CHECK:             %[[VAL_116:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_117:.*]] = arith.subi %[[VAL_108]], %[[VAL_116]] : i32
// CHECK:             %[[VAL_118:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:             %[[VAL_119:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_120:.*]] = arith.addi %[[VAL_118]], %[[VAL_119]] : i32
// CHECK:             %[[VAL_121:.*]] = arith.cmpi sge, %[[VAL_120]], %[[VAL_25]] : i32
// CHECK:             %[[VAL_122:.*]] = arith.subi %[[VAL_120]], %[[VAL_25]] : i32
// CHECK:             %[[VAL_123:.*]] = arith.select %[[VAL_121]], %[[VAL_122]], %[[VAL_120]] : i32
// CHECK:             memref.store %[[VAL_123]], %[[VAL_17]]{{\[}}%[[VAL_24]]] : memref<2xi32>
// CHECK:             %[[VAL_124:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_124]])
// CHECK:             %[[VAL_125:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_126:.*]] = arith.subi %[[VAL_100]], %[[VAL_125]] : i32
// CHECK:             %[[VAL_127:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             %[[VAL_128:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_129:.*]] = arith.addi %[[VAL_127]], %[[VAL_128]] : i32
// CHECK:             %[[VAL_130:.*]] = arith.cmpi sge, %[[VAL_129]], %[[VAL_23]] : i32
// CHECK:             %[[VAL_131:.*]] = arith.subi %[[VAL_129]], %[[VAL_23]] : i32
// CHECK:             %[[VAL_132:.*]] = arith.select %[[VAL_130]], %[[VAL_131]], %[[VAL_129]] : i32
// CHECK:             memref.store %[[VAL_132]], %[[VAL_17]]{{\[}}%[[VAL_22]]] : memref<2xi32>
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @input_fifo_shim_alloc(%[[VAL_3]], MM2S, 0)
// CHECK:           %[[VAL_133:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_134:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_135:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_135]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_136:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_136]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_137:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_137]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_138:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_138]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_139:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             %[[VAL_140:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_140]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_141:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_141]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             %[[VAL_142:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_142]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_143:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_143]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @output_fifo_shim_alloc(%[[VAL_3]], S2MM, 0)
// CHECK:         }

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
