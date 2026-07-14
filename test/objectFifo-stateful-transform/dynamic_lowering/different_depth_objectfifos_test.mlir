//===- different_depth_objectfifos_test.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
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
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "input_fifo_cons_buff_2"} : memref<10xi32>
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_4]], 0) {init = 3 : i32, sym_name = "input_fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_4]], 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_3]], 0) {init = 0 : i32, sym_name = "input_fifo_prod_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_4]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_4]]) : memref<2xi32>
// CHECK:           %[[VAL_19:.*]] = aie.core(%[[VAL_4]]) {
// CHECK:             %[[VAL_20:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_21:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_22:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_24:.*]] = arith.constant 2 : i32
// CHECK:             memref.store %[[VAL_22]], %[[VAL_18]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:             %[[VAL_25:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_26:.*]] = arith.constant 3 : i32
// CHECK:             memref.store %[[VAL_22]], %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:             %[[VAL_27:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_28:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_29:.*]] = arith.constant 9 : index
// CHECK:             %[[VAL_30:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_31:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_32:.*]] = arith.subi %[[VAL_30]], %[[VAL_21]] : i32
// CHECK:             %[[VAL_33:.*]] = arith.maxsi %[[VAL_32]], %[[VAL_31]] : i32
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             %[[VAL_34:.*]] = arith.addi %[[VAL_21]], %[[VAL_33]] : i32
// CHECK:             %[[VAL_35:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:             %[[VAL_36:.*]] = arith.index_cast %[[VAL_35]] : i32 to index
// CHECK:             %[[VAL_37:.*]] = scf.index_switch %[[VAL_36]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_8]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[VAL_38:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_39:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_40:.*]] = arith.subi %[[VAL_38]], %[[VAL_20]] : i32
// CHECK:             %[[VAL_41:.*]] = arith.maxsi %[[VAL_40]], %[[VAL_39]] : i32
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_41]])
// CHECK:             %[[VAL_42:.*]] = arith.addi %[[VAL_20]], %[[VAL_41]] : i32
// CHECK:             %[[VAL_43:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:             %[[VAL_44:.*]] = arith.index_cast %[[VAL_43]] : i32 to index
// CHECK:             %[[VAL_45:.*]] = scf.index_switch %[[VAL_44]] -> memref<10xi32>
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
// CHECK:             func.call @add_10_i32(%[[VAL_45]], %[[VAL_45]], %[[VAL_37]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:             %[[VAL_46:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_46]])
// CHECK:             %[[VAL_47:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_48:.*]] = arith.subi %[[VAL_34]], %[[VAL_47]] : i32
// CHECK:             %[[VAL_49:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:             %[[VAL_50:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_51:.*]] = arith.addi %[[VAL_49]], %[[VAL_50]] : i32
// CHECK:             %[[VAL_52:.*]] = arith.cmpi sge, %[[VAL_51]], %[[VAL_24]] : i32
// CHECK:             %[[VAL_53:.*]] = arith.subi %[[VAL_51]], %[[VAL_24]] : i32
// CHECK:             %[[VAL_54:.*]] = arith.select %[[VAL_52]], %[[VAL_53]], %[[VAL_51]] : i32
// CHECK:             memref.store %[[VAL_54]], %[[VAL_18]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:             %[[VAL_55:.*]]:2 = scf.for %[[VAL_56:.*]] = %[[VAL_27]] to %[[VAL_29]] step %[[VAL_28]] iter_args(%[[VAL_57:.*]] = %[[VAL_48]], %[[VAL_58:.*]] = %[[VAL_42]]) -> (i32, i32) {
// CHECK:               %[[VAL_59:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_60:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_61:.*]] = arith.subi %[[VAL_59]], %[[VAL_57]] : i32
// CHECK:               %[[VAL_62:.*]] = arith.maxsi %[[VAL_61]], %[[VAL_60]] : i32
// CHECK:               aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_62]])
// CHECK:               %[[VAL_63:.*]] = arith.addi %[[VAL_57]], %[[VAL_62]] : i32
// CHECK:               %[[VAL_64:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:               %[[VAL_65:.*]] = arith.index_cast %[[VAL_64]] : i32 to index
// CHECK:               %[[VAL_66:.*]] = scf.index_switch %[[VAL_65]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_8]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:               }
// CHECK:               %[[VAL_67:.*]] = arith.constant 2 : i32
// CHECK:               %[[VAL_68:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_69:.*]] = arith.subi %[[VAL_67]], %[[VAL_58]] : i32
// CHECK:               %[[VAL_70:.*]] = arith.maxsi %[[VAL_69]], %[[VAL_68]] : i32
// CHECK:               aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_70]])
// CHECK:               %[[VAL_71:.*]] = arith.addi %[[VAL_58]], %[[VAL_70]] : i32
// CHECK:               %[[VAL_72:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:               %[[VAL_73:.*]] = arith.index_cast %[[VAL_72]] : i32 to index
// CHECK:               %[[VAL_74:.*]] = scf.index_switch %[[VAL_73]] -> memref<10xi32>
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
// CHECK:               %[[VAL_75:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:               %[[VAL_76:.*]] = arith.index_cast %[[VAL_75]] : i32 to index
// CHECK:               %[[VAL_77:.*]] = scf.index_switch %[[VAL_76]] -> memref<10xi32>
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
// CHECK:               func.call @add_10_i32(%[[VAL_74]], %[[VAL_77]], %[[VAL_66]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               %[[VAL_78:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_14]], Release, %[[VAL_78]])
// CHECK:               %[[VAL_79:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_80:.*]] = arith.subi %[[VAL_71]], %[[VAL_79]] : i32
// CHECK:               %[[VAL_81:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:               %[[VAL_82:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_83:.*]] = arith.addi %[[VAL_81]], %[[VAL_82]] : i32
// CHECK:               %[[VAL_84:.*]] = arith.cmpi sge, %[[VAL_83]], %[[VAL_26]] : i32
// CHECK:               %[[VAL_85:.*]] = arith.subi %[[VAL_83]], %[[VAL_26]] : i32
// CHECK:               %[[VAL_86:.*]] = arith.select %[[VAL_84]], %[[VAL_85]], %[[VAL_83]] : i32
// CHECK:               memref.store %[[VAL_86]], %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:               %[[VAL_87:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_10]], Release, %[[VAL_87]])
// CHECK:               %[[VAL_88:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_89:.*]] = arith.subi %[[VAL_63]], %[[VAL_88]] : i32
// CHECK:               %[[VAL_90:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:               %[[VAL_91:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_92:.*]] = arith.addi %[[VAL_90]], %[[VAL_91]] : i32
// CHECK:               %[[VAL_93:.*]] = arith.cmpi sge, %[[VAL_92]], %[[VAL_24]] : i32
// CHECK:               %[[VAL_94:.*]] = arith.subi %[[VAL_92]], %[[VAL_24]] : i32
// CHECK:               %[[VAL_95:.*]] = arith.select %[[VAL_93]], %[[VAL_94]], %[[VAL_92]] : i32
// CHECK:               memref.store %[[VAL_95]], %[[VAL_18]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:               scf.yield %[[VAL_89]], %[[VAL_80]] : i32, i32
// CHECK:             }
// CHECK:             %[[VAL_96:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_97:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_98:.*]] = arith.subi %[[VAL_96]], %[[VAL_99:.*]]#0 : i32
// CHECK:             %[[VAL_100:.*]] = arith.maxsi %[[VAL_98]], %[[VAL_97]] : i32
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_100]])
// CHECK:             %[[VAL_101:.*]] = arith.addi %[[VAL_99]]#0, %[[VAL_100]] : i32
// CHECK:             %[[VAL_102:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:             %[[VAL_103:.*]] = arith.index_cast %[[VAL_102]] : i32 to index
// CHECK:             %[[VAL_104:.*]] = scf.index_switch %[[VAL_103]] -> memref<10xi32>
// CHECK:             case 0 {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             case 1 {
// CHECK:               scf.yield %[[VAL_8]] : memref<10xi32>
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_7]] : memref<10xi32>
// CHECK:             }
// CHECK:             %[[VAL_105:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_106:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_107:.*]] = arith.subi %[[VAL_105]], %[[VAL_99]]#1 : i32
// CHECK:             %[[VAL_108:.*]] = arith.maxsi %[[VAL_107]], %[[VAL_106]] : i32
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_108]])
// CHECK:             %[[VAL_109:.*]] = arith.addi %[[VAL_99]]#1, %[[VAL_108]] : i32
// CHECK:             %[[VAL_110:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:             %[[VAL_111:.*]] = arith.index_cast %[[VAL_110]] : i32 to index
// CHECK:             %[[VAL_112:.*]] = scf.index_switch %[[VAL_111]] -> memref<10xi32>
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
// CHECK:             %[[VAL_113:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:             %[[VAL_114:.*]] = arith.index_cast %[[VAL_113]] : i32 to index
// CHECK:             %[[VAL_115:.*]] = scf.index_switch %[[VAL_114]] -> memref<10xi32>
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
// CHECK:             func.call @add_10_i32(%[[VAL_112]], %[[VAL_115]], %[[VAL_104]]) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
// CHECK:             %[[VAL_116:.*]] = arith.constant 2 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_116]])
// CHECK:             %[[VAL_117:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_118:.*]] = arith.subi %[[VAL_109]], %[[VAL_117]] : i32
// CHECK:             %[[VAL_119:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:             %[[VAL_120:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_121:.*]] = arith.addi %[[VAL_119]], %[[VAL_120]] : i32
// CHECK:             %[[VAL_122:.*]] = arith.cmpi sge, %[[VAL_121]], %[[VAL_26]] : i32
// CHECK:             %[[VAL_123:.*]] = arith.subi %[[VAL_121]], %[[VAL_26]] : i32
// CHECK:             %[[VAL_124:.*]] = arith.select %[[VAL_122]], %[[VAL_123]], %[[VAL_121]] : i32
// CHECK:             memref.store %[[VAL_124]], %[[VAL_18]]{{\[}}%[[VAL_25]]] : memref<2xi32>
// CHECK:             %[[VAL_125:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_125]])
// CHECK:             %[[VAL_126:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_127:.*]] = arith.subi %[[VAL_101]], %[[VAL_126]] : i32
// CHECK:             %[[VAL_128:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:             %[[VAL_129:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_130:.*]] = arith.addi %[[VAL_128]], %[[VAL_129]] : i32
// CHECK:             %[[VAL_131:.*]] = arith.cmpi sge, %[[VAL_130]], %[[VAL_24]] : i32
// CHECK:             %[[VAL_132:.*]] = arith.subi %[[VAL_130]], %[[VAL_24]] : i32
// CHECK:             %[[VAL_133:.*]] = arith.select %[[VAL_131]], %[[VAL_132]], %[[VAL_130]] : i32
// CHECK:             memref.store %[[VAL_133]], %[[VAL_18]]{{\[}}%[[VAL_23]]] : memref<2xi32>
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @input_fifo_shim_alloc(%[[VAL_3]], MM2S, 0)
// CHECK:           %[[VAL_134:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_135:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_136:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_136]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_137:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_137]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_138:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_138]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_139:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_139]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_140:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_140]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_141:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_141]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %[[VAL_142:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb7)
// CHECK:           ^bb5:
// CHECK:             %[[VAL_143:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_143]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_144:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_144]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             %[[VAL_145:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_145]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<10xi32> offset = {{.*}} len = {{.*}})
// CHECK:             %[[VAL_146:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_146]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb7:
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
