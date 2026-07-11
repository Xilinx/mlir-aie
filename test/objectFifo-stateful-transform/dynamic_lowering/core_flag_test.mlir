//===- core_flag_test.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_1col) {
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
// CHECK:           %[[VAL_30:.*]] = aie.core(%[[VAL_3]]) {
// CHECK:             %[[VAL_31:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_32:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_33:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_34:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_35:.*]] = arith.constant 2 : i32
// CHECK:             memref.store %[[VAL_33]], %[[VAL_29]]{{\[}}%[[VAL_34]]] : memref<2xi32>
// CHECK:             %[[VAL_36:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_37:.*]] = arith.constant 2 : i32
// CHECK:             memref.store %[[VAL_33]], %[[VAL_29]]{{\[}}%[[VAL_36]]] : memref<2xi32>
// CHECK:             %[[VAL_38:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_39:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_40:.*]] = arith.constant 10 : index
// CHECK:             %[[VAL_41:.*]]:2 = scf.for %[[VAL_42:.*]] = %[[VAL_38]] to %[[VAL_40]] step %[[VAL_39]] iter_args(%[[VAL_43:.*]] = %[[VAL_32]], %[[VAL_44:.*]] = %[[VAL_31]]) -> (i32, i32) {
// CHECK:               %[[VAL_45:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_46:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_47:.*]] = arith.subi %[[VAL_45]], %[[VAL_43]] : i32
// CHECK:               %[[VAL_48:.*]] = arith.maxsi %[[VAL_47]], %[[VAL_46]] : i32
// CHECK:               aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_48]])
// CHECK:               %[[VAL_49:.*]] = arith.addi %[[VAL_43]], %[[VAL_48]] : i32
// CHECK:               %[[VAL_50:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_34]]] : memref<2xi32>
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
// CHECK:               %[[VAL_53:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_54:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_55:.*]] = arith.subi %[[VAL_53]], %[[VAL_44]] : i32
// CHECK:               %[[VAL_56:.*]] = arith.maxsi %[[VAL_55]], %[[VAL_54]] : i32
// CHECK:               aie.use_lock(%[[VAL_26]], AcquireGreaterEqual, %[[VAL_56]])
// CHECK:               %[[VAL_57:.*]] = arith.addi %[[VAL_44]], %[[VAL_56]] : i32
// CHECK:               %[[VAL_58:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_36]]] : memref<2xi32>
// CHECK:               %[[VAL_59:.*]] = arith.index_cast %[[VAL_58]] : i32 to index
// CHECK:               %[[VAL_60:.*]] = scf.index_switch %[[VAL_59]] -> memref<10xi32>
// CHECK:               case 0 {
// CHECK:                 scf.yield %[[VAL_23]] : memref<10xi32>
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 scf.yield %[[VAL_24]] : memref<10xi32>
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_23]] : memref<10xi32>
// CHECK:               }
// CHECK:               func.call @passthrough_10_i32(%[[VAL_60]], %[[VAL_52]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               %[[VAL_61:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_25]], Release, %[[VAL_61]])
// CHECK:               %[[VAL_62:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_63:.*]] = arith.subi %[[VAL_57]], %[[VAL_62]] : i32
// CHECK:               %[[VAL_64:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_36]]] : memref<2xi32>
// CHECK:               %[[VAL_65:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_66:.*]] = arith.addi %[[VAL_64]], %[[VAL_65]] : i32
// CHECK:               %[[VAL_67:.*]] = arith.cmpi sge, %[[VAL_66]], %[[VAL_37]] : i32
// CHECK:               %[[VAL_68:.*]] = arith.subi %[[VAL_66]], %[[VAL_37]] : i32
// CHECK:               %[[VAL_69:.*]] = arith.select %[[VAL_67]], %[[VAL_68]], %[[VAL_66]] : i32
// CHECK:               memref.store %[[VAL_69]], %[[VAL_29]]{{\[}}%[[VAL_36]]] : memref<2xi32>
// CHECK:               %[[VAL_70:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_22]], Release, %[[VAL_70]])
// CHECK:               %[[VAL_71:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_72:.*]] = arith.subi %[[VAL_49]], %[[VAL_71]] : i32
// CHECK:               %[[VAL_73:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_34]]] : memref<2xi32>
// CHECK:               %[[VAL_74:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_75:.*]] = arith.addi %[[VAL_73]], %[[VAL_74]] : i32
// CHECK:               %[[VAL_76:.*]] = arith.cmpi sge, %[[VAL_75]], %[[VAL_35]] : i32
// CHECK:               %[[VAL_77:.*]] = arith.subi %[[VAL_75]], %[[VAL_35]] : i32
// CHECK:               %[[VAL_78:.*]] = arith.select %[[VAL_76]], %[[VAL_77]], %[[VAL_75]] : i32
// CHECK:               memref.store %[[VAL_78]], %[[VAL_29]]{{\[}}%[[VAL_34]]] : memref<2xi32>
// CHECK:               scf.yield %[[VAL_72]], %[[VAL_63]] : i32, i32
// CHECK:             }
// CHECK:             aie.end
// CHECK:           } {dynamic_objfifo_lowering = true}
// CHECK:           %[[VAL_79:.*]] = aie.core(%[[VAL_4]]) {
// CHECK:             %[[VAL_80:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_81:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_82:.*]] = arith.constant 10 : index
// CHECK:             %[[VAL_83:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[VAL_84:.*]] = %[[VAL_80]] to %[[VAL_82]] step %[[VAL_83]] {
// CHECK:               %[[VAL_85:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_85]])
// CHECK:               %[[VAL_86:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_86]])
// CHECK:               func.call @passthrough_10_i32(%[[VAL_11]], %[[VAL_7]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               %[[VAL_87:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_13]], Release, %[[VAL_87]])
// CHECK:               %[[VAL_88:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_10]], Release, %[[VAL_88]])
// CHECK:               %[[VAL_89:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_89]])
// CHECK:               %[[VAL_90:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_90]])
// CHECK:               func.call @passthrough_10_i32(%[[VAL_12]], %[[VAL_8]]) : (memref<10xi32>, memref<10xi32>) -> ()
// CHECK:               %[[VAL_91:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_13]], Release, %[[VAL_91]])
// CHECK:               %[[VAL_92:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_10]], Release, %[[VAL_92]])
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @input_fifo_shim_alloc(%[[VAL_2]], MM2S, 0)
// CHECK:           %[[VAL_93:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_94:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_95:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_25]], AcquireGreaterEqual, %[[VAL_95]])
// CHECK:             aie.dma_bd(%[[VAL_23]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_96:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_26]], Release, %[[VAL_96]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_97:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_25]], AcquireGreaterEqual, %[[VAL_97]])
// CHECK:             aie.dma_bd(%[[VAL_24]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_98:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_26]], Release, %[[VAL_98]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_99:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             %[[VAL_100:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, %[[VAL_100]])
// CHECK:             aie.dma_bd(%[[VAL_19]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_101:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], Release, %[[VAL_101]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             %[[VAL_102:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, %[[VAL_102]])
// CHECK:             aie.dma_bd(%[[VAL_20]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_103:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], Release, %[[VAL_103]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @output_fifo_shim_alloc(%[[VAL_2]], S2MM, 0)
// CHECK:           aie.shim_dma_allocation @input_fifo2_shim_alloc(%[[VAL_2]], MM2S, 1)
// CHECK:           %[[VAL_104:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_105:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_106:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_106]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_107:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_107]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_108:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_108]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_109:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_109]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_110:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             %[[VAL_111:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_111]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_112:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_112]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             %[[VAL_113:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_113]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<10xi32>, 0, 10)
// CHECK:             %[[VAL_114:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_114]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @output_fifo2_shim_alloc(%[[VAL_2]], S2MM, 1)
// CHECK:         }

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
    } {dynamic_objfifo_lowering = true}

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
