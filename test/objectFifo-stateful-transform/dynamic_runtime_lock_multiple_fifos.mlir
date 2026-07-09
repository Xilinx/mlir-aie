//===- dynamic_runtime_lock_multiple_fifos.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two objectFifos acquired/released with different counts in the same loop
// body. Each fifo gets its own runtime held counter and its own
// value-carrying `AcquireGreaterEqual`; the trailing post-loop releases
// decrement the respective counters.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK-LABEL: module {
// CHECK:         aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifoY_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifoY_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifoY_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]], 2) {init = 3 : i32, sym_name = "fifoY_cons_prod_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]], 3) {init = 0 : i32, sym_name = "fifoY_cons_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifoY_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifoY_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_0]], 2) {init = 2 : i32, sym_name = "fifoY_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "fifoY_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifoX_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifoX_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifoX_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifoX_cons_buff_3"} : memref<8xi8>
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_1]], 0) {init = 4 : i32, sym_name = "fifoX_cons_prod_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "fifoX_cons_cons_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifoX_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifoX_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_19:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifoX_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_0]], 0) {init = 3 : i32, sym_name = "fifoX_prod_lock_0"}
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "fifoX_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 1, %[[VAL_1]], DMA : 1)
// CHECK:           %[[VAL_22:.*]] = aie.buffer(%[[VAL_1]]) : memref<2xi32>
// CHECK:           %[[VAL_23:.*]] = aie.buffer(%[[VAL_1]]) : memref<2xi32>
// CHECK:           %[[VAL_24:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_25:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_26:.*]] = arith.constant 0 : index
// CHECK:             memref.store %[[VAL_25]], %[[VAL_23]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:             %[[VAL_27:.*]] = arith.constant 1 : index
// CHECK:             memref.store %[[VAL_25]], %[[VAL_23]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:             %[[VAL_28:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_29:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_30:.*]] = arith.constant 4 : i32
// CHECK:             memref.store %[[VAL_28]], %[[VAL_22]]{{\[}}%[[VAL_29]]] : memref<2xi32>
// CHECK:             %[[VAL_31:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_32:.*]] = arith.constant 3 : i32
// CHECK:             memref.store %[[VAL_28]], %[[VAL_22]]{{\[}}%[[VAL_31]]] : memref<2xi32>
// CHECK:             %[[VAL_33:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_34:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_35:.*]] = arith.constant 14 : index
// CHECK:             scf.for %[[VAL_36:.*]] = %[[VAL_33]] to %[[VAL_35]] step %[[VAL_34]] {
// CHECK:               %[[VAL_37:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:               %[[VAL_38:.*]] = arith.constant 3 : i32
// CHECK:               %[[VAL_39:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_40:.*]] = arith.subi %[[VAL_38]], %[[VAL_37]] : i32
// CHECK:               %[[VAL_41:.*]] = arith.maxsi %[[VAL_40]], %[[VAL_39]] : i32
// CHECK:               aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_41]])
// CHECK:               %[[VAL_42:.*]] = arith.addi %[[VAL_37]], %[[VAL_41]] : i32
// CHECK:               memref.store %[[VAL_42]], %[[VAL_23]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:               %[[VAL_43:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:               %[[VAL_44:.*]] = arith.constant 2 : i32
// CHECK:               %[[VAL_45:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_46:.*]] = arith.subi %[[VAL_44]], %[[VAL_43]] : i32
// CHECK:               %[[VAL_47:.*]] = arith.maxsi %[[VAL_46]], %[[VAL_45]] : i32
// CHECK:               aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_47]])
// CHECK:               %[[VAL_48:.*]] = arith.addi %[[VAL_43]], %[[VAL_47]] : i32
// CHECK:               memref.store %[[VAL_48]], %[[VAL_23]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:               %[[VAL_49:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_15]], Release, %[[VAL_49]])
// CHECK:               %[[VAL_50:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:               %[[VAL_51:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_52:.*]] = arith.subi %[[VAL_50]], %[[VAL_51]] : i32
// CHECK:               memref.store %[[VAL_52]], %[[VAL_23]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:               %[[VAL_53:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_29]]] : memref<2xi32>
// CHECK:               %[[VAL_54:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_55:.*]] = arith.addi %[[VAL_53]], %[[VAL_54]] : i32
// CHECK:               %[[VAL_56:.*]] = arith.cmpi sge, %[[VAL_55]], %[[VAL_30]] : i32
// CHECK:               %[[VAL_57:.*]] = arith.subi %[[VAL_55]], %[[VAL_30]] : i32
// CHECK:               %[[VAL_58:.*]] = arith.select %[[VAL_56]], %[[VAL_57]], %[[VAL_55]] : i32
// CHECK:               memref.store %[[VAL_58]], %[[VAL_22]]{{\[}}%[[VAL_29]]] : memref<2xi32>
// CHECK:               %[[VAL_59:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_5]], Release, %[[VAL_59]])
// CHECK:               %[[VAL_60:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:               %[[VAL_61:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_62:.*]] = arith.subi %[[VAL_60]], %[[VAL_61]] : i32
// CHECK:               memref.store %[[VAL_62]], %[[VAL_23]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:               %[[VAL_63:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_31]]] : memref<2xi32>
// CHECK:               %[[VAL_64:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_65:.*]] = arith.addi %[[VAL_63]], %[[VAL_64]] : i32
// CHECK:               %[[VAL_66:.*]] = arith.cmpi sge, %[[VAL_65]], %[[VAL_32]] : i32
// CHECK:               %[[VAL_67:.*]] = arith.subi %[[VAL_65]], %[[VAL_32]] : i32
// CHECK:               %[[VAL_68:.*]] = arith.select %[[VAL_66]], %[[VAL_67]], %[[VAL_65]] : i32
// CHECK:               memref.store %[[VAL_68]], %[[VAL_22]]{{\[}}%[[VAL_31]]] : memref<2xi32>
// CHECK:             }
// CHECK:             %[[VAL_69:.*]] = arith.constant 2 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_69]])
// CHECK:             %[[VAL_70:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:             %[[VAL_71:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_72:.*]] = arith.subi %[[VAL_70]], %[[VAL_71]] : i32
// CHECK:             memref.store %[[VAL_72]], %[[VAL_23]]{{\[}}%[[VAL_26]]] : memref<2xi32>
// CHECK:             %[[VAL_73:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_29]]] : memref<2xi32>
// CHECK:             %[[VAL_74:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_75:.*]] = arith.addi %[[VAL_73]], %[[VAL_74]] : i32
// CHECK:             %[[VAL_76:.*]] = arith.cmpi sge, %[[VAL_75]], %[[VAL_30]] : i32
// CHECK:             %[[VAL_77:.*]] = arith.subi %[[VAL_75]], %[[VAL_30]] : i32
// CHECK:             %[[VAL_78:.*]] = arith.select %[[VAL_76]], %[[VAL_77]], %[[VAL_75]] : i32
// CHECK:             memref.store %[[VAL_78]], %[[VAL_22]]{{\[}}%[[VAL_29]]] : memref<2xi32>
// CHECK:             %[[VAL_79:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_79]])
// CHECK:             %[[VAL_80:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:             %[[VAL_81:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_82:.*]] = arith.subi %[[VAL_80]], %[[VAL_81]] : i32
// CHECK:             memref.store %[[VAL_82]], %[[VAL_23]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:             %[[VAL_83:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_31]]] : memref<2xi32>
// CHECK:             %[[VAL_84:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_85:.*]] = arith.addi %[[VAL_83]], %[[VAL_84]] : i32
// CHECK:             %[[VAL_86:.*]] = arith.cmpi sge, %[[VAL_85]], %[[VAL_32]] : i32
// CHECK:             %[[VAL_87:.*]] = arith.subi %[[VAL_85]], %[[VAL_32]] : i32
// CHECK:             %[[VAL_88:.*]] = arith.select %[[VAL_86]], %[[VAL_87]], %[[VAL_85]] : i32
// CHECK:             memref.store %[[VAL_88]], %[[VAL_22]]{{\[}}%[[VAL_31]]] : memref<2xi32>
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_89:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_90:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_91:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_91]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_92:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_20]], Release, %[[VAL_92]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_93:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_93]])
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_94:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_20]], Release, %[[VAL_94]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_95:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_95]])
// CHECK:             aie.dma_bd(%[[VAL_19]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_96:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_20]], Release, %[[VAL_96]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %[[VAL_97:.*]] = aie.dma_start(MM2S, 1, ^bb5, ^bb7)
// CHECK:           ^bb5:
// CHECK:             %[[VAL_98:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_98]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_99:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_99]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             %[[VAL_100:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_100]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_101:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_101]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_102:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_103:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_104:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_104]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_105:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_105]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_106:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_106]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_107:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_107]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_108:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_108]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_109:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_109]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             %[[VAL_110:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_110]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_111:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_111]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             %[[VAL_112:.*]] = aie.dma_start(S2MM, 1, ^bb6, ^bb9)
// CHECK:           ^bb6:
// CHECK:             %[[VAL_113:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_113]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_114:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_114]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             %[[VAL_115:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_115]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_116:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_116]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             %[[VAL_117:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_117]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_118:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_118]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb9:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
// CHECK:       }

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifoX(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>
    aie.objectfifo @fifoY(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %x = aie.objectfifo.acquire @fifoX(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        %y = aie.objectfifo.acquire @fifoY(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifoX(Consume, 1)
        aie.objectfifo.release @fifoY(Consume, 1)
      }
      aie.objectfifo.release @fifoX(Consume, 2)
      aie.objectfifo.release @fifoY(Consume, 1)
      aie.end
    }
  }
}
