//===- dynamic_runtime_lock_nested_loops.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Acquires/releases at both the outer and inner loop levels on two fifos.
// The runtime held counters are shared across the loop nest, so the inner
// loop's per-iteration acquire delta and the outer loop's acquire delta are
// each computed from the current held value regardless of loop depth.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK-LABEL: module {
// CHECK:         aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inOF_X_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inOF_X_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inOF_X_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inOF_X_cons_buff_3"} : memref<8xi8>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]], 2) {init = 4 : i32, sym_name = "inOF_X_cons_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_1]], 3) {init = 0 : i32, sym_name = "inOF_X_cons_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_X_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_X_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_X_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_0]], 2) {init = 3 : i32, sym_name = "inOF_X_prod_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "inOF_X_cons_lock_0"}
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inOF_W_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inOF_W_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_15:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "inOF_W_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_1]], 0) {init = 3 : i32, sym_name = "inOF_W_cons_prod_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "inOF_W_cons_cons_lock_0"}
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_W_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_19:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_W_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_20:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "inOF_W_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_0]], 0) {init = 3 : i32, sym_name = "inOF_W_prod_lock_0"}
// CHECK:           %[[VAL_22:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "inOF_W_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 1, %[[VAL_1]], DMA : 1)
// CHECK:           %[[VAL_23:.*]] = aie.buffer(%[[VAL_1]]) : memref<2xi32>
// CHECK:           %[[VAL_24:.*]] = aie.buffer(%[[VAL_1]]) : memref<2xi32>
// CHECK:           %[[VAL_25:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_26:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_27:.*]] = arith.constant 0 : index
// CHECK:             memref.store %[[VAL_26]], %[[VAL_24]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:             %[[VAL_28:.*]] = arith.constant 1 : index
// CHECK:             memref.store %[[VAL_26]], %[[VAL_24]]{{\[}}%[[VAL_28]]] : memref<2xi32>
// CHECK:             %[[VAL_29:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_30:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_31:.*]] = arith.constant 3 : i32
// CHECK:             memref.store %[[VAL_29]], %[[VAL_23]]{{\[}}%[[VAL_30]]] : memref<2xi32>
// CHECK:             %[[VAL_32:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_33:.*]] = arith.constant 4 : i32
// CHECK:             memref.store %[[VAL_29]], %[[VAL_23]]{{\[}}%[[VAL_32]]] : memref<2xi32>
// CHECK:             %[[VAL_34:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_35:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_36:.*]] = arith.constant 14 : index
// CHECK:             scf.for %[[VAL_37:.*]] = %[[VAL_34]] to %[[VAL_36]] step %[[VAL_35]] {
// CHECK:               %[[VAL_38:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:               %[[VAL_39:.*]] = arith.constant 2 : i32
// CHECK:               %[[VAL_40:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_41:.*]] = arith.subi %[[VAL_39]], %[[VAL_38]] : i32
// CHECK:               %[[VAL_42:.*]] = arith.maxsi %[[VAL_41]], %[[VAL_40]] : i32
// CHECK:               aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, %[[VAL_42]])
// CHECK:               %[[VAL_43:.*]] = arith.addi %[[VAL_38]], %[[VAL_42]] : i32
// CHECK:               memref.store %[[VAL_43]], %[[VAL_24]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:               scf.for %[[VAL_44:.*]] = %[[VAL_34]] to %[[VAL_36]] step %[[VAL_35]] {
// CHECK:                 %[[VAL_45:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_28]]] : memref<2xi32>
// CHECK:                 %[[VAL_46:.*]] = arith.constant 3 : i32
// CHECK:                 %[[VAL_47:.*]] = arith.constant 0 : i32
// CHECK:                 %[[VAL_48:.*]] = arith.subi %[[VAL_46]], %[[VAL_45]] : i32
// CHECK:                 %[[VAL_49:.*]] = arith.maxsi %[[VAL_48]], %[[VAL_47]] : i32
// CHECK:                 aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_49]])
// CHECK:                 %[[VAL_50:.*]] = arith.addi %[[VAL_45]], %[[VAL_49]] : i32
// CHECK:                 memref.store %[[VAL_50]], %[[VAL_24]]{{\[}}%[[VAL_28]]] : memref<2xi32>
// CHECK:                 %[[VAL_51:.*]] = arith.constant 1 : i32
// CHECK:                 aie.use_lock(%[[VAL_6]], Release, %[[VAL_51]])
// CHECK:                 %[[VAL_52:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_28]]] : memref<2xi32>
// CHECK:                 %[[VAL_53:.*]] = arith.constant 1 : i32
// CHECK:                 %[[VAL_54:.*]] = arith.subi %[[VAL_52]], %[[VAL_53]] : i32
// CHECK:                 memref.store %[[VAL_54]], %[[VAL_24]]{{\[}}%[[VAL_28]]] : memref<2xi32>
// CHECK:                 %[[VAL_55:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_32]]] : memref<2xi32>
// CHECK:                 %[[VAL_56:.*]] = arith.constant 1 : i32
// CHECK:                 %[[VAL_57:.*]] = arith.addi %[[VAL_55]], %[[VAL_56]] : i32
// CHECK:                 %[[VAL_58:.*]] = arith.cmpi sge, %[[VAL_57]], %[[VAL_33]] : i32
// CHECK:                 %[[VAL_59:.*]] = arith.subi %[[VAL_57]], %[[VAL_33]] : i32
// CHECK:                 %[[VAL_60:.*]] = arith.select %[[VAL_58]], %[[VAL_59]], %[[VAL_57]] : i32
// CHECK:                 memref.store %[[VAL_60]], %[[VAL_23]]{{\[}}%[[VAL_32]]] : memref<2xi32>
// CHECK:               }
// CHECK:               %[[VAL_61:.*]] = arith.constant 2 : i32
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[VAL_61]])
// CHECK:               %[[VAL_62:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_28]]] : memref<2xi32>
// CHECK:               %[[VAL_63:.*]] = arith.constant 2 : i32
// CHECK:               %[[VAL_64:.*]] = arith.subi %[[VAL_62]], %[[VAL_63]] : i32
// CHECK:               memref.store %[[VAL_64]], %[[VAL_24]]{{\[}}%[[VAL_28]]] : memref<2xi32>
// CHECK:               %[[VAL_65:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_32]]] : memref<2xi32>
// CHECK:               %[[VAL_66:.*]] = arith.constant 2 : i32
// CHECK:               %[[VAL_67:.*]] = arith.addi %[[VAL_65]], %[[VAL_66]] : i32
// CHECK:               %[[VAL_68:.*]] = arith.cmpi sge, %[[VAL_67]], %[[VAL_33]] : i32
// CHECK:               %[[VAL_69:.*]] = arith.subi %[[VAL_67]], %[[VAL_33]] : i32
// CHECK:               %[[VAL_70:.*]] = arith.select %[[VAL_68]], %[[VAL_69]], %[[VAL_67]] : i32
// CHECK:               memref.store %[[VAL_70]], %[[VAL_23]]{{\[}}%[[VAL_32]]] : memref<2xi32>
// CHECK:               %[[VAL_71:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_16]], Release, %[[VAL_71]])
// CHECK:               %[[VAL_72:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:               %[[VAL_73:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_74:.*]] = arith.subi %[[VAL_72]], %[[VAL_73]] : i32
// CHECK:               memref.store %[[VAL_74]], %[[VAL_24]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:               %[[VAL_75:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_30]]] : memref<2xi32>
// CHECK:               %[[VAL_76:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_77:.*]] = arith.addi %[[VAL_75]], %[[VAL_76]] : i32
// CHECK:               %[[VAL_78:.*]] = arith.cmpi sge, %[[VAL_77]], %[[VAL_31]] : i32
// CHECK:               %[[VAL_79:.*]] = arith.subi %[[VAL_77]], %[[VAL_31]] : i32
// CHECK:               %[[VAL_80:.*]] = arith.select %[[VAL_78]], %[[VAL_79]], %[[VAL_77]] : i32
// CHECK:               memref.store %[[VAL_80]], %[[VAL_23]]{{\[}}%[[VAL_30]]] : memref<2xi32>
// CHECK:             }
// CHECK:             %[[VAL_81:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_81]])
// CHECK:             %[[VAL_82:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:             %[[VAL_83:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_84:.*]] = arith.subi %[[VAL_82]], %[[VAL_83]] : i32
// CHECK:             memref.store %[[VAL_84]], %[[VAL_24]]{{\[}}%[[VAL_27]]] : memref<2xi32>
// CHECK:             %[[VAL_85:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_30]]] : memref<2xi32>
// CHECK:             %[[VAL_86:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_87:.*]] = arith.addi %[[VAL_85]], %[[VAL_86]] : i32
// CHECK:             %[[VAL_88:.*]] = arith.cmpi sge, %[[VAL_87]], %[[VAL_31]] : i32
// CHECK:             %[[VAL_89:.*]] = arith.subi %[[VAL_87]], %[[VAL_31]] : i32
// CHECK:             %[[VAL_90:.*]] = arith.select %[[VAL_88]], %[[VAL_89]], %[[VAL_87]] : i32
// CHECK:             memref.store %[[VAL_90]], %[[VAL_23]]{{\[}}%[[VAL_30]]] : memref<2xi32>
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_91:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_92:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_93:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, %[[VAL_93]])
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_94:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], Release, %[[VAL_94]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_95:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, %[[VAL_95]])
// CHECK:             aie.dma_bd(%[[VAL_19]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_96:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], Release, %[[VAL_96]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_97:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, %[[VAL_97]])
// CHECK:             aie.dma_bd(%[[VAL_20]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_98:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_21]], Release, %[[VAL_98]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %[[VAL_99:.*]] = aie.dma_start(MM2S, 1, ^bb5, ^bb8)
// CHECK:           ^bb5:
// CHECK:             %[[VAL_100:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_100]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_101:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_101]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             %[[VAL_102:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_102]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_103:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_103]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             %[[VAL_104:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_104]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_105:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_105]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb8:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_106:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_107:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_108:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_108]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_109:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_109]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_110:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_110]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_111:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_111]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_112:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_112]])
// CHECK:             aie.dma_bd(%[[VAL_15]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_113:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_113]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %[[VAL_114:.*]] = aie.dma_start(S2MM, 1, ^bb5, ^bb9)
// CHECK:           ^bb5:
// CHECK:             %[[VAL_115:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_115]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_116:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_116]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             %[[VAL_117:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_117]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_118:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_118]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             %[[VAL_119:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_119]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_120:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_120]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             %[[VAL_121:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_121]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_122:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_122]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb9:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
// CHECK:       }

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @inOF_W(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>
    aie.objectfifo @inOF_X(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %w = aie.objectfifo.acquire @inOF_W(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
        scf.for %arg1 = %c0 to %c14 step %c1 {
          %x = aie.objectfifo.acquire @inOF_X(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
          aie.objectfifo.release @inOF_X(Consume, 1)
        }
        aie.objectfifo.release @inOF_X(Consume, 2)
        aie.objectfifo.release @inOF_W(Consume, 1)
      }
      aie.objectfifo.release @inOF_W(Consume, 1)
      aie.end
    }
  }
}
