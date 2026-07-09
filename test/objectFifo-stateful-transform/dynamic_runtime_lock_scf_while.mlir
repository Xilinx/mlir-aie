//===- dynamic_runtime_lock_scf_while.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Acquire/release inside an scf.while body. The runtime held-counter approach
// needs no loop peeling or trip-count reasoning: the counter lives across
// iterations, so the while body lowers directly with a value-carrying
// `AcquireGreaterEqual`. A post-loop release decrements the counter.

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s | FileCheck %s

// CHECK-LABEL: module {
// CHECK:         aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifo_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifo_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifo_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifo_cons_buff_3"} : memref<8xi8>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]], 0) {init = 4 : i32, sym_name = "fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "fifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_0]], 0) {init = 3 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "buf"} : memref<1xindex>
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_1]]) : memref<1xi32>
// CHECK:           %[[VAL_15:.*]] = aie.buffer(%[[VAL_1]]) : memref<1xi32>
// CHECK:           %[[VAL_16:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_17:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : index
// CHECK:             memref.store %[[VAL_17]], %[[VAL_15]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:             %[[VAL_19:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_20:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_21:.*]] = arith.constant 4 : i32
// CHECK:             memref.store %[[VAL_19]], %[[VAL_14]]{{\[}}%[[VAL_20]]] : memref<1xi32>
// CHECK:             %[[VAL_22:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_23:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_24:.*]] = arith.constant 14 : index
// CHECK:             %[[VAL_25:.*]] = scf.while (%[[VAL_26:.*]] = %[[VAL_22]]) : (index) -> index {
// CHECK:               %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_24]] : index
// CHECK:               scf.condition(%[[VAL_27]]) %[[VAL_26]] : index
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_28:.*]]: index):
// CHECK:               %[[VAL_29:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:               %[[VAL_30:.*]] = arith.constant 3 : i32
// CHECK:               %[[VAL_31:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_32:.*]] = arith.subi %[[VAL_30]], %[[VAL_29]] : i32
// CHECK:               %[[VAL_33:.*]] = arith.maxsi %[[VAL_32]], %[[VAL_31]] : i32
// CHECK:               aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:               %[[VAL_34:.*]] = arith.addi %[[VAL_29]], %[[VAL_33]] : i32
// CHECK:               memref.store %[[VAL_34]], %[[VAL_15]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:               %[[VAL_35:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[VAL_35]])
// CHECK:               %[[VAL_36:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:               %[[VAL_37:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_38:.*]] = arith.subi %[[VAL_36]], %[[VAL_37]] : i32
// CHECK:               memref.store %[[VAL_38]], %[[VAL_15]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:               %[[VAL_39:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_20]]] : memref<1xi32>
// CHECK:               %[[VAL_40:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_41:.*]] = arith.addi %[[VAL_39]], %[[VAL_40]] : i32
// CHECK:               %[[VAL_42:.*]] = arith.cmpi sge, %[[VAL_41]], %[[VAL_21]] : i32
// CHECK:               %[[VAL_43:.*]] = arith.subi %[[VAL_41]], %[[VAL_21]] : i32
// CHECK:               %[[VAL_44:.*]] = arith.select %[[VAL_42]], %[[VAL_43]], %[[VAL_41]] : i32
// CHECK:               memref.store %[[VAL_44]], %[[VAL_14]]{{\[}}%[[VAL_20]]] : memref<1xi32>
// CHECK:               %[[VAL_45:.*]] = arith.addi %[[VAL_28]], %[[VAL_23]] : index
// CHECK:               scf.yield %[[VAL_45]] : index
// CHECK:             }
// CHECK:             memref.store %[[VAL_25]], %[[VAL_13]]{{\[}}%[[VAL_22]]] : memref<1xindex>
// CHECK:             %[[VAL_46:.*]] = arith.constant 2 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_46]])
// CHECK:             %[[VAL_47:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:             %[[VAL_48:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_49:.*]] = arith.subi %[[VAL_47]], %[[VAL_48]] : i32
// CHECK:             memref.store %[[VAL_49]], %[[VAL_15]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:             %[[VAL_50:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_20]]] : memref<1xi32>
// CHECK:             %[[VAL_51:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_52:.*]] = arith.addi %[[VAL_50]], %[[VAL_51]] : i32
// CHECK:             %[[VAL_53:.*]] = arith.cmpi sge, %[[VAL_52]], %[[VAL_21]] : i32
// CHECK:             %[[VAL_54:.*]] = arith.subi %[[VAL_52]], %[[VAL_21]] : i32
// CHECK:             %[[VAL_55:.*]] = arith.select %[[VAL_53]], %[[VAL_54]], %[[VAL_52]] : i32
// CHECK:             memref.store %[[VAL_55]], %[[VAL_14]]{{\[}}%[[VAL_20]]] : memref<1xi32>
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_56:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_57:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_58:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_58]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_59:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_59]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_60:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_60]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_61:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_61]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_62:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_62]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_63:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_63]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_64:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_65:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_66:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_66]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_67:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_67]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_68:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_68]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_69:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_69]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_70:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_70]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_71:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_71]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             %[[VAL_72:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_72]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_73:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_73]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
// CHECK:       }

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) {sym_name = "buf"} : memref<1xindex>

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %r = scf.while (%arg0 = %c0) : (index) -> index {
        %cond = arith.cmpi slt, %arg0, %c14 : index
        scf.condition(%cond) %arg0 : index
      } do {
      ^bb0(%arg1: index):
        %x = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifo(Consume, 1)
        %next = arith.addi %arg1, %c1 : index
        scf.yield %next : index
      }
      memref.store %r, %buf[%c0] : memref<1xindex>
      aie.objectfifo.release @fifo(Consume, 2)
      aie.end
    }
  }
}
