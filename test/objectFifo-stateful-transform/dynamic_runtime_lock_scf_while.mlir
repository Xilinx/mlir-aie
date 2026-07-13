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

// CHECK-LABEL:   aie.device(npu2) {
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
// CHECK:           %[[VAL_15:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_16:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_17:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 4 : i32
// CHECK:             memref.store %[[VAL_17]], %[[VAL_14]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:             %[[VAL_20:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_21:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_22:.*]] = arith.constant 14 : index
// CHECK:             %[[VAL_23:.*]]:2 = scf.while (%[[VAL_24:.*]] = %[[VAL_20]], %[[VAL_25:.*]] = %[[VAL_16]]) : (index, i32) -> (index, i32) {
// CHECK:               %[[VAL_26:.*]] = arith.cmpi slt, %[[VAL_24]], %[[VAL_22]] : index
// CHECK:               scf.condition(%[[VAL_26]]) %[[VAL_24]], %[[VAL_25]] : index, i32
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_27:.*]]: index, %[[VAL_28:.*]]: i32):
// CHECK:               %[[VAL_29:.*]] = arith.constant 3 : i32
// CHECK:               %[[VAL_30:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_31:.*]] = arith.subi %[[VAL_29]], %[[VAL_28]] : i32
// CHECK:               %[[VAL_32:.*]] = arith.maxsi %[[VAL_31]], %[[VAL_30]] : i32
// CHECK:               aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_32]])
// CHECK:               %[[VAL_33:.*]] = arith.addi %[[VAL_28]], %[[VAL_32]] : i32
// CHECK:               %[[VAL_34:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[VAL_34]])
// CHECK:               %[[VAL_35:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_36:.*]] = arith.subi %[[VAL_33]], %[[VAL_35]] : i32
// CHECK:               %[[VAL_37:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:               %[[VAL_38:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_39:.*]] = arith.addi %[[VAL_37]], %[[VAL_38]] : i32
// CHECK:               %[[VAL_40:.*]] = arith.cmpi sge, %[[VAL_39]], %[[VAL_19]] : i32
// CHECK:               %[[VAL_41:.*]] = arith.subi %[[VAL_39]], %[[VAL_19]] : i32
// CHECK:               %[[VAL_42:.*]] = arith.select %[[VAL_40]], %[[VAL_41]], %[[VAL_39]] : i32
// CHECK:               memref.store %[[VAL_42]], %[[VAL_14]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:               %[[VAL_43:.*]] = arith.addi %[[VAL_27]], %[[VAL_21]] : index
// CHECK:               scf.yield %[[VAL_43]], %[[VAL_36]] : index, i32
// CHECK:             }
// CHECK:             memref.store %[[VAL_44:.*]]#0, %[[VAL_13]]{{\[}}%[[VAL_20]]] : memref<1xindex>
// CHECK:             %[[VAL_45:.*]] = arith.constant 2 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_45]])
// CHECK:             %[[VAL_46:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_47:.*]] = arith.subi %[[VAL_44]]#1, %[[VAL_46]] : i32
// CHECK:             %[[VAL_48:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:             %[[VAL_49:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_50:.*]] = arith.addi %[[VAL_48]], %[[VAL_49]] : i32
// CHECK:             %[[VAL_51:.*]] = arith.cmpi sge, %[[VAL_50]], %[[VAL_19]] : i32
// CHECK:             %[[VAL_52:.*]] = arith.subi %[[VAL_50]], %[[VAL_19]] : i32
// CHECK:             %[[VAL_53:.*]] = arith.select %[[VAL_51]], %[[VAL_52]], %[[VAL_50]] : i32
// CHECK:             memref.store %[[VAL_53]], %[[VAL_14]]{{\[}}%[[VAL_18]]] : memref<1xi32>
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_55:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_56:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_56]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_57:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_57]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_58:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_58]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_59:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_59]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_60:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_60]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_61:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_61]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_62:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_63:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_64:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_64]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_65:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_65]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             %[[VAL_66:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_66]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_67:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_67]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_68:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_68]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_69:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_69]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             %[[VAL_70:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_70]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<8xi8>, 0, 8)
// CHECK:             %[[VAL_71:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_71]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

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
