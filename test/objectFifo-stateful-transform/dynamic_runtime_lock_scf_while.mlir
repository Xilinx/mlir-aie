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

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 3 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "fifo_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "fifo_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "fifo_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "fifo_cons_buff_3"} : memref<8xi8>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_6]]) {init = 4 : i32, sym_name = "fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_6]]) {init = 0 : i32, sym_name = "fifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "buf"} : memref<1xindex>
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_6]], DMA : 0)
// CHECK:           %[[VAL_14:.*]] = aie.core(%[[VAL_6]]) {
// CHECK:             %[[VAL_15:.*]] = arith.constant 14 : index
// CHECK:             %[[VAL_16:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_17:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_19:.*]] = arith.constant 3 : i32
// CHECK:             %[[VAL_20:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_21:.*]] = arith.constant 4 : i32
// CHECK:             %[[VAL_22:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_23:.*]]:3 = scf.while (%[[VAL_24:.*]] = %[[VAL_17]], %[[VAL_25:.*]] = %[[VAL_18]], %[[VAL_26:.*]] = %[[VAL_18]]) : (index, i32, i32) -> (index, i32, i32) {
// CHECK:               %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_24]], %[[VAL_15]] : index
// CHECK:               scf.condition(%[[VAL_27]]) %[[VAL_24]], %[[VAL_25]], %[[VAL_26]] : index, i32, i32
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_28:.*]]: index, %[[VAL_29:.*]]: i32, %[[VAL_30:.*]]: i32):
// CHECK:               %[[VAL_31:.*]] = arith.subi %[[VAL_19]], %[[VAL_30]] : i32
// CHECK:               %[[VAL_32:.*]] = arith.maxsi %[[VAL_31]], %[[VAL_18]] : i32
// CHECK:               aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_32]])
// CHECK:               %[[VAL_33:.*]] = arith.addi %[[VAL_30]], %[[VAL_32]] : i32
// CHECK:               aie.use_lock(%[[VAL_11]], Release, %[[VAL_20]])
// CHECK:               %[[VAL_34:.*]] = arith.subi %[[VAL_33]], %[[VAL_20]] : i32
// CHECK:               %[[VAL_35:.*]] = arith.addi %[[VAL_29]], %[[VAL_20]] : i32
// CHECK:               %[[VAL_36:.*]] = arith.cmpi sge, %[[VAL_35]], %[[VAL_21]] : i32
// CHECK:               %[[VAL_37:.*]] = arith.select %[[VAL_36]], %[[VAL_18]], %[[VAL_35]] : i32
// CHECK:               %[[VAL_38:.*]] = arith.addi %[[VAL_28]], %[[VAL_16]] : index
// CHECK:               scf.yield %[[VAL_38]], %[[VAL_37]], %[[VAL_34]] : index, i32, i32
// CHECK:             }
// CHECK:             memref.store %[[VAL_39:.*]]#0, %[[VAL_13]]{{\[}}%[[VAL_17]]] : memref<1xindex>
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_22]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_40:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_41:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_42:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_41]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_41]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_41]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_41]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_41]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_41]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_43:.*]] = aie.mem(%[[VAL_6]]) {
// CHECK:             %[[VAL_44:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_45:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_44]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_44]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_44]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_44]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_44]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_44]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_44]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_44]])
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
        %x_obj0, %x_obj1, %x_obj2 = aie.objectfifo.acquire @fifo(Consume) : memref<8xi8>, memref<8xi8>, memref<8xi8>
        aie.objectfifo.release @fifo(Consume) [1]
        %next = arith.addi %arg1, %c1 : index
        scf.yield %next : index
      }
      memref.store %r, %buf[%c0] : memref<1xindex>
      aie.objectfifo.release @fifo(Consume) [2]
      aie.end
    }
  }
}
