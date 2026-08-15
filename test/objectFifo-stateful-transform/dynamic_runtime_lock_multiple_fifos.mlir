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

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll="default-dynamic=true" %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifoX_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifoX_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifoX_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 3 : i32, sym_name = "fifoX_prod_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "fifoX_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifoY_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifoY_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_0]]) {init = 2 : i32, sym_name = "fifoY_prod_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "fifoY_cons_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_10]]) {sym_name = "fifoX_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_10]]) {sym_name = "fifoX_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_10]]) {sym_name = "fifoX_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_10]]) {sym_name = "fifoX_cons_buff_3"} : memref<8xi8>
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_10]]) {init = 4 : i32, sym_name = "fifoX_cons_prod_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_10]]) {init = 0 : i32, sym_name = "fifoX_cons_cons_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_10]]) {sym_name = "fifoY_cons_buff_0"} : memref<8xi8>
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_10]]) {sym_name = "fifoY_cons_buff_1"} : memref<8xi8>
// CHECK:           %[[VAL_19:.*]] = aie.buffer(%[[VAL_10]]) {sym_name = "fifoY_cons_buff_2"} : memref<8xi8>
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_10]]) {init = 3 : i32, sym_name = "fifoY_cons_prod_lock_0"}
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_10]]) {init = 0 : i32, sym_name = "fifoY_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_10]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 1, %[[VAL_10]], DMA : 1)
// CHECK:           %[[VAL_22:.*]] = aie.core(%[[VAL_10]]) {
// CHECK:             %[[VAL_23:.*]] = arith.constant 14 : index
// CHECK:             %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_25:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_26:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_27:.*]] = arith.constant 3 : i32
// CHECK:             %[[VAL_28:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_29:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_30:.*]] = arith.constant 4 : i32
// CHECK:             %[[VAL_31:.*]]:4 = scf.for %[[VAL_32:.*]] = %[[VAL_25]] to %[[VAL_23]] step %[[VAL_24]] iter_args(%[[VAL_33:.*]] = %[[VAL_26]], %[[VAL_34:.*]] = %[[VAL_26]], %[[VAL_35:.*]] = %[[VAL_26]], %[[VAL_36:.*]] = %[[VAL_26]]) -> (i32, i32, i32, i32) {
// CHECK:               %[[VAL_37:.*]] = arith.subi %[[VAL_27]], %[[VAL_35]] : i32
// CHECK:               %[[VAL_38:.*]] = arith.maxsi %[[VAL_37]], %[[VAL_26]] : i32
// CHECK:               aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_38]])
// CHECK:               %[[VAL_39:.*]] = arith.addi %[[VAL_35]], %[[VAL_38]] : i32
// CHECK:               %[[VAL_40:.*]] = arith.subi %[[VAL_28]], %[[VAL_36]] : i32
// CHECK:               %[[VAL_41:.*]] = arith.maxsi %[[VAL_40]], %[[VAL_26]] : i32
// CHECK:               aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_41]])
// CHECK:               %[[VAL_42:.*]] = arith.addi %[[VAL_36]], %[[VAL_41]] : i32
// CHECK:               aie.use_lock(%[[VAL_15]], Release, %[[VAL_29]])
// CHECK:               %[[VAL_43:.*]] = arith.subi %[[VAL_39]], %[[VAL_29]] : i32
// CHECK:               %[[VAL_44:.*]] = arith.addi %[[VAL_33]], %[[VAL_29]] : i32
// CHECK:               %[[VAL_45:.*]] = arith.cmpi sge, %[[VAL_44]], %[[VAL_30]] : i32
// CHECK:               %[[VAL_46:.*]] = arith.select %[[VAL_45]], %[[VAL_26]], %[[VAL_44]] : i32
// CHECK:               aie.use_lock(%[[VAL_20]], Release, %[[VAL_29]])
// CHECK:               %[[VAL_47:.*]] = arith.subi %[[VAL_42]], %[[VAL_29]] : i32
// CHECK:               %[[VAL_48:.*]] = arith.addi %[[VAL_34]], %[[VAL_29]] : i32
// CHECK:               %[[VAL_49:.*]] = arith.cmpi sge, %[[VAL_48]], %[[VAL_27]] : i32
// CHECK:               %[[VAL_50:.*]] = arith.select %[[VAL_49]], %[[VAL_26]], %[[VAL_48]] : i32
// CHECK:               scf.yield %[[VAL_46]], %[[VAL_50]], %[[VAL_43]], %[[VAL_47]] : i32, i32, i32, i32
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_28]])
// CHECK:             aie.use_lock(%[[VAL_20]], Release, %[[VAL_29]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_51:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_52:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_53:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_52]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_52]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_52]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_52]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_52]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_52]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %[[VAL_54:.*]] = aie.dma_start(MM2S, 1, ^bb5, ^bb7)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_52]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_8]], Release, %[[VAL_52]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_52]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_8]], Release, %[[VAL_52]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_55:.*]] = aie.mem(%[[VAL_10]]) {
// CHECK:             %[[VAL_56:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_57:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_56]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_56]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_56]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_56]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_56]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_56]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_56]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_56]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             %[[VAL_58:.*]] = aie.dma_start(S2MM, 1, ^bb6, ^bb9)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, %[[VAL_56]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_21]], Release, %[[VAL_56]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, %[[VAL_56]])
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_21]], Release, %[[VAL_56]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, %[[VAL_56]])
// CHECK:             aie.dma_bd(%[[VAL_19]] : memref<8xi8> offset = 0 len = 8)
// CHECK:             aie.use_lock(%[[VAL_21]], Release, %[[VAL_56]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb9:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

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
        %x_obj0, %x_obj1, %x_obj2 = aie.objectfifo.acquire @fifoX(Consume) : memref<8xi8>, memref<8xi8>, memref<8xi8>
        %y_obj0, %y_obj1 = aie.objectfifo.acquire @fifoY(Consume) : memref<8xi8>, memref<8xi8>
        aie.objectfifo.release @fifoX(Consume) [1]
        aie.objectfifo.release @fifoY(Consume) [1]
      }
      aie.objectfifo.release @fifoX(Consume) [2]
      aie.objectfifo.release @fifoY(Consume) [1]
      aie.end
    }
  }
}
