//===- link_test_broadcast_skip_connection.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: June 28th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "link1_cons_buff_0"} : memref<48xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "link1_cons_buff_1"} : memref<48xi32>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "link1_cons_prod_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "link1_cons_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "link2_0_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "link2_0_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_6]]) {init = 2 : i32, sym_name = "link2_0_cons_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_6]]) {init = 0 : i32, sym_name = "link2_0_cons_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "skip_connection_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "skip_connection_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_6]]) {init = 2 : i32, sym_name = "skip_connection_prod_lock_0"}
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_6]]) {init = 0 : i32, sym_name = "skip_connection_cons_lock_0"}
// CHECK:           %[[VAL_15:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_16:.*]] = aie.buffer(%[[VAL_15]]) {sym_name = "link2_1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_15]]) {sym_name = "link2_1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_15]]) {sym_name = "link2_1_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_19:.*]] = aie.lock(%[[VAL_15]]) {init = 3 : i32, sym_name = "link2_1_cons_prod_lock_0"}
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_15]]) {init = 0 : i32, sym_name = "link2_1_cons_cons_lock_0"}
// CHECK:           %[[VAL_21:.*]] = aie.buffer(%[[VAL_15]]) {sym_name = "skip_connection_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_22:.*]] = aie.buffer(%[[VAL_15]]) {sym_name = "skip_connection_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_23:.*]] = aie.lock(%[[VAL_15]]) {init = 2 : i32, sym_name = "skip_connection_cons_prod_lock_0"}
// CHECK:           %[[VAL_24:.*]] = aie.lock(%[[VAL_15]]) {init = 0 : i32, sym_name = "skip_connection_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_6]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_15]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_6]], DMA : 0, %[[VAL_15]], DMA : 1)
// CHECK:           aie.shim_dma_allocation @link1_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           %[[VAL_25:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_26:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_27:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<48xi32> offset = 0 len = 48)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<48xi32> offset = 0 len = 48)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_28:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<48xi32> offset = 0 len = 48)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<48xi32> offset = 0 len = 48)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = aie.mem(%[[VAL_6]]) {
// CHECK:             %[[VAL_30:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_31:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_30]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_30]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_32:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[VAL_30]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[VAL_30]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = aie.mem(%[[VAL_15]]) {
// CHECK:             %[[VAL_34:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_35:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, %[[VAL_34]])
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_20]], Release, %[[VAL_34]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, %[[VAL_34]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_20]], Release, %[[VAL_34]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, %[[VAL_34]])
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_20]], Release, %[[VAL_34]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             %[[VAL_36:.*]] = aie.dma_start(S2MM, 1, ^bb5, ^bb7)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_23]], AcquireGreaterEqual, %[[VAL_34]])
// CHECK:             aie.dma_bd(%[[VAL_21]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_24]], Release, %[[VAL_34]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_23]], AcquireGreaterEqual, %[[VAL_34]])
// CHECK:             aie.dma_bd(%[[VAL_22]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_24]], Release, %[[VAL_34]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_test_broadcast_skip_connection {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile33 = aie.tile(3, 3)

        aie.objectfifo @link1 (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
        aie.objectfifo @link2 (%tile21, {%tile22, %tile33}, [2, 2, 3]) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo @skip_connection (%tile22, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo.link [@link1] -> [@link2] ([] [])
    }
}
