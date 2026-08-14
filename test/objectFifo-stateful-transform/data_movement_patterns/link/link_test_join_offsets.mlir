//===- link_test_join_offsets.mlir -----------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_4:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "link4_buff_0"} : memref<48xi32>
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "link4_buff_1"} : memref<48xi32>
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "link4_prod_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "link4_cons_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "link4_prod_lock_1"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "link4_cons_lock_1"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "link4_prod_lock_2"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "link4_cons_lock_2"}
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "link3_buff_0"} : memref<12xi32>
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "link3_buff_1"} : memref<12xi32>
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_4]]) {init = 2 : i32, sym_name = "link3_prod_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_4]]) {init = 0 : i32, sym_name = "link3_cons_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "link2_buff_0"} : memref<20xi32>
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "link2_buff_1"} : memref<20xi32>
// CHECK:           %[[VAL_19:.*]] = aie.lock(%[[VAL_3]]) {init = 2 : i32, sym_name = "link2_prod_lock_0"}
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "link2_cons_lock_0"}
// CHECK:           %[[VAL_21:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "link1_buff_0"} : memref<4x4xi32>
// CHECK:           %[[VAL_22:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "link1_buff_1"} : memref<4x4xi32>
// CHECK:           %[[VAL_23:.*]] = aie.lock(%[[VAL_2]]) {init = 2 : i32, sym_name = "link1_prod_lock_0"}
// CHECK:           %[[VAL_24:.*]] = aie.lock(%[[VAL_2]]) {init = 0 : i32, sym_name = "link1_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_1]], DMA : 1)
// CHECK:           aie.flow(%[[VAL_4]], DMA : 0, %[[VAL_1]], DMA : 2)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           aie.shim_dma_allocation @link4_shim_alloc(%[[VAL_0]], S2MM, 0)
// CHECK:           %[[VAL_25:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_26:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_27:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_24]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_21]] : memref<4x4xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_23]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_24]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_22]] : memref<4x4xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_23]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_29:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_30:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<48xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_8]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<48xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_8]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_31:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<48xi32> offset = 16 len = 20)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<48xi32> offset = 16 len = 20)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             %[[VAL_32:.*]] = aie.dma_start(S2MM, 2, ^bb7, ^bb9)
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<48xi32> offset = 36 len = 12)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<48xi32> offset = 36 len = 12)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:
// CHECK:             %[[VAL_33:.*]] = aie.dma_start(MM2S, 0, ^bb10, ^bb16)
// CHECK:           ^bb10:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<48xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb11
// CHECK:           ^bb11:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<48xi32> offset = 16 len = 20)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb12
// CHECK:           ^bb12:
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<48xi32> offset = 36 len = 12)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb13
// CHECK:           ^bb13:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<48xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb14
// CHECK:           ^bb14:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<48xi32> offset = 16 len = 20)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb15
// CHECK:           ^bb15:
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_29]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<48xi32> offset = 36 len = 12)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_29]])
// CHECK:             aie.next_bd ^bb10
// CHECK:           ^bb16:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_35:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_36:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<20xi32> offset = 0 len = 20)
// CHECK:             aie.use_lock(%[[VAL_19]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<20xi32> offset = 0 len = 20)
// CHECK:             aie.use_lock(%[[VAL_19]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_38:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_39:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_38]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<12xi32> offset = 0 len = 12)
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_38]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_16]], AcquireGreaterEqual, %[[VAL_38]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<12xi32> offset = 0 len = 12)
// CHECK:             aie.use_lock(%[[VAL_15]], Release, %[[VAL_38]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_distribute_offsets {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)
        %tile33 = aie.tile(3, 3)

        aie.objectfifo @link1 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
        aie.objectfifo @link2 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
        aie.objectfifo @link3 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<12xi32>>
        aie.objectfifo @link4 (%tile21, {%tile20}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

        aie.objectfifo.link [@link1, @link2, @link3] -> [@link4] ([0, 16, 36][])
    }
}
