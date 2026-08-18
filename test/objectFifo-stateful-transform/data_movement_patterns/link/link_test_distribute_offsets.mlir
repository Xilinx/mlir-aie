//===- link_test_distribute_offsets.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: June 28th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "link1_cons_buff_0"} : memref<48xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "link1_cons_buff_1"} : memref<48xi32>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "link1_cons_prod_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "link1_cons_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "link1_cons_prod_lock_1"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "link1_cons_cons_lock_1"}
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "link1_cons_prod_lock_2"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "link1_cons_cons_lock_2"}
// CHECK:           %[[VAL_10:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_10]]) {sym_name = "link2_cons_buff_0"} : memref<4x4xi32>
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_10]]) {sym_name = "link2_cons_buff_1"} : memref<4x4xi32>
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_10]]) {init = 2 : i32, sym_name = "link2_cons_prod_lock_0"}
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_10]]) {init = 0 : i32, sym_name = "link2_cons_cons_lock_0"}
// CHECK:           %[[VAL_15:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_16:.*]] = aie.buffer(%[[VAL_15]]) {sym_name = "link3_cons_buff_0"} : memref<20xi32>
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_15]]) {sym_name = "link3_cons_buff_1"} : memref<20xi32>
// CHECK:           %[[VAL_18:.*]] = aie.lock(%[[VAL_15]]) {init = 2 : i32, sym_name = "link3_cons_prod_lock_0"}
// CHECK:           %[[VAL_19:.*]] = aie.lock(%[[VAL_15]]) {init = 0 : i32, sym_name = "link3_cons_cons_lock_0"}
// CHECK:           %[[VAL_20:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_21:.*]] = aie.buffer(%[[VAL_20]]) {sym_name = "link4_cons_buff_0"} : memref<12xi32>
// CHECK:           %[[VAL_22:.*]] = aie.buffer(%[[VAL_20]]) {sym_name = "link4_cons_buff_1"} : memref<12xi32>
// CHECK:           %[[VAL_23:.*]] = aie.lock(%[[VAL_20]]) {init = 2 : i32, sym_name = "link4_cons_prod_lock_0"}
// CHECK:           %[[VAL_24:.*]] = aie.lock(%[[VAL_20]]) {init = 0 : i32, sym_name = "link4_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_10]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_15]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 2, %[[VAL_20]], DMA : 0)
// CHECK:           aie.shim_dma_allocation @link1_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           %[[VAL_25:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_26:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_27:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<48xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<48xi32> offset = 16 len = 20)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<48xi32> offset = 36 len = 12)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<48xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<48xi32> offset = 16 len = 20)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<48xi32> offset = 36 len = 12)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb7:
// CHECK:             %[[VAL_28:.*]] = aie.dma_start(MM2S, 0, ^bb8, ^bb10)
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<48xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb9
// CHECK:           ^bb9:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<48xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb10:
// CHECK:             %[[VAL_29:.*]] = aie.dma_start(MM2S, 1, ^bb11, ^bb13)
// CHECK:           ^bb11:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<48xi32> offset = 16 len = 20)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb12
// CHECK:           ^bb12:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<48xi32> offset = 16 len = 20)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb11
// CHECK:           ^bb13:
// CHECK:             %[[VAL_30:.*]] = aie.dma_start(MM2S, 2, ^bb14, ^bb16)
// CHECK:           ^bb14:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<48xi32> offset = 36 len = 12)
// CHECK:             aie.use_lock(%[[VAL_8]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb15
// CHECK:           ^bb15:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<48xi32> offset = 36 len = 12)
// CHECK:             aie.use_lock(%[[VAL_8]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb14
// CHECK:           ^bb16:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = aie.mem(%[[VAL_10]]) {
// CHECK:             %[[VAL_32:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_33:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_32]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<4x4xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_32]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_32]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<4x4xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_32]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = aie.mem(%[[VAL_15]]) {
// CHECK:             %[[VAL_35:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_36:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<20xi32> offset = 0 len = 20)
// CHECK:             aie.use_lock(%[[VAL_19]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, %[[VAL_35]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<20xi32> offset = 0 len = 20)
// CHECK:             aie.use_lock(%[[VAL_19]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = aie.mem(%[[VAL_20]]) {
// CHECK:             %[[VAL_38:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_39:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_23]], AcquireGreaterEqual, %[[VAL_38]])
// CHECK:             aie.dma_bd(%[[VAL_21]] : memref<12xi32> offset = 0 len = 12)
// CHECK:             aie.use_lock(%[[VAL_24]], Release, %[[VAL_38]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_23]], AcquireGreaterEqual, %[[VAL_38]])
// CHECK:             aie.dma_bd(%[[VAL_22]] : memref<12xi32> offset = 0 len = 12)
// CHECK:             aie.use_lock(%[[VAL_24]], Release, %[[VAL_38]])
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

        aie.objectfifo @link1 (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
        aie.objectfifo @link2 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
        aie.objectfifo @link3 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
        aie.objectfifo @link4 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

        aie.objectfifo.link [@link1] -> [@link2, @link3, @link4] ([][0, 16, 36])
    }
}
