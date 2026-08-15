//===- link_test_output_sizes.mlir -----------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "link1_cons_buff_0"} : memref<64xi32>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_1]]) {init = 1 : i32, sym_name = "link1_cons_prod_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "link1_cons_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]]) {init = 1 : i32, sym_name = "link1_cons_prod_lock_1"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "link1_cons_cons_lock_1"}
// CHECK:           %[[VAL_7:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "link2_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_7]]) {sym_name = "link2_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_7]]) {init = 2 : i32, sym_name = "link2_cons_prod_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_7]]) {init = 0 : i32, sym_name = "link2_cons_cons_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_12]]) {sym_name = "link3_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_12]]) {sym_name = "link3_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_12]]) {init = 2 : i32, sym_name = "link3_cons_prod_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_12]]) {init = 0 : i32, sym_name = "link3_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_7]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_12]], DMA : 0)
// CHECK:           aie.shim_dma_allocation @link1_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           %[[VAL_17:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_18:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_19:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<64xi32> offset = 0 len = 32)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<64xi32> offset = 32 len = 32)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_20:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb5)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<64xi32> offset = 0 len = 32)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_21:.*]] = aie.dma_start(MM2S, 1, ^bb6, ^bb7)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<64xi32> offset = 32 len = 32)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = aie.mem(%[[VAL_7]]) {
// CHECK:             %[[VAL_23:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_24:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = aie.mem(%[[VAL_12]]) {
// CHECK:             %[[VAL_26:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_27:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_distribute_output_sizes {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)

        aie.objectfifo @link1 (%tile20, {%tile21}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
        aie.objectfifo @link2 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @link3 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo.link [@link1] -> [@link2, @link3] ([][0, 32])
    }
}
