//===- init_values_join_input_test.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of2_buff_0"} : memref<8xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of2_buff_1"} : memref<8xi32>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "of2_prod_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of2_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "of2_prod_lock_1"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of2_cons_lock_1"}
// CHECK:           %[[VAL_8:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_8]]) {sym_name = "of0_buff_0"} : memref<2x2xi32> = dense<{{\[\[}}0, 1], [2, 3]]>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_8]]) {sym_name = "of0_buff_1"} : memref<2x2xi32> = dense<{{\[\[}}4, 5], [6, 7]]>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_8]]) {init = 0 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_8]]) {init = 2 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:           %[[VAL_13:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_13]]) {sym_name = "of1_buff_0"} : memref<2x2xi32> = dense<{{\[\[}}0, 1], [2, 3]]>
// CHECK:           %[[VAL_15:.*]] = aie.buffer(%[[VAL_13]]) {sym_name = "of1_buff_1"} : memref<2x2xi32> = dense<{{\[\[}}4, 5], [6, 7]]>
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_13]]) {init = 0 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.lock(%[[VAL_13]]) {init = 2 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_8]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_13]], DMA : 0, %[[VAL_1]], DMA : 1)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           aie.shim_dma_allocation @of2_shim_alloc(%[[VAL_0]], S2MM, 0)
// CHECK:           %[[VAL_18:.*]] = aie.mem(%[[VAL_8]]) {
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_20:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_19]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_19]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_19]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_19]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_22:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_23:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_22]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_22]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_24:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi32> offset = 4 len = 4)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_22]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi32> offset = 4 len = 4)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_22]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             %[[VAL_25:.*]] = aie.dma_start(MM2S, 0, ^bb7, ^bb11)
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_22]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<8xi32> offset = 4 len = 4)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_22]])
// CHECK:             aie.next_bd ^bb9
// CHECK:           ^bb9:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_22]])
// CHECK:             aie.next_bd ^bb10
// CHECK:           ^bb10:
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_22]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<8xi32> offset = 4 len = 4)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_22]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb11:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = aie.mem(%[[VAL_13]]) {
// CHECK:             %[[VAL_27:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_28:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, %[[VAL_27]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_27]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, %[[VAL_27]])
// CHECK:             aie.dma_bd(%[[VAL_15]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_27]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @init_join_input {
 aie.device(xcve2302) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile12, {%tile11}, 2 : i32) : !aie.objectfifo<memref<2x2xi32>> = [dense<[[0, 1], [2, 3]]> : memref<2x2xi32>,
                                                                                            dense<[[4, 5], [6, 7]]> : memref<2x2xi32>]
    aie.objectfifo @of1 (%tile23, {%tile11}, 2 : i32) : !aie.objectfifo<memref<2x2xi32>> = [dense<[[0, 1], [2, 3]]> : memref<2x2xi32>,
                                                                                            dense<[[4, 5], [6, 7]]> : memref<2x2xi32>]
    aie.objectfifo @of2 (%tile11, {%tile10}, 2 : i32) : !aie.objectfifo<memref<8xi32>>

    aie.objectfifo.link [@of0, @of1] -> [@of2] ([0, 4] [])
 }
}
