//===- init_values_distribute_input_test.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "of2_cons_buff_0"} : memref<2xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "of2_cons_buff_1"} : memref<2xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_3]]) {init = 2 : i32, sym_name = "of2_cons_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_3]]) {init = 0 : i32, sym_name = "of2_cons_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of1_cons_buff_0"} : memref<2xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of1_cons_buff_1"} : memref<2xi32>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_2]]) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_2]]) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_0"} : memref<4xi32> = dense<[0, 1, 2, 3]>
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_1"} : memref<4xi32> = dense<[4, 5, 6, 7]>
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_0]]) {init = 2 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of0_cons_buff_0"} : memref<4xi32>
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of0_cons_buff_1"} : memref<4xi32>
// CHECK:           %[[VAL_18:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "of0_cons_prod_lock_0"}
// CHECK:           %[[VAL_19:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of0_cons_cons_lock_0"}
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "of0_cons_prod_lock_1"}
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of0_cons_cons_lock_1"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_3]], DMA : 0)
// CHECK:           %[[VAL_22:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_23:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_24:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<4xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<4xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_23]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_26:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_27:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<4xi32> offset = 0 len = 2)
// CHECK:             aie.use_lock(%[[VAL_19]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<4xi32> offset = 2 len = 2)
// CHECK:             aie.use_lock(%[[VAL_21]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<4xi32> offset = 0 len = 2)
// CHECK:             aie.use_lock(%[[VAL_19]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<4xi32> offset = 2 len = 2)
// CHECK:             aie.use_lock(%[[VAL_21]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             %[[VAL_28:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<4xi32> offset = 0 len = 2)
// CHECK:             aie.use_lock(%[[VAL_18]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<4xi32> offset = 0 len = 2)
// CHECK:             aie.use_lock(%[[VAL_18]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb8:
// CHECK:             %[[VAL_29:.*]] = aie.dma_start(MM2S, 1, ^bb9, ^bb11)
// CHECK:           ^bb9:
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<4xi32> offset = 2 len = 2)
// CHECK:             aie.use_lock(%[[VAL_20]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb10
// CHECK:           ^bb10:
// CHECK:             aie.use_lock(%[[VAL_21]], AcquireGreaterEqual, %[[VAL_26]])
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<4xi32> offset = 2 len = 2)
// CHECK:             aie.use_lock(%[[VAL_20]], Release, %[[VAL_26]])
// CHECK:             aie.next_bd ^bb9
// CHECK:           ^bb11:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_31:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_32:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_31]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<2xi32> offset = 0 len = 2)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_31]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_31]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<2xi32> offset = 0 len = 2)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_31]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_34:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_35:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_34]])
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<2xi32> offset = 0 len = 2)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_34]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_34]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<2xi32> offset = 0 len = 2)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_34]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @init_distribute_input {
 aie.device(xcve2302) {
    %tile13 = aie.tile(1, 3)
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile13, {%tile11}, 2 : i32) : !aie.objectfifo<memref<4xi32>> = [dense<[0, 1, 2, 3]> : memref<4xi32>,
                                                                                          dense<[4, 5, 6, 7]> : memref<4xi32>]
    aie.objectfifo @of1 (%tile11, {%tile12}, 2 : i32) : !aie.objectfifo<memref<2xi32>>
    aie.objectfifo @of2 (%tile11, {%tile23}, 2 : i32) : !aie.objectfifo<memref<2xi32>>

    aie.objectfifo.link [@of0] -> [@of1, @of2] ([] [0, 2])
 }
}
