//===- init_values_test.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_0"} : memref<2x2xi32> = dense<{{\[\[}}0, 1], [2, 3]]>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_1"} : memref<2x2xi32> = dense<{{\[\[}}4, 5], [6, 7]]>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_2"} : memref<2x2xi32> = dense<{{\[\[}}8, 9], [10, 11]]>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]]) {init = 3 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "of0_cons_buff_0"} : memref<2x2xi32>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "of0_cons_buff_1"} : memref<2x2xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "of0_cons_buff_2"} : memref<2x2xi32>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_6]]) {init = 3 : i32, sym_name = "of0_cons_prod_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_6]]) {init = 0 : i32, sym_name = "of0_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_6]], DMA : 0)
// CHECK:           %[[VAL_12:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_13:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_14:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_13]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_13]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_13]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_13]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_13]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_13]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = aie.mem(%[[VAL_6]]) {
// CHECK:             %[[VAL_16:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_17:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_16]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_16]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_16]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_16]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_16]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_16]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @init {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile12, {%tile23}, 3 : i32) : !aie.objectfifo<memref<2x2xi32>> = [dense<[[0, 1], [2, 3]]> : memref<2x2xi32>,
                                                                                            dense<[[4, 5], [6, 7]]> : memref<2x2xi32>,
                                                                                            dense<[[8, 9], [10, 11]]> : memref<2x2xi32>]
 }
}
