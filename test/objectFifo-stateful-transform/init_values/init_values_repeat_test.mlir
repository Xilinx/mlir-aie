//===- init_values_repeat_test.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_0"} : memref<2x2xi32> = dense<{{\[\[}}0, 1], [2, 3]]>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_1"} : memref<2x2xi32> = dense<{{\[\[}}4, 5], [6, 7]]>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 6 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "of0_cons_buff_0"} : memref<2x2xi32>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "of0_cons_buff_1"} : memref<2x2xi32>
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_5]]) {init = 2 : i32, sym_name = "of0_cons_prod_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_5]]) {init = 0 : i32, sym_name = "of0_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_5]], DMA : 0)
// CHECK:           %[[VAL_10:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_11:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_12:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb7)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_11]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_11]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_11]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_11]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_11]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_11]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_11]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_11]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_11]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_11]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_11]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_11]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = aie.mem(%[[VAL_5]]) {
// CHECK:             %[[VAL_14:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_15:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_14]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<2x2xi32> offset = 0 len = 4)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_14]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @init_repeat {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of0 (%tile12, {%tile13}, 2 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<2x2xi32>> = [dense<[[0, 1], [2, 3]]> : memref<2x2xi32>,
                                                                                                                     dense<[[4, 5], [6, 7]]> : memref<2x2xi32>]
 }
}
