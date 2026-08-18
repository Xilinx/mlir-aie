//===- via_DMA_test.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_shared_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_shared_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_0]]) {init = 2 : i32, sym_name = "of_shared_prod_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of_shared_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_stream_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of_stream_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]]) {init = 2 : i32, sym_name = "of_stream_prod_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of_stream_cons_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_9]]) {sym_name = "of_stream_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_9]]) {sym_name = "of_stream_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_9]]) {init = 2 : i32, sym_name = "of_stream_cons_prod_lock_0"}
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_9]]) {init = 0 : i32, sym_name = "of_stream_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_9]], DMA : 0)
// CHECK:           %[[VAL_14:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_15:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_16:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_15]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_15]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_15]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_15]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = aie.mem(%[[VAL_9]]) {
// CHECK:             %[[VAL_18:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_19:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @viaDMA {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of_shared (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_stream (%tile12, {%tile13}, 2 : i32) {via_DMA = true} : !aie.objectfifo<memref<16xi32>>
 }
}
