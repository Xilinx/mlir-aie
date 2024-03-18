//===- shim_broadcast_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: July 3rd 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_2:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_in_0_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_in_0_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]], 0) {init = 2 : i32, sym_name = "of_in_0_cons_prod_lock"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of_in_0_cons_cons_lock"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of_in_1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of_in_1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_2]], 0) {init = 2 : i32, sym_name = "of_in_1_cons_prod_lock"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "of_in_1_cons_cons_lock"}
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "of_in_2_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "of_in_2_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_3]], 0) {init = 2 : i32, sym_name = "of_in_2_cons_prod_lock"}
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "of_in_2_cons_cons_lock"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_0]], 0) {init = 1 : i32, sym_name = "of_in_prod_lock"}
// CHECK:           %[[VAL_17:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "of_in_cons_lock"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           %[[VAL_18:.*]] = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
// CHECK:           aie.shim_dma_allocation @of_in(MM2S, 0, 2)
// CHECK:           %[[VAL_19:.*]] = aie.shim_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_20:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:             aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<64xi32>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_16]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_22:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_24:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_11]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_11]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_26:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_15]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_15]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @shim_broadcast {
   aie.device(xcve2302) {
      %tile20 = aie.tile(2, 0)
      %tile22 = aie.tile(2, 2)
      %tile23 = aie.tile(2, 3)
      %tile33 = aie.tile(3, 3)

      aie.objectfifo @of_in (%tile20, {%tile22, %tile23, %tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

      %ext_buffer_in  = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<64xi32>
      aie.objectfifo.register_external_buffers @of_in (%tile20, {%ext_buffer_in}) : (memref<64xi32>)
   }
}
