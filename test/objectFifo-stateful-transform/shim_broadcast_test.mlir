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

// CHECK-LABEL:   AIE.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(2, 3)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_4:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "of_in_0_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "of_in_0_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = AIE.lock(%[[VAL_1]], 0) {init = 2 : i32, sym_name = "of_in_0_cons_prod_lock"}
// CHECK:           %[[VAL_7:.*]] = AIE.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of_in_0_cons_cons_lock"}
// CHECK:           %[[VAL_8:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "of_in_1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "of_in_1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = AIE.lock(%[[VAL_2]], 0) {init = 2 : i32, sym_name = "of_in_1_cons_prod_lock"}
// CHECK:           %[[VAL_11:.*]] = AIE.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "of_in_1_cons_cons_lock"}
// CHECK:           %[[VAL_12:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = "of_in_2_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_13:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = "of_in_2_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_14:.*]] = AIE.lock(%[[VAL_3]], 0) {init = 2 : i32, sym_name = "of_in_2_cons_prod_lock"}
// CHECK:           %[[VAL_15:.*]] = AIE.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "of_in_2_cons_cons_lock"}
// CHECK:           %[[VAL_16:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 1 : i32, sym_name = "of_in_prod_lock"}
// CHECK:           %[[VAL_17:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "of_in_cons_lock"}
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           %[[VAL_18:.*]] = AIE.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
// CHECK:           AIE.shimDMAAllocation @of_in(MM2S, 0, 2)
// CHECK:           %[[VAL_19:.*]] = AIE.shimDMA(%[[VAL_0]]) {
// CHECK:             %[[VAL_20:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:             AIE.useLock(%[[VAL_17]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_18]] : memref<64xi32>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_16]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = AIE.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_22:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_4]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_5]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = AIE.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_24:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_8]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_9]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = AIE.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_26:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_14]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_12]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_15]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_14]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_13]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_15]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @shim_broadcast {
   AIE.device(xcve2302) {
      %tile20 = AIE.tile(2, 0)
      %tile22 = AIE.tile(2, 2)
      %tile23 = AIE.tile(2, 3)
      %tile33 = AIE.tile(3, 3)

      AIE.objectfifo @of_in (%tile20, {%tile22, %tile23, %tile33}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>

      %ext_buffer_in  = AIE.external_buffer {sym_name = "ext_buffer_in"}: memref<64xi32>
      AIE.objectfifo.register_external_buffers @of_in (%tile20, {%ext_buffer_in}) : (memref<64xi32>)
   }
}
