//===- shim_AIE2_test.mlir --------------------------*- MLIR -*-===//
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
// CHECK:           %[[VAL_0:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_2:.*]] = AIE.lock(%[[VAL_1]], 2) {init = 1 : i32, sym_name = "of_out_cons_prod_lock"}
// CHECK:           %[[VAL_3:.*]] = AIE.lock(%[[VAL_1]], 3) {init = 0 : i32, sym_name = "of_out_cons_cons_lock"}
// CHECK:           %[[VAL_4:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of_out_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of_out_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = AIE.lock(%[[VAL_0]], 2) {init = 2 : i32, sym_name = "of_out_prod_lock"}
// CHECK:           %[[VAL_7:.*]] = AIE.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "of_out_cons_lock"}
// CHECK:           %[[VAL_8:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of_in_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of_in_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 2 : i32, sym_name = "of_in_cons_prod_lock"}
// CHECK:           %[[VAL_11:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "of_in_cons_cons_lock"}
// CHECK:           %[[VAL_12:.*]] = AIE.lock(%[[VAL_1]], 0) {init = 1 : i32, sym_name = "of_in_prod_lock"}
// CHECK:           %[[VAL_13:.*]] = AIE.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of_in_cons_lock"}
// CHECK:           AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           %[[VAL_14:.*]] = AIE.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
// CHECK:           %[[VAL_15:.*]] = AIE.external_buffer {sym_name = "ext_buffer_out"} : memref<64xi32>
// CHECK:           AIE.shimDMAAllocation @of_in(MM2S, 0, 2)
// CHECK:           %[[VAL_16:.*]] = AIE.shimDMA(%[[VAL_1]]) {
// CHECK:             %[[VAL_17:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:             AIE.useLock(%[[VAL_13]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_14]] : memref<64xi32>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_12]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             %[[VAL_18:.*]] = AIE.dmaStart(S2MM, 0, ^bb3, ^bb4)
// CHECK:           ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK:             AIE.useLock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_15]] : memref<64xi32>, 0, 64>, 0)
// CHECK:             AIE.useLock(%[[VAL_3]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb4:  // pred: ^bb2
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           AIE.shimDMAAllocation @of_out(S2MM, 0, 2)
// CHECK:           %[[VAL_19:.*]] = AIE.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_20:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
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
// CHECK:             %[[VAL_21:.*]] = AIE.dmaStart(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:             AIE.useLock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_4]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             AIE.useLock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_5]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:             AIE.nextBd ^bb4
// CHECK:           ^bb6:  // pred: ^bb3
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @shim_AIE2 {
   AIE.device(xcve2302) {
      %tile22 = AIE.tile(2, 2)
      %tile20 = AIE.tile(2, 0)

      AIE.objectfifo @of_in (%tile20, {%tile22}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>
      AIE.objectfifo @of_out (%tile22, {%tile20}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>

      %ext_buffer_in  = AIE.external_buffer {sym_name = "ext_buffer_in"}: memref<64xi32>
      %ext_buffer_out  = AIE.external_buffer {sym_name = "ext_buffer_out"}: memref<64xi32>
      AIE.objectfifo.register_external_buffers @of_in (%tile20, {%ext_buffer_in}) : (memref<64xi32>)
      AIE.objectfifo.register_external_buffers @of_out (%tile20, {%ext_buffer_out}) : (memref<64xi32>)
   }
}
