//===- link_test_AIE1.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: May 9th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           memref.global "public" @of2_cons : memref<16xi32>
// CHECK:           memref.global "public" @of2 : memref<16xi32>
// CHECK:           memref.global "public" @of1_cons : memref<16xi32>
// CHECK:           memref.global "public" @of1 : memref<16xi32>
// CHECK:           %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(2, 4)
// CHECK:           %[[VAL_3:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "of2_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "of2_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = AIE.lock(%[[VAL_2]], 0) {init = 0 : i32, sym_name = "of2_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = AIE.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "of2_cons_lock_1"}
// CHECK:           %[[VAL_7:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_8:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = AIE.lock(%[[VAL_1]], 0) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           %[[VAL_10:.*]] = AIE.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of1_cons_lock_1"}
// CHECK:           %[[VAL_11:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "of1_lock_0"}
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           %[[VAL_12:.*]] = AIE.external_buffer {sym_name = "ext_buff_in"} : memref<16xi32>
// CHECK:           AIE.shimDMAAllocation @of1(MM2S, 0, 2)
// CHECK:           %[[VAL_13:.*]] = AIE.shimDMA(%[[VAL_0]]) {
// CHECK:             %[[VAL_14:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:             AIE.useLock(%[[VAL_11]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_12]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 0)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = AIE.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_16:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_9]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_7]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_9]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_10]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_8]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_10]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_17:.*]] = AIE.dmaStart(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:             AIE.useLock(%[[VAL_9]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_7]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_9]], Release, 0)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             AIE.useLock(%[[VAL_10]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_8]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_10]], Release, 0)
// CHECK:             AIE.nextBd ^bb4
// CHECK:           ^bb6:  // pred: ^bb3
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = AIE.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_19:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_5]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_3]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_6]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_4]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @link_AIE1 {
    AIE.device(xcvc1902) {
        %tile20 = AIE.tile(2, 0)
        %tile22 = AIE.tile(2, 2)
        %tile24 = AIE.tile(2, 4)

        AIE.objectfifo @of1 (%tile20, {%tile22}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>
        AIE.objectfifo @of2 (%tile22, {%tile24}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>

        AIE.objectfifo.link [@of1] -> [@of2] ()

        %ext_buff_in = AIE.external_buffer {sym_name = "ext_buff_in"} : memref<16xi32>
        AIE.objectfifo.register_external_buffers @of1 (%tile20, {%ext_buff_in}) : (memref<16xi32>)
    }
}
